#!/usr/bin/env python3
"""
Decentralized Learning Simulator with Trust-Weighted Aggregation

Supports four aggregation strategies over a peer graph:
  1) Decentralized FedAvg (synchronous neighbor averaging per round)
  2) Gossip averaging (asynchronous-style, simulated with K random edge gossips per round)
  3) Decentralized Krum (Byzantine-robust aggregation, selects most similar model)
  4) Trust-weighted variants (FedAvg and Gossip with gradient-based trust monitoring)

Trust-weighted aggregation uses lightweight gradient fingerprinting to detect
anomalous behavior and weights neighbor contributions based on trust scores.

Example usage:
  # Clean run with trust-weighted FedAvg
  python decentralized_fl_sim.py \
      --dataset femnist --num-nodes 8 --rounds 20 --local-epochs 1 \
      --agg trust-fedavg --graph ring --lr 0.01

  # Trust-weighted gossip under attack
  python decentralized_fl_sim.py \
      --dataset celeba --num-nodes 10 --rounds 30 --local-epochs 1 \
      --agg trust-gossip --gossip-steps 50 --graph erdos --p 0.3 \
      --attack-percentage 0.3 --attack-type directed_deviation
"""
from __future__ import annotations

import argparse
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, RandomSampler
from scipy import stats as scipy_stats

from leaf_datasets import (
    load_leaf_dataset,
    create_leaf_client_partitions,
    LEAFFEMNISTModel,
    LEAFCelebAModel
)


# ---------------------------- Utilities ---------------------------- #

def set_seed(seed: int):
    """Set seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@dataclass
class Graph:
    """Graph structure for decentralized communication topology."""
    n: int
    neighbors: List[List[int]]  # adjacency list, exclude self
    edges: List[Tuple[int, int]]


def make_graph(n: int, kind: str, p: float = 0.3) -> Graph:
    """Create different graph topologies for decentralized learning."""
    kind = kind.lower()
    neighbors = [[] for _ in range(n)]
    edges: List[Tuple[int, int]] = []

    if kind == "ring":
        for i in range(n):
            j = (i + 1) % n
            neighbors[i].append(j)
            neighbors[j].append(i)
            edges.append((min(i, j), max(i, j)))
    elif kind == "fully":
        for i in range(n):
            for j in range(i + 1, n):
                neighbors[i].append(j)
                neighbors[j].append(i)
                edges.append((i, j))
    elif kind in ("erdos", "er"):
        rng = random.Random(12345)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                    edges.append((i, j))
        # Ensure connectivity
        for i in range(n):
            if not neighbors[i]:
                j = (i + 1) % n
                neighbors[i].append(j)
                neighbors[j].append(i)
                edges.append((min(i, j), max(i, j)))
    else:
        raise ValueError(f"Unknown graph kind: {kind}")

    neighbors = [sorted(set(ns)) for ns in neighbors]
    edges = sorted(set(tuple(sorted(e)) for e in edges))
    return Graph(n=n, neighbors=neighbors, edges=edges)


# ---------------------------- Trust Monitor Implementation ---------------------------- #

@dataclass
class TrustConfig:
    """
    Configuration for trust monitoring in decentralized learning.

    Args:
        initial_trust: Starting trust score for new neighbors (0.0-1.0)
        history_window: Number of past fingerprints to store for consistency analysis
        trust_decay_rate: Rate at which trust decreases when anomalies detected (0.0-1.0)
        min_influence_weight: Minimum aggregation weight to preserve connectivity (0.0-1.0)
        enable_trust_monitoring: Whether to enable trust monitoring (if False, uses uniform weights)
    """
    initial_trust: float = 0.8
    history_window: int = 5
    trust_decay_rate: float = 0.1
    min_influence_weight: float = 0.1
    enable_trust_monitoring: bool = True


@dataclass
class GradientFingerprint:
    """
    Lightweight gradient fingerprint for anomaly detection.

    Captures key statistical properties of model parameter updates that can
    indicate potential attacks or anomalous behavior:

    Args:
        gradient_norm: Overall L2 norm of all parameters (detects magnitude attacks)
        cross_layer_correlation: Correlation between consecutive layers (detects structural manipulation)
        gradient_entropy: Shannon entropy of parameter distribution (detects distribution attacks)
        layer_variance: Variance across layer norms (detects layer-specific attacks)
    """
    gradient_norm: float
    cross_layer_correlation: float
    gradient_entropy: float
    layer_variance: float

    def distance_to(self, other: 'GradientFingerprint') -> float:
        """
        Compute normalized distance between two gradient fingerprints.

        Uses relative differences to handle varying magnitudes across training rounds.

        Args:
            other: Another gradient fingerprint to compare against

        Returns:
            Distance score in [0, ∞), where 0 = identical, higher = more different
        """
        norm_diff = abs(self.gradient_norm - other.gradient_norm) / (self.gradient_norm + other.gradient_norm + 1e-8)
        corr_diff = abs(self.cross_layer_correlation - other.cross_layer_correlation)
        entropy_diff = abs(self.gradient_entropy - other.gradient_entropy) / (self.gradient_entropy + other.gradient_entropy + 1e-8)
        var_diff = abs(self.layer_variance - other.layer_variance) / (self.layer_variance + other.layer_variance + 1e-8)

        return (norm_diff + corr_diff + entropy_diff + var_diff) / 4.0


class SimpleTrustMonitor:
    """
    Simplified trust monitor for decentralized learning.

    Implements gradient fingerprint-based trust scoring without the complexity
    of the full research implementation. Focuses on practical anomaly detection
    while maintaining computational efficiency.

    Key features:
    - Lightweight gradient analysis using statistical fingerprints
    - Self-reference comparison against own honest behavior
    - Historical consistency tracking across rounds
    - Neighborhood consensus for distributed anomaly detection
    - Continuous trust scores without binary detection thresholds
    """

    def __init__(self, node_id: str, config: TrustConfig):
        """
        Initialize trust monitor for a specific node.

        Args:
            node_id: Unique identifier for this node
            config: Trust monitoring configuration parameters
        """
        self.node_id = node_id
        self.config = config

        # Trust state tracking
        self.trust_scores: Dict[str, float] = {}       # Current trust score for each neighbor
        self.influence_weights: Dict[str, float] = {}  # Aggregation weights derived from trust

        # Historical fingerprint storage for consistency analysis
        self.fingerprint_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.history_window)
        )
        self.own_fingerprint_history: deque = deque(maxlen=config.history_window)

    @staticmethod
    def compute_gradient_fingerprint(model_state: Dict[str, torch.Tensor]) -> GradientFingerprint:
        """
        Compute lightweight gradient fingerprint from model parameters.

        Extracts key statistical properties that can indicate anomalous behavior
        while remaining computationally efficient for real-time use.

        Args:
            model_state: Dictionary mapping parameter names to tensors

        Returns:
            GradientFingerprint containing computed statistics
        """
        all_params = []
        layer_norms = []

        # Extract and flatten parameters from all layers
        for name, param in model_state.items():
            if hasattr(param, 'cpu'):
                param_array = param.cpu().numpy().flatten()
            else:
                param_array = np.array(param).flatten()

            # Remove NaN/Inf values that could skew statistics
            param_array = param_array[np.isfinite(param_array)]

            if len(param_array) > 0:
                all_params.extend(param_array.tolist())
                layer_norms.append(np.linalg.norm(param_array))

        # Need minimum parameters for meaningful statistics
        if len(all_params) < 10:
            return GradientFingerprint(0.0, 0.0, 0.0, 0.0)

        all_params = np.array(all_params)

        # 1. Overall gradient norm - detects magnitude-based attacks
        gradient_norm = float(np.linalg.norm(all_params))

        # 2. Cross-layer correlation - detects structural manipulation
        if len(layer_norms) >= 2:
            try:
                # Correlation between consecutive layer norms
                corr_matrix = np.corrcoef(layer_norms[:-1], layer_norms[1:])
                cross_layer_correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
            except:
                cross_layer_correlation = 0.0
        else:
            cross_layer_correlation = 0.0

        # 3. Gradient entropy - detects distribution manipulation
        if len(all_params) > 30:  # Need sufficient data points for histogram
            hist, _ = np.histogram(all_params, bins=20)
            hist = hist / (hist.sum() + 1e-10)  # Normalize to probabilities
            gradient_entropy = float(-np.sum(hist * np.log(hist + 1e-10)))  # Shannon entropy
        else:
            gradient_entropy = 0.0

        # 4. Layer variance - detects layer-specific attacks
        layer_variance = float(np.var(layer_norms)) if len(layer_norms) > 1 else 0.0

        return GradientFingerprint(
            gradient_norm=gradient_norm,
            cross_layer_correlation=cross_layer_correlation,
            gradient_entropy=gradient_entropy,
            layer_variance=layer_variance
        )

    def compute_trust_signals(self, neighbor_id: str, neighbor_fp: GradientFingerprint,
                              all_neighbor_fps: Dict[str, GradientFingerprint],
                              own_fp: Optional[GradientFingerprint] = None) -> float:
        """
        Compute trust signals (anomaly score) for a neighbor.

        Combines multiple anomaly detection methods to provide robust assessment:
        1. Self-reference: Compare against own honest behavior
        2. Historical consistency: Compare against neighbor's past behavior
        3. Neighborhood consensus: Compare against other neighbors
        4. Extreme value detection: Flag obviously anomalous values

        Args:
            neighbor_id: ID of the neighbor being evaluated
            neighbor_fp: Neighbor's current gradient fingerprint
            all_neighbor_fps: All neighbors' fingerprints for consensus
            own_fp: Own fingerprint for self-reference comparison

        Returns:
            Anomaly score in [0, 1] where 0=trustworthy, 1=highly anomalous
        """
        anomaly_signals = []

        # 1. Self-reference comparison (most reliable if available)
        if own_fp is not None:
            self_distance = neighbor_fp.distance_to(own_fp)
            # Scale distance to anomaly score - higher distance = more anomalous
            anomaly_signals.append(min(1.0, self_distance * 2.0))

        # 2. Historical consistency check
        if neighbor_id in self.fingerprint_history and len(self.fingerprint_history[neighbor_id]) >= 2:
            historical_fps = list(self.fingerprint_history[neighbor_id])
            # Compare against recent history (last 3 fingerprints)
            historical_distances = [neighbor_fp.distance_to(fp) for fp in historical_fps[-3:]]
            avg_historical_distance = np.mean(historical_distances)
            # Sudden changes in behavior are suspicious
            anomaly_signals.append(min(1.0, avg_historical_distance * 3.0))

        # 3. Neighborhood consensus check
        if len(all_neighbor_fps) >= 3:  # Need multiple neighbors for meaningful consensus
            other_fps = [fp for nid, fp in all_neighbor_fps.items() if nid != neighbor_id]
            if other_fps:
                consensus_distances = [neighbor_fp.distance_to(fp) for fp in other_fps]
                avg_consensus_distance = np.mean(consensus_distances)
                # Outliers from neighborhood consensus are suspicious
                anomaly_signals.append(min(1.0, avg_consensus_distance * 1.5))

        # 4. Extreme value detection for obvious attacks
        extreme_signals = []

        # Very high gradient norm could indicate gradient explosion attack
        if neighbor_fp.gradient_norm > 10.0:  # Threshold based on typical ranges
            extreme_signals.append(0.8)

        # Very low entropy indicates structured manipulation
        if neighbor_fp.gradient_entropy < 1.0:
            extreme_signals.append(0.6)

        if extreme_signals:
            anomaly_signals.append(max(extreme_signals))

        # Combine all signals with equal weighting
        if anomaly_signals:
            return min(1.0, np.mean(anomaly_signals))
        else:
            # Small baseline anomaly even when no signals detected
            # This accounts for natural variation in honest behavior
            return 0.1

    def update_trust_scores(self, neighbor_updates: Dict[str, Dict[str, torch.Tensor]],
                            own_parameters: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Update trust scores for all neighbors based on their parameter updates.

        This is the main entry point for trust monitoring. It:
        1. Computes gradient fingerprints for all neighbors
        2. Evaluates anomaly scores using multiple detection methods
        3. Updates trust scores with momentum-based smoothing
        4. Converts trust scores to influence weights for aggregation

        Args:
            neighbor_updates: Map of neighbor_id to their model parameters
            own_parameters: Own model parameters for self-reference baseline

        Returns:
            Dictionary mapping neighbor_id to influence weight for aggregation
        """
        if not self.config.enable_trust_monitoring:
            # Return uniform weights if trust monitoring disabled
            return {nid: 1.0 for nid in neighbor_updates.keys()}

        # Compute fingerprints for all neighbors
        neighbor_fps = {}
        for neighbor_id, params in neighbor_updates.items():
            fp = self.compute_gradient_fingerprint(params)
            neighbor_fps[neighbor_id] = fp
            # Store in history for consistency analysis
            self.fingerprint_history[neighbor_id].append(fp)

        # Compute own fingerprint for self-reference
        own_fp = None
        if own_parameters is not None:
            own_fp = self.compute_gradient_fingerprint(own_parameters)
            self.own_fingerprint_history.append(own_fp)

        # Update trust scores for each neighbor
        for neighbor_id, neighbor_fp in neighbor_fps.items():
            # Compute current anomaly score
            anomaly_score = self.compute_trust_signals(neighbor_id, neighbor_fp, neighbor_fps, own_fp)

            # Initialize trust score for new neighbors
            if neighbor_id not in self.trust_scores:
                self.trust_scores[neighbor_id] = self.config.initial_trust

            current_trust = self.trust_scores[neighbor_id]

            # Update trust based on anomaly level
            if anomaly_score > 0.5:  # High anomaly detected
                # Decrease trust proportional to anomaly severity
                trust_change = -self.config.trust_decay_rate * anomaly_score
            else:  # Low anomaly - allow slow recovery
                # Gradual recovery towards perfect trust
                trust_change = 0.02 * (1.0 - current_trust)

            # Apply trust change with clipping to valid range
            new_trust = np.clip(current_trust + trust_change, 0.01, 1.0)
            self.trust_scores[neighbor_id] = new_trust

            # Convert trust score to influence weight for aggregation
            # High trust = high influence, but maintain minimum weight for connectivity
            if new_trust > 0.7:
                influence = 1.0  # Full influence for highly trusted neighbors
            elif new_trust > 0.4:
                # Linear interpolation in medium trust range
                influence = 0.5 + (new_trust - 0.4) / 0.3 * 0.5
            else:
                # Low trust gets minimal influence but non-zero for connectivity
                influence = self.config.min_influence_weight + (new_trust / 0.4) * (0.5 - self.config.min_influence_weight)

            self.influence_weights[neighbor_id] = influence

        return self.influence_weights.copy()

    def get_trust_summary(self) -> Dict:
        """
        Get summary of current trust state for logging and analysis.

        Returns:
            Dictionary containing trust statistics and current state
        """
        trust_values = list(self.trust_scores.values())
        return {
            "node_id": self.node_id,
            "num_neighbors": len(self.trust_scores),
            "mean_trust": np.mean(trust_values) if trust_values else 1.0,
            "min_trust": np.min(trust_values) if trust_values else 1.0,
            "trust_scores": self.trust_scores.copy(),
            "influence_weights": self.influence_weights.copy()
        }


# ---------------------------- Training helpers ---------------------------- #

def evaluate(model: nn.Module, loader: DataLoader, device_: torch.device) -> Tuple[float, float, int, int]:
    """Evaluate model performance on a dataset."""
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device_), yb.to(device_)
            logits = model(xb)
            loss = crit(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
    return correct / max(1, total), loss_sum / max(1, total), correct, total


def local_train(model: nn.Module, loader: DataLoader, epochs: int, lr: float, device_: torch.device):
    """Train model locally for specified number of epochs."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device_), yb.to(device_)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()


def get_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract model parameters efficiently."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def set_state(model: nn.Module, state: Dict[str, torch.Tensor]):
    """Load model parameters efficiently."""
    model.load_state_dict(state, strict=False)


def average_states(states: List[Dict[str, torch.Tensor]], weights: List[float] | None = None) -> Dict[str, torch.Tensor]:
    """Compute weighted average of model states."""
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    out: Dict[str, torch.Tensor] = {}
    for k in states[0].keys():
        acc = None
        for s, w in zip(states, weights):
            t = s[k]
            acc = t * w if acc is None else acc + t * w
        out[k] = acc
    return out


def compute_model_distance(state1: Dict[str, torch.Tensor], state2: Dict[str, torch.Tensor]) -> float:
    """Compute Euclidean distance between two model states."""
    total_distance = 0.0
    for key in state1.keys():
        diff = state1[key] - state2[key]
        total_distance += torch.sum(diff * diff).item()
    return total_distance


def krum_select(model_states: List[Dict[str, torch.Tensor]], num_compromised: int = 0) -> int:
    """Select model using Krum algorithm."""
    m = len(model_states)
    c = num_compromised

    if c >= (m - 2) / 2:
        raise ValueError(f"Krum requires c < (m-2)/2, got c={c}, m={m}")

    distances = []
    for i in range(m):
        model_distances = []
        for j in range(m):
            if i != j:
                dist = compute_model_distance(model_states[i], model_states[j])
                model_distances.append(dist)
        distances.append(model_distances)

    scores = []
    for i in range(m):
        sorted_distances = sorted(distances[i])
        num_closest = max(1, m - c - 2)
        closest_distances = sorted_distances[:num_closest]
        score = sum(closest_distances)
        scores.append(score)

    selected_idx = scores.index(min(scores))
    return selected_idx


# ---------------------------- Attack Implementation ---------------------------- #

class LocalModelPoisoningAttacker:
    """
    Implementation of local model poisoning attacks for decentralized learning.

    Based on "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
    but adapted for decentralized setting with the following assumptions:

    Key Assumptions:
    - Partial knowledge: attacker only knows compromised nodes' data
    - Global coordination: compromised nodes can coordinate (attacker's privilege)
    - Neighborhood communication: each node only sees its immediate neighbors
    - Decentralized aggregation: no central server, only peer-to-peer communication

    Attack Types:
    - directed_deviation: Craft models that push parameters opposite to natural learning direction
    - random: Baseline attack that adds random noise to parameters

    The attack works by:
    1. Estimating the natural parameter change directions (s vector from paper)
    2. Crafting malicious models that push in opposite direction: w' = w_ref - λ * s
    3. For Krum: creating multiple similar malicious models to game selection
    """

    def __init__(self, num_nodes: int, attack_percentage: float, attack_type: str,
                 lambda_param: float = 1.0, seed: int = 42):
        """
        Initialize the attacker.

        Args:
            num_nodes: Total number of nodes in the network
            attack_percentage: Fraction of nodes to compromise (0.0-1.0)
            attack_type: Type of attack ('directed_deviation' or 'random')
            lambda_param: Attack strength parameter (λ from paper) - higher = stronger attack
            seed: Random seed for reproducible node selection
        """
        self.num_nodes = num_nodes
        self.attack_percentage = attack_percentage
        self.attack_type = attack_type
        self.lambda_param = lambda_param

        # Deterministically select which nodes are compromised
        num_compromised = int(num_nodes * attack_percentage)
        if num_compromised == 0 and attack_percentage > 0:
            num_compromised = 1  # At least one node if any attack requested

        random.seed(seed)
        self.compromised_nodes = set(random.sample(range(num_nodes), min(num_compromised, num_nodes)))
        print(f"Attack: Compromised {len(self.compromised_nodes)}/{num_nodes} nodes: {sorted(self.compromised_nodes)}")

        # Storage for attack coordination and direction estimation
        self.previous_neighborhood_avgs = {}  # For neighborhood-based direction estimation
        self.compromised_node_states = {}     # For global attacker coordination

    def update_compromised_states(self, round_num: int, node_states: List[Dict[str, torch.Tensor]]):
        """
        Collect states from all compromised nodes for global coordination.

        This implements the attacker's privilege of coordinating across compromised nodes
        while maintaining the decentralized property for honest nodes.

        Args:
            round_num: Current training round
            node_states: List of model states for all nodes
        """
        self.compromised_node_states[round_num] = {}
        for node_id in self.compromised_nodes:
            self.compromised_node_states[round_num][node_id] = {k: v.clone() for k, v in node_states[node_id].items()}

    def estimate_global_directions_from_compromised(self, round_num: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Estimate global changing directions using only compromised nodes' data.

        This implements the partial knowledge assumption - the attacker can only
        observe compromised nodes' parameters to estimate the direction the global
        model would naturally move.

        The attack estimates the "s" vector from the paper by comparing the average
        of compromised nodes between consecutive rounds:
        - If parameter value increased: direction = +1
        - If parameter value decreased: direction = -1

        Args:
            round_num: Current training round

        Returns:
            Dictionary mapping parameter names to direction tensors (+1 or -1)
            None if insufficient data for estimation
        """
        if round_num < 1 or round_num not in self.compromised_node_states:
            return None

        prev_round = round_num - 1
        if prev_round not in self.compromised_node_states:
            return None

        current_compromised = list(self.compromised_node_states[round_num].values())
        previous_compromised = list(self.compromised_node_states[prev_round].values())

        if not previous_compromised or not current_compromised:
            return None

        # Compute mean of compromised nodes for current and previous rounds
        current_mean = average_states(current_compromised)
        previous_mean = average_states(previous_compromised)

        # Direction estimation: if current > previous, direction is +1, else -1
        # This estimates the "s" vector from the paper
        directions = {}
        for key in current_mean.keys():
            directions[key] = torch.where(
                current_mean[key] > previous_mean[key],
                torch.ones_like(current_mean[key]),
                -torch.ones_like(current_mean[key])
            )

        return directions

    def estimate_neighborhood_directions(self, node_id: int, current_neigh_states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Estimate directions using only neighborhood information.

        This is the fallback when global direction estimation fails.
        Uses the change in neighborhood average to estimate parameter directions.

        Args:
            node_id: ID of the node whose neighborhood to analyze
            current_neigh_states: Current model states of honest neighbors

        Returns:
            Dictionary mapping parameter names to estimated directions
        """
        if not current_neigh_states:
            # Fallback: generate random directions
            if self.compromised_node_states:
                sample_state = list(self.compromised_node_states[max(self.compromised_node_states.keys())].values())[0]
                directions = {}
                for key in sample_state.keys():
                    directions[key] = torch.sign(torch.randn_like(sample_state[key]))
                return directions
            return {}

        current_neigh_avg = average_states(current_neigh_states)
        previous_neigh_avg = self.previous_neighborhood_avgs.get(node_id)

        if previous_neigh_avg is None:
            # First round - use random directions
            directions = {}
            for key in current_neigh_avg.keys():
                directions[key] = torch.sign(torch.randn_like(current_neigh_avg[key]))
            # Store for next round
            self.previous_neighborhood_avgs[node_id] = {k: v.clone() for k, v in current_neigh_avg.items()}
            return directions

        # Estimate directions based on neighborhood parameter changes
        directions = {}
        for key in current_neigh_avg.keys():
            directions[key] = torch.where(
                current_neigh_avg[key] > previous_neigh_avg[key],
                torch.ones_like(current_neigh_avg[key]),
                -torch.ones_like(current_neigh_avg[key])
            )

        # Store current average for next round
        self.previous_neighborhood_avgs[node_id] = {k: v.clone() for k, v in current_neigh_avg.items()}

        return directions

    def craft_malicious_models_decentralized(self, node_id: int, neighborhood_indices: List[int],
                                             honest_neigh_states: List[Dict[str, torch.Tensor]],
                                             round_num: int) -> List[Dict[str, torch.Tensor]]:
        """
        Craft malicious models for compromised nodes in this neighborhood.

        This implements the core attack logic adapted for decentralized setting:
        1. Estimate changing directions (s vector from paper)
        2. Craft models that push in opposite direction: w' = w_ref - λ * s
        3. For Krum: create multiple similar malicious models to increase selection probability

        Args:
            node_id: Current node ID (for direction estimation context)
            neighborhood_indices: List of node IDs in this neighborhood
            honest_neigh_states: Model states from honest neighbors (partial knowledge)
            round_num: Current training round

        Returns:
            List of crafted malicious model states for compromised nodes in neighborhood
        """
        num_compromised_in_neigh = len([i for i in neighborhood_indices if i in self.compromised_nodes])
        if num_compromised_in_neigh == 0:
            return []

        if self.attack_type == "random":
            # Random baseline attack for comparison
            malicious_states = []
            for _ in range(num_compromised_in_neigh):
                if honest_neigh_states:
                    base_state = honest_neigh_states[0]
                elif self.compromised_node_states and round_num in self.compromised_node_states:
                    base_state = list(self.compromised_node_states[round_num].values())[0]
                else:
                    continue

                random_state = {}
                for key in base_state.keys():
                    noise = torch.randn_like(base_state[key]) * 0.5
                    random_state[key] = base_state[key] + noise
                malicious_states.append(random_state)
            return malicious_states

        # Directed deviation attack (main attack from paper)

        # Step 1: Get changing directions (s vector)
        # Try global estimation first (using compromised node coordination)
        directions = self.estimate_global_directions_from_compromised(round_num)
        if directions is None:
            # Fallback to neighborhood-based estimation
            directions = self.estimate_neighborhood_directions(node_id, honest_neigh_states)

        if not directions:
            return []

        # Step 2: Determine reference state (partial knowledge assumption)
        # Use honest neighbors' average as reference point
        if honest_neigh_states:
            reference_state = average_states(honest_neigh_states)
        elif self.compromised_node_states and round_num in self.compromised_node_states:
            # Fallback: use compromised nodes' average
            compromised_states = list(self.compromised_node_states[round_num].values())
            reference_state = average_states(compromised_states)
        else:
            return []

        # Step 3: Craft malicious models
        malicious_states = []

        # Primary malicious model: w' = w_ref - λ * s
        # This pushes parameters in opposite direction to natural learning
        primary_malicious = {}
        for key in reference_state.keys():
            primary_malicious[key] = reference_state[key] - self.lambda_param * directions[key]

        malicious_states.append(primary_malicious)

        # Supporting models for Krum attack
        # Create additional malicious models similar to primary (within small epsilon)
        # This makes Krum more likely to select the malicious model as "representative"
        epsilon = 0.01
        for _ in range(num_compromised_in_neigh - 1):
            supporting_model = {}
            for key in primary_malicious.keys():
                noise = torch.randn_like(primary_malicious[key]) * epsilon
                supporting_model[key] = primary_malicious[key] + noise
            malicious_states.append(supporting_model)

        return malicious_states[:num_compromised_in_neigh]


# ---------------------------- Aggregation Functions ---------------------------- #

def decentralized_fedavg_step_with_attack(models: List[nn.Module], graph: Graph, round_num: int = 0,
                                          attacker: Optional[LocalModelPoisoningAttacker] = None):
    """Decentralized FedAvg with attack capability."""
    states = [get_state(m) for m in models]

    if attacker:
        attacker.update_compromised_states(round_num, states)

    new_states = []
    for i in range(graph.n):
        neigh = [i] + graph.neighbors[i]

        if attacker and any(j in attacker.compromised_nodes for j in neigh):
            compromised_in_neigh = [j for j in neigh if j in attacker.compromised_nodes]
            honest_in_neigh = [j for j in neigh if j not in attacker.compromised_nodes]
            honest_neigh_states = [states[j] for j in honest_in_neigh]

            malicious_states = attacker.craft_malicious_models_decentralized(
                i, neigh, honest_neigh_states, round_num
            )

            modified_neigh_states = []
            malicious_idx = 0
            for j in neigh:
                if j in attacker.compromised_nodes and malicious_idx < len(malicious_states):
                    modified_neigh_states.append(malicious_states[malicious_idx])
                    malicious_idx += 1
                else:
                    modified_neigh_states.append(states[j])

            neigh_states = modified_neigh_states
        else:
            neigh_states = [states[j] for j in neigh]

        w = [1.0 / len(neigh_states)] * len(neigh_states)
        new_states.append(average_states(neigh_states, w))

    for model, st in zip(models, new_states):
        set_state(model, st)


def gossip_round_with_attack(models: List[nn.Module], graph: Graph, steps: int = 1,
                             round_num: int = 0, seed: int = 42,
                             attacker: Optional[LocalModelPoisoningAttacker] = None):
    """Gossip averaging with attack capability."""
    if not graph.edges:
        return

    states = [get_state(m) for m in models]

    if attacker and attacker.compromised_nodes:
        attacker.update_compromised_states(round_num, states)

        for comp_idx in attacker.compromised_nodes:
            neighbors_of_comp = graph.neighbors[comp_idx] + [comp_idx]
            honest_neighbors = [j for j in neighbors_of_comp if j not in attacker.compromised_nodes]
            honest_neigh_states = [states[j] for j in honest_neighbors]

            malicious_states = attacker.craft_malicious_models_decentralized(
                comp_idx, neighbors_of_comp, honest_neigh_states, round_num
            )

            if malicious_states:
                set_state(models[comp_idx], malicious_states[0])

    rng = random.Random(seed + round_num * 100)
    for _ in range(steps):
        i, j = rng.choice(graph.edges)
        si, sj = get_state(models[i]), get_state(models[j])
        avg = average_states([si, sj], [0.5, 0.5])
        set_state(models[i], avg)
        set_state(models[j], avg)


def decentralized_krum_step_with_attack(models: List[nn.Module], graph: Graph,
                                        pct_compromised: float, round_num: int = 0,
                                        attacker: Optional[LocalModelPoisoningAttacker] = None):
    """Decentralized Krum with attack capability."""
    states = [get_state(m) for m in models]

    if attacker:
        attacker.update_compromised_states(round_num, states)

    new_states = []

    for i in range(graph.n):
        neigh = [i] + graph.neighbors[i]
        original_neigh_states = [states[j] for j in neigh]

        if attacker and any(j in attacker.compromised_nodes for j in neigh):
            compromised_in_neigh = [j for j in neigh if j in attacker.compromised_nodes]
            honest_in_neigh = [j for j in neigh if j not in attacker.compromised_nodes]
            honest_neigh_states = [states[j] for j in honest_in_neigh]

            malicious_states = attacker.craft_malicious_models_decentralized(
                i, neigh, honest_neigh_states, round_num
            )

            modified_neigh_states = []
            malicious_idx = 0
            for j in neigh:
                if j in attacker.compromised_nodes and malicious_idx < len(malicious_states):
                    modified_neigh_states.append(malicious_states[malicious_idx])
                    malicious_idx += 1
                else:
                    modified_neigh_states.append(states[j])

            neigh_states = modified_neigh_states
        else:
            neigh_states = original_neigh_states

        if len(neigh_states) > 1:
            try:
                neighborhood_size = len(neigh_states)
                c = int(neighborhood_size * pct_compromised)
                selected_idx = krum_select(neigh_states, c)
                selected_state = neigh_states[selected_idx]
                new_states.append(selected_state)
            except ValueError:
                print(f"Warning: Krum failed for node {i}, falling back to averaging")
                w = [1.0 / len(neigh_states)] * len(neigh_states)
                new_states.append(average_states(neigh_states, w))
        else:
            new_states.append(neigh_states[0])

    for model, st in zip(models, new_states):
        set_state(model, st)


def trust_weighted_fedavg_step(models: List[nn.Module], graph: Graph, trust_monitors: Dict[str, SimpleTrustMonitor],
                               round_num: int = 0, attacker: Optional[LocalModelPoisoningAttacker] = None):
    """Trust-weighted FedAvg aggregation step."""
    states = [get_state(m) for m in models]

    if attacker:
        attacker.update_compromised_states(round_num, states)

    new_states = []
    for i in range(graph.n):
        node_id = str(i)
        neighbors = [i] + graph.neighbors[i]

        if attacker and any(j in attacker.compromised_nodes for j in neighbors):
            compromised_in_neigh = [j for j in neighbors if j in attacker.compromised_nodes]
            honest_in_neigh = [j for j in neighbors if j not in attacker.compromised_nodes]
            honest_neigh_states = [states[j] for j in honest_in_neigh]

            malicious_states = attacker.craft_malicious_models_decentralized(
                i, neighbors, honest_neigh_states, round_num
            )

            modified_states = []
            malicious_idx = 0
            for j in neighbors:
                if j in attacker.compromised_nodes and malicious_idx < len(malicious_states):
                    modified_states.append(malicious_states[malicious_idx])
                    malicious_idx += 1
                else:
                    modified_states.append(states[j])

            neighbor_states_dict = {str(neighbors[k]): modified_states[k] for k in range(len(neighbors)) if neighbors[k] != i}
            own_state = states[i]
        else:
            neighbor_states_dict = {str(j): states[j] for j in neighbors if j != i}
            own_state = states[i]

        if neighbor_states_dict:
            trust_monitor = trust_monitors.get(node_id)
            if trust_monitor:
                influence_weights = trust_monitor.update_trust_scores(neighbor_states_dict, own_state)
            else:
                influence_weights = {nid: 1.0 for nid in neighbor_states_dict.keys()}

            all_states = [own_state]
            all_weights = [1.0]

            for nid, state in neighbor_states_dict.items():
                all_states.append(state)
                all_weights.append(influence_weights.get(nid, 0.1))

            total_weight = sum(all_weights)
            normalized_weights = [w / total_weight for w in all_weights]

            new_states.append(average_states(all_states, normalized_weights))
        else:
            new_states.append(own_state)

    for model, state in zip(models, new_states):
        set_state(model, state)


def trust_weighted_gossip_round(models: List[nn.Module], graph: Graph, trust_monitors: Dict[str, SimpleTrustMonitor],
                                steps: int = 1, round_num: int = 0, seed: int = 42,
                                attacker: Optional[LocalModelPoisoningAttacker] = None):
    """Trust-weighted gossip averaging."""
    if not graph.edges:
        return

    states = [get_state(m) for m in models]

    if attacker and attacker.compromised_nodes:
        attacker.update_compromised_states(round_num, states)

        for comp_idx in attacker.compromised_nodes:
            neighbors_of_comp = graph.neighbors[comp_idx] + [comp_idx]
            honest_neighbors = [j for j in neighbors_of_comp if j not in attacker.compromised_nodes]
            honest_neigh_states = [states[j] for j in honest_neighbors]

            malicious_states = attacker.craft_malicious_models_decentralized(
                comp_idx, neighbors_of_comp, honest_neigh_states, round_num
            )

            if malicious_states:
                set_state(models[comp_idx], malicious_states[0])

    rng = random.Random(seed + round_num * 100)

    for _ in range(steps):
        i, j = rng.choice(graph.edges)

        state_i = get_state(models[i])
        state_j = get_state(models[j])

        trust_monitor_i = trust_monitors.get(str(i))
        trust_monitor_j = trust_monitors.get(str(j))

        if trust_monitor_i:
            neighbor_updates = {str(j): state_j}
            influence_weights = trust_monitor_i.update_trust_scores(neighbor_updates, state_i)
            weight_j_from_i = influence_weights.get(str(j), 0.5)
        else:
            weight_j_from_i = 0.5

        if trust_monitor_j:
            neighbor_updates = {str(i): state_i}
            influence_weights = trust_monitor_j.update_trust_scores(neighbor_updates, state_j)
            weight_i_from_j = influence_weights.get(str(i), 0.5)
        else:
            weight_i_from_j = 0.5

        avg_trust_i = weight_i_from_j
        avg_trust_j = weight_j_from_i

        total_trust = avg_trust_i + avg_trust_j
        if total_trust > 0:
            norm_weight_i = avg_trust_i / total_trust
            norm_weight_j = avg_trust_j / total_trust
        else:
            norm_weight_i = norm_weight_j = 0.5

        avg_state = average_states([state_i, state_j], [norm_weight_i, norm_weight_j])

        set_state(models[i], avg_state)
        set_state(models[j], avg_state)


# Original aggregation functions (no attack)
def decentralized_fedavg_step(models: List[nn.Module], graph: Graph):
    """Original decentralized FedAvg without attack capability."""
    states = [get_state(m) for m in models]
    new_states = []
    for i in range(graph.n):
        neigh = [i] + graph.neighbors[i]
        neigh_states = [states[j] for j in neigh]
        w = [1.0 / len(neigh_states)] * len(neigh_states)
        new_states.append(average_states(neigh_states, w))
    for model, st in zip(models, new_states):
        set_state(model, st)


def gossip_round(models: List[nn.Module], graph: Graph, steps: int = 1, round_num: int = 0, seed: int = 42):
    """Original gossip averaging without attack capability."""
    if not graph.edges:
        return
    rng = random.Random(seed + round_num * 100)
    for _ in range(steps):
        i, j = rng.choice(graph.edges)
        si, sj = get_state(models[i]), get_state(models[j])
        avg = average_states([si, sj], [0.5, 0.5])
        set_state(models[i], avg)
        set_state(models[j], avg)


def decentralized_krum_step(models: List[nn.Module], graph: Graph, pct_compromised: float = 0.0):
    """Original decentralized Krum without attack capability."""
    states = [get_state(m) for m in models]
    new_states = []

    for i in range(graph.n):
        neigh = [i] + graph.neighbors[i]
        neigh_states = [states[j] for j in neigh]

        if len(neigh_states) > 1:
            try:
                neighborhood_size = len(neigh_states)
                c = int(neighborhood_size * pct_compromised)
                selected_idx = krum_select(neigh_states, c)
                selected_state = neigh_states[selected_idx]
                new_states.append(selected_state)
            except ValueError:
                w = [1.0 / len(neigh_states)] * len(neigh_states)
                new_states.append(average_states(neigh_states, w))
        else:
            new_states.append(neigh_states[0])

    for model, st in zip(models, new_states):
        set_state(model, st)


# ---------------------------- Main Simulator ---------------------------- #

def run_sim(args):
    """Main simulation function."""
    set_seed(args.seed)
    dev = device()
    print(f"Device: {dev}")
    print(f"Seed: {args.seed}")

    # Load dataset
    if args.dataset.lower() == "femnist":
        data_path = "./leaf/data/femnist/data"
        train_ds, test_ds, model_template, num_classes, input_size = load_leaf_dataset("femnist", data_path)
        image_size = input_size
    elif args.dataset.lower() == "celeba":
        data_path = "./leaf/data/celeba/data"
        train_ds, test_ds, model_template, num_classes, input_size = load_leaf_dataset("celeba", data_path)
        image_size = input_size
    else:
        raise ValueError(f"Dataset {args.dataset} not supported. Use 'femnist' or 'celeba'")

    # Create client partitions
    train_partitions, test_partitions = create_leaf_client_partitions(train_ds, test_ds, args.num_nodes, seed=args.seed)
    parts = [Subset(train_ds, indices) for indices in train_partitions]
    test_parts = [Subset(test_ds, indices) for indices in test_partitions]

    num_workers = 4
    pin_memory = dev.type != "cpu"

    use_sampling = args.max_samples is not None
    if use_sampling:
        print(f"Will sample {args.max_samples} samples per client per epoch")

    test_loaders = [DataLoader(tp, batch_size=512, shuffle=False,
                               num_workers=0, pin_memory=False) for tp in test_parts]

    # Create graph topology
    graph = make_graph(args.num_nodes, args.graph, p=args.p)
    print(f"Graph: {args.graph}, nodes: {args.num_nodes}, edges: {len(graph.edges)}")

    # Initialize attacker if requested
    attacker = None
    if args.attack_percentage > 0:
        attacker = LocalModelPoisoningAttacker(
            args.num_nodes,
            args.attack_percentage,
            args.attack_type,
            args.attack_lambda,
            args.seed
        )
        print(f"Attack type: {args.attack_type}, lambda: {args.attack_lambda}")

    # Initialize trust monitors for trust-weighted aggregation
    trust_monitors = {}
    if args.agg in ["trust-fedavg", "trust-gossip"]:
        trust_config = TrustConfig(
            initial_trust=0.8,
            history_window=5,
            trust_decay_rate=args.trust_decay_rate,
            min_influence_weight=0.1,
            enable_trust_monitoring=True
        )
        for i in range(args.num_nodes):
            trust_monitors[str(i)] = SimpleTrustMonitor(str(i), trust_config)
        print(f"Trust monitoring enabled with decay rate {args.trust_decay_rate}")

    # Initialize models
    models = []
    for i in range(args.num_nodes):
        torch.manual_seed(args.seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + i)

        if args.dataset.lower() == "femnist":
            model = LEAFFEMNISTModel(num_classes=num_classes).to(dev)
        elif args.dataset.lower() == "celeba":
            model = LEAFCelebAModel(num_classes=num_classes, image_size=image_size).to(dev)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        if dev.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass
        models.append(model)

    # Evaluate initial performance
    with torch.no_grad():
        base_accs = []
        for i, m in enumerate(models):
            acc, _, _, _ = evaluate(m, test_loaders[i], dev)
            base_accs.append(acc)
        print(f"Initial test acc across nodes: mean={np.mean(base_accs):.4f} ± {np.std(base_accs):.4f}")

    # Main training loop
    for r in range(1, args.rounds + 1):
        # Create data loaders
        if use_sampling:
            loaders = []
            for i, p in enumerate(parts):
                num_samples = min(args.max_samples, len(p))
                round_seed = args.seed + r * 1000 + i
                sampler = RandomSampler(p, replacement=False, num_samples=num_samples,
                                        generator=torch.Generator().manual_seed(round_seed))
                loader = DataLoader(p, batch_size=args.batch_size, sampler=sampler,
                                    num_workers=num_workers, pin_memory=pin_memory)
                loaders.append(loader)
        else:
            loaders = []
            for i, p in enumerate(parts):
                round_seed = args.seed + r * 1000 + i
                generator = torch.Generator().manual_seed(round_seed)
                loader = DataLoader(p, batch_size=args.batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_memory,
                                    persistent_workers=(num_workers > 0 and r == 1),
                                    prefetch_factor=2 if num_workers > 0 else None,
                                    generator=generator)
                loaders.append(loader)

        # Local training phase
        for i, (m, ld) in enumerate(zip(models, loaders)):
            local_train(m, ld, epochs=args.local_epochs, lr=args.lr, device_=dev)

        # Communication/aggregation phase
        if args.agg == "d-fedavg":
            if attacker:
                decentralized_fedavg_step_with_attack(models, graph, r, attacker)
            else:
                decentralized_fedavg_step(models, graph)
        elif args.agg == "gossip":
            if attacker:
                gossip_round_with_attack(models, graph, steps=args.gossip_steps,
                                         round_num=r, seed=args.seed, attacker=attacker)
            else:
                gossip_round(models, graph, steps=args.gossip_steps, round_num=r, seed=args.seed)
        elif args.agg == "krum":
            if attacker:
                decentralized_krum_step_with_attack(models, graph, args.pct_compromised, r, attacker)
            else:
                decentralized_krum_step(models, graph, args.pct_compromised)
        elif args.agg == "trust-fedavg":
            trust_weighted_fedavg_step(models, graph, trust_monitors, r, attacker)
        elif args.agg == "trust-gossip":
            trust_weighted_gossip_round(models, graph, trust_monitors, steps=args.gossip_steps,
                                        round_num=r, seed=args.seed, attacker=attacker)
        else:
            raise ValueError("agg must be 'd-fedavg', 'gossip', 'krum', 'trust-fedavg', or 'trust-gossip'")

        # Evaluation phase
        accs = []
        losses = []
        correct_totals = []
        for i, m in enumerate(models):
            acc, loss, correct, total = evaluate(m, test_loaders[i], dev)
            accs.append(acc)
            losses.append(loss)
            correct_totals.append((correct, total))

        # Print round statistics
        print(f"Round {r:03d}: test acc mean={np.mean(accs):.4f} ± {np.std(accs):.4f} | "
              f"min={np.min(accs):.4f} max={np.max(accs):.4f}")
        print(f"         : test loss mean={np.mean(losses):.4f} ± {np.std(losses):.4f}")

        if args.verbose:
            acc_strs = [f"{acc:.6f}" for acc in accs]
            print(f"         : individual accs = {acc_strs}")
            print(f"         : correct/total = {correct_totals}")

            # Show compromised vs honest node performance if under attack
            if attacker:
                compromised_accs = [accs[i] for i in attacker.compromised_nodes]
                honest_accs = [accs[i] for i in range(args.num_nodes) if i not in attacker.compromised_nodes]
                if compromised_accs and honest_accs:
                    print(f"         : compromised nodes: mean acc={np.mean(compromised_accs):.4f}")
                    print(f"         : honest nodes: mean acc={np.mean(honest_accs):.4f}")

            # Show trust information for trust-weighted methods
            if args.agg in ["trust-fedavg", "trust-gossip"] and trust_monitors:
                trust_summaries = []
                for node_id, monitor in trust_monitors.items():
                    summary = monitor.get_trust_summary()
                    if summary["num_neighbors"] > 0:
                        trust_summaries.append(f"Node {node_id}: {summary['mean_trust']:.3f}")
                if trust_summaries:
                    print(f"         : trust scores = {trust_summaries}")

    # Final evaluation and summary
    accs = []
    for i, m in enumerate(models):
        acc, _, _, _ = evaluate(m, test_loaders[i], dev)
        accs.append(acc)

    print("\n=== FINAL RESULTS ===")
    print(f"Dataset: {args.dataset}, Nodes: {args.num_nodes}, Graph: {args.graph}, Agg: {args.agg}")
    if attacker:
        print(f"Attack: {args.attack_type}, {args.attack_percentage*100:.1f}% compromised, lambda={args.attack_lambda}")
        compromised_accs = [accs[i] for i in attacker.compromised_nodes]
        honest_accs = [accs[i] for i in range(args.num_nodes) if i not in attacker.compromised_nodes]
        if compromised_accs and honest_accs:
            print(f"Final accuracy - Compromised: {np.mean(compromised_accs):.4f}, Honest: {np.mean(honest_accs):.4f}")
    else:
        print("No attack (clean run)")
    print(f"Test accuracy: mean={np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # Print trust summary for trust-weighted methods
    if args.agg in ["trust-fedavg", "trust-gossip"] and trust_monitors:
        print("\n=== TRUST SUMMARY ===")
        all_trust_scores = []
        for node_id, monitor in trust_monitors.items():
            summary = monitor.get_trust_summary()
            all_trust_scores.extend(summary["trust_scores"].values())
            if summary["num_neighbors"] > 0:
                print(f"Node {node_id}: neighbors={summary['num_neighbors']}, "
                      f"mean_trust={summary['mean_trust']:.3f}, min_trust={summary['min_trust']:.3f}")

        if all_trust_scores:
            print(f"Overall trust distribution: mean={np.mean(all_trust_scores):.3f} ± {np.std(all_trust_scores):.3f}")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Decentralized Learning Simulator with Trust-Weighted Aggregation")

    # Dataset and basic training parameters
    p.add_argument("--dataset", type=str, choices=["femnist", "celeba"], required=True,
                   help="LEAF dataset to use")
    p.add_argument("--num-nodes", type=int, default=8,
                   help="Number of nodes in the decentralized network")
    p.add_argument("--rounds", type=int, default=20,
                   help="Number of training rounds")
    p.add_argument("--local-epochs", type=int, default=1,
                   help="Number of local training epochs per round")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Batch size for local training")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Learning rate for local SGD")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Max samples per client per epoch (for large datasets)")

    # Aggregation algorithm parameters
    p.add_argument("--agg", type=str,
                   choices=["d-fedavg", "gossip", "krum", "trust-fedavg", "trust-gossip"],
                   default="d-fedavg",
                   help="Aggregation algorithm")
    p.add_argument("--gossip-steps", type=int, default=10,
                   help="Number of random edge gossips per round (for gossip aggregation)")
    p.add_argument("--pct-compromised", type=float, default=0.0,
                   help="Max percentage of compromised models per neighborhood for Krum")

    # Trust monitoring parameters
    p.add_argument("--trust-decay-rate", type=float, default=0.1,
                   help="Trust decay rate for trust-weighted aggregation")

    # Graph topology parameters
    p.add_argument("--graph", type=str, choices=["ring", "fully", "erdos"], default="ring",
                   help="Graph topology for decentralized communication")
    p.add_argument("--p", type=float, default=0.3,
                   help="Edge probability for Erdős–Rényi random graph")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")

    # Attack parameters
    p.add_argument("--attack-percentage", type=float, default=0.0,
                   help="Percentage of nodes to compromise (0.0-1.0)")
    p.add_argument("--attack-type", type=str, choices=["directed_deviation", "random"],
                   default="directed_deviation",
                   help="Attack type")
    p.add_argument("--attack-lambda", type=float, default=1.0,
                   help="Lambda parameter controlling attack strength")

    # Debug/verbose
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose output for debugging")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sim(args)
