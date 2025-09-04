#!/usr/bin/env python3
"""
Decentralized Learning Simulator with BALANCE and JL-BALANCE

Supports aggregation strategies over a peer graph:
  1) Decentralized FedAvg (synchronous neighbor averaging per round)
  2) Gossip averaging (asynchronous-style, simulated with K random edge gossips per round)
  3) Decentralized Krum (Byzantine-robust aggregation, selects most similar model)
  4) BALANCE (Byzantine-robust averaging through local similarity)
  5) JL-BALANCE (Lightweight BALANCE using Johnson-Lindenstrauss projection)

CORRECTED: Attacks now happen at parameter level before projection for fair comparison.

Example usage:
  # Clean run with BALANCE
  python decentralized_fl_sim.py \
      --dataset femnist --num-nodes 8 --rounds 20 --local-epochs 1 \
      --agg balance --graph ring --lr 0.01

  # JL-BALANCE for massive speedup
  python decentralized_fl_sim.py \
      --dataset femnist --num-nodes 8 --rounds 20 --local-epochs 1 \
      --agg jl-balance --graph ring --lr 0.01 --jl-projection-dim 200

  # JL-BALANCE under attack (same attack model as other algorithms)
  python decentralized_fl_sim.py \
      --dataset femnist --num-nodes 10 --rounds 30 --local-epochs 1 \
      --agg jl-balance --graph erdos --p 0.3 \
      --attack-percentage 0.3 --attack-type directed_deviation --jl-projection-dim 150
"""
from __future__ import annotations

import argparse
import random
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, RandomSampler

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


# ---------------------------- BALANCE Implementation ---------------------------- #

@dataclass
class BALANCEConfig:
    """Configuration for BALANCE algorithm."""
    gamma: float = 2.0          # Base similarity threshold multiplier
    kappa: float = 1.0          # Exponential decay rate for threshold tightening
    alpha: float = 0.5          # Weight for own update vs neighbors (0.5 = equal weight)
    min_neighbors: int = 1      # Minimum neighbors to accept before fallback


@dataclass
class JLBALANCEConfig(BALANCEConfig):
    """Configuration for JL-BALANCE algorithm."""
    # BALANCE parameters inherited from BALANCEConfig

    # JL-specific parameters
    projection_dim: int = 200           # Target projection dimension k
    network_seed: int = 42              # Shared seed for projection matrix
    epsilon: float = 0.1                # JL distortion parameter
    reconstruction_method: str = "pinv" # "pinv" or "iterative"


class BALANCE:
    """
    BALANCE algorithm implementation.

    BALANCE filters neighbors based on L2 distance similarity to own update,
    with exponentially tightening thresholds over time.

    Core similarity condition:
    ||w_i - w_j|| <= gamma * exp(-kappa * lambda(t)) * ||w_i||

    where lambda(t) = current_round / total_rounds
    """

    def __init__(self, node_id: str, config: BALANCEConfig, total_rounds: int):
        self.node_id = node_id
        self.config = config
        self.total_rounds = total_rounds

        # Statistics tracking
        self.acceptance_history = []
        self.threshold_history = []
        self.neighbor_distances = defaultdict(list)

    def compute_similarity_threshold(self, own_update: Dict[str, torch.Tensor],
                                     current_round: int) -> float:
        """
        Compute time-adaptive similarity threshold.

        Args:
            own_update: Node's own model update
            current_round: Current training round (1-indexed)

        Returns:
            Similarity threshold for accepting neighbors
        """
        # Compute lambda(t) = current_round / total_rounds
        lambda_t = current_round / max(1, self.total_rounds)

        # Exponential decay: gamma * exp(-kappa * lambda(t))
        threshold_factor = self.config.gamma * np.exp(-self.config.kappa * lambda_t)

        # Scale by own update magnitude: threshold_factor * ||w_i||
        own_update_norm = self._compute_l2_norm(own_update)
        threshold = threshold_factor * own_update_norm

        self.threshold_history.append(threshold)
        return threshold

    def _compute_l2_norm(self, model_update: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of flattened model parameters."""
        total_norm_sq = 0.0
        for param in model_update.values():
            if param.numel() > 0:
                total_norm_sq += torch.sum(param * param).item()
        return np.sqrt(total_norm_sq)

    def _compute_l2_distance(self, update1: Dict[str, torch.Tensor],
                             update2: Dict[str, torch.Tensor]) -> float:
        """Compute L2 distance between two model updates."""
        total_dist_sq = 0.0

        # Only compute distance for common parameters
        common_keys = set(update1.keys()) & set(update2.keys())

        for key in common_keys:
            diff = update1[key] - update2[key]
            total_dist_sq += torch.sum(diff * diff).item()

        return np.sqrt(total_dist_sq)

    def filter_neighbors(self, own_update: Dict[str, torch.Tensor],
                         neighbor_updates: Dict[str, Dict[str, torch.Tensor]],
                         current_round: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Filter neighbor updates based on similarity to own update.

        Args:
            own_update: Node's own model update
            neighbor_updates: Dictionary mapping neighbor_id to their updates
            current_round: Current training round

        Returns:
            Dictionary of accepted neighbor updates
        """
        threshold = self.compute_similarity_threshold(own_update, current_round)
        accepted_neighbors = {}
        distances = {}

        for neighbor_id, neighbor_update in neighbor_updates.items():
            # Compute L2 distance between updates
            distance = self._compute_l2_distance(own_update, neighbor_update)
            distances[neighbor_id] = distance

            # Store distance history for analysis
            self.neighbor_distances[neighbor_id].append(distance)

            # Accept if within threshold
            if distance <= threshold:
                accepted_neighbors[neighbor_id] = neighbor_update

        # Track acceptance statistics
        acceptance_rate = len(accepted_neighbors) / max(1, len(neighbor_updates))
        self.acceptance_history.append(acceptance_rate)

        if len(accepted_neighbors) < self.config.min_neighbors and neighbor_updates:
            # Fallback: accept closest neighbor if too few accepted
            closest_neighbor = min(distances.items(), key=lambda x: x[1])
            accepted_neighbors[closest_neighbor[0]] = neighbor_updates[closest_neighbor[0]]

        return accepted_neighbors

    def aggregate_updates(self, own_update: Dict[str, torch.Tensor],
                          accepted_neighbors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate own update with accepted neighbor updates.

        Uses BALANCE aggregation rule:
        w_i^{t+1} = alpha * w_i^{t+1/2} + (1-alpha) * (1/|S_i|) * sum(w_j^{t+1/2})

        Args:
            own_update: Node's own model update
            accepted_neighbors: Dictionary of accepted neighbor updates

        Returns:
            Aggregated model update
        """
        if not accepted_neighbors:
            return own_update

        # Compute neighbor average
        neighbor_avg = {}
        num_neighbors = len(accepted_neighbors)

        for key in own_update.keys():
            neighbor_sum = torch.zeros_like(own_update[key])

            for neighbor_update in accepted_neighbors.values():
                if key in neighbor_update:
                    neighbor_sum += neighbor_update[key]

            neighbor_avg[key] = neighbor_sum / num_neighbors

        # BALANCE aggregation: alpha * own + (1-alpha) * neighbor_avg
        aggregated_update = {}
        for key in own_update.keys():
            aggregated_update[key] = (self.config.alpha * own_update[key] +
                                      (1 - self.config.alpha) * neighbor_avg[key])

        return aggregated_update

    def get_statistics(self) -> Dict:
        """Get BALANCE algorithm statistics."""
        return {
            "node_id": self.node_id,
            "mean_acceptance_rate": np.mean(self.acceptance_history) if self.acceptance_history else 0.0,
            "current_threshold": self.threshold_history[-1] if self.threshold_history else 0.0,
            "total_rounds_processed": len(self.acceptance_history)
        }


class LightweightJLBALANCE:
    """
    Lightweight JL-BALANCE: High-performance Byzantine-robust aggregation.

    Uses Johnson-Lindenstrauss random projection for:
    - Fast distance computations in low-dimensional space: O(k) vs O(d)
    - Massive communication savings: share k-dim vectors vs d-dim parameters
    - Efficient memory usage: store projected neighbors
    - Scalable to very large models
    """

    def __init__(self, node_id: str, config: JLBALANCEConfig, total_rounds: int, model_dim: int):
        self.node_id = node_id
        self.config = config
        self.total_rounds = total_rounds
        self.model_dim = model_dim

        # Generate shared projection matrix (same across all nodes)
        self.projection_matrix, self.reconstruction_matrix = self._generate_projection_matrices()

        # BALANCE tracking (in projected space)
        self.acceptance_history = []
        self.threshold_history = []
        self.neighbor_distances = defaultdict(list)

        # Performance tracking
        self.projection_time = 0.0
        self.filtering_time = 0.0
        self.reconstruction_time = 0.0

    def _generate_projection_matrices(self):
        """Generate shared random projection matrix and its reconstruction matrix."""
        # Use shared network seed for deterministic generation across all nodes
        np.random.seed(self.config.network_seed)

        # Gaussian random projection matrix (normalized)
        R = np.random.randn(self.config.projection_dim, self.model_dim) / np.sqrt(self.config.projection_dim)

        # Compute pseudoinverse for reconstruction
        if self.config.reconstruction_method == "pinv":
            R_recon = np.linalg.pinv(R)
        else:
            # For very large matrices, iterative methods might be better
            R_recon = R.T  # Transpose as simple approximation

        return R, R_recon

    def flatten_model_update(self, model_update: Dict[str, torch.Tensor]) -> np.ndarray:
        """Flatten model parameters into a single vector."""
        flattened_parts = []
        for param in model_update.values():
            flattened_parts.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(flattened_parts)

    def unflatten_to_model_update(self, flattened: np.ndarray,
                                  reference_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reshape flattened vector back to model parameter structure."""
        result = {}
        start_idx = 0

        for param_name, param_tensor in reference_update.items():
            param_shape = param_tensor.shape
            param_size = param_tensor.numel()

            # Extract the slice and reshape
            param_slice = flattened[start_idx:start_idx + param_size]
            param_reshaped = param_slice.reshape(param_shape)

            # Convert back to tensor with same device/dtype
            result[param_name] = torch.tensor(param_reshaped,
                                              dtype=param_tensor.dtype,
                                              device=param_tensor.device)

            start_idx += param_size

        return result

    def project_update(self, model_update: Dict[str, torch.Tensor]) -> np.ndarray:
        """Project model update to low-dimensional space for sharing."""
        start_time = time.time()

        # Flatten parameters
        flattened = self.flatten_model_update(model_update)

        # Project: z = R @ w
        projected = self.projection_matrix @ flattened

        self.projection_time += time.time() - start_time
        return projected

    def compute_projected_threshold(self, own_projected: np.ndarray, current_round: int) -> float:
        """Compute BALANCE threshold in projected space."""
        # Time-adaptive threshold: γ * exp(-κ * λ(t)) * ||z_i||
        lambda_t = current_round / max(1, self.total_rounds)
        threshold_factor = self.config.gamma * np.exp(-self.config.kappa * lambda_t)
        own_norm = np.linalg.norm(own_projected)
        threshold = threshold_factor * own_norm

        self.threshold_history.append(threshold)
        return threshold

    def filter_neighbors_in_projected_space(self, own_projected: np.ndarray,
                                            neighbor_projected_dict: Dict[str, np.ndarray],
                                            current_round: int) -> Dict[str, np.ndarray]:
        """Perform BALANCE filtering in projected space - the core speedup!"""
        start_time = time.time()

        threshold = self.compute_projected_threshold(own_projected, current_round)
        accepted_neighbors = {}
        distances = {}

        for neighbor_id, neighbor_projected in neighbor_projected_dict.items():
            # Fast distance computation in k-dimensional space: O(k) instead of O(d)!
            distance = np.linalg.norm(own_projected - neighbor_projected)
            distances[neighbor_id] = distance

            # Track distance history for analysis
            self.neighbor_distances[neighbor_id].append(distance)

            # BALANCE acceptance decision
            if distance <= threshold:
                accepted_neighbors[neighbor_id] = neighbor_projected

        # Track acceptance statistics
        acceptance_rate = len(accepted_neighbors) / max(1, len(neighbor_projected_dict))
        self.acceptance_history.append(acceptance_rate)

        # Fallback mechanism: ensure minimum neighbors
        if len(accepted_neighbors) < self.config.min_neighbors and neighbor_projected_dict:
            closest_neighbor_id = min(distances.items(), key=lambda x: x[1])[0]
            if closest_neighbor_id not in accepted_neighbors:
                accepted_neighbors[closest_neighbor_id] = neighbor_projected_dict[closest_neighbor_id]

        self.filtering_time += time.time() - start_time
        return accepted_neighbors

    def aggregate_in_projected_space(self, own_projected: np.ndarray,
                                     accepted_neighbors_projected: Dict[str, np.ndarray]) -> np.ndarray:
        """Perform BALANCE aggregation in projected space - super fast!"""
        if not accepted_neighbors_projected:
            return own_projected

        # Compute neighbor average in projected space: O(k) instead of O(d)
        neighbor_projections = list(accepted_neighbors_projected.values())
        neighbor_avg_projected = np.mean(neighbor_projections, axis=0)

        # BALANCE aggregation rule: α * own + (1-α) * neighbor_avg
        aggregated_projected = (self.config.alpha * own_projected +
                                (1 - self.config.alpha) * neighbor_avg_projected)

        return aggregated_projected

    def reconstruct_from_projected_space(self, aggregated_projected: np.ndarray,
                                         reference_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reconstruct full parameters from projected aggregation."""
        start_time = time.time()

        if self.config.reconstruction_method == "pinv":
            # Standard pseudoinverse reconstruction
            reconstructed_flat = self.reconstruction_matrix @ aggregated_projected
        elif self.config.reconstruction_method == "iterative":
            # Iterative refinement for better reconstruction
            reconstructed_flat = self._iterative_reconstruction(aggregated_projected)
        else:
            raise ValueError(f"Unknown reconstruction method: {self.config.reconstruction_method}")

        # Reshape back to model structure
        reconstructed_params = self.unflatten_to_model_update(reconstructed_flat, reference_update)

        self.reconstruction_time += time.time() - start_time
        return reconstructed_params

    def _iterative_reconstruction(self, target_projected: np.ndarray, num_iterations: int = 3) -> np.ndarray:
        """Iterative refinement for better reconstruction quality."""
        # Start with pseudoinverse solution
        current_recon = self.reconstruction_matrix @ target_projected

        # Refine iteratively
        for _ in range(num_iterations):
            # Project current reconstruction
            current_proj = self.projection_matrix @ current_recon

            # Compute error in projected space
            error_proj = target_projected - current_proj

            # Apply correction in original space
            correction = self.reconstruction_matrix @ error_proj
            current_recon += 0.5 * correction  # Damped update

        return current_recon

    def jl_balance_round(self, own_update: Dict[str, torch.Tensor],
                         neighbor_projected_dict: Dict[str, np.ndarray],
                         current_round: int) -> Dict[str, torch.Tensor]:
        """
        Complete JL-BALANCE round: the main entry point.

        Input: own_update (full parameters), neighbor_projected_dict (received projections)
        Output: aggregated model update (full parameters)
        """
        # Step 1: Project own update to low-dimensional space
        own_projected = self.project_update(own_update)

        # Step 2: Filter neighbors using fast projected distances
        accepted_neighbors_projected = self.filter_neighbors_in_projected_space(
            own_projected, neighbor_projected_dict, current_round
        )

        # Step 3: Aggregate in projected space
        aggregated_projected = self.aggregate_in_projected_space(
            own_projected, accepted_neighbors_projected
        )

        # Step 4: Reconstruct final parameters
        final_update = self.reconstruct_from_projected_space(aggregated_projected, own_update)

        return final_update

    def get_projected_update_for_sharing(self, model_update: Dict[str, torch.Tensor]) -> np.ndarray:
        """Get projected version of update for sharing with neighbors."""
        return self.project_update(model_update)

    def get_statistics(self) -> Dict:
        """Get detailed performance statistics."""
        total_time = self.projection_time + self.filtering_time + self.reconstruction_time

        return {
            "node_id": self.node_id,
            "algorithm": "JL-BALANCE",
            "total_rounds_processed": len(self.acceptance_history),

            # BALANCE statistics
            "mean_acceptance_rate": np.mean(self.acceptance_history) if self.acceptance_history else 0.0,
            "current_threshold": self.threshold_history[-1] if self.threshold_history else 0.0,

            # Performance statistics
            "total_computation_time": total_time,
            "projection_time": self.projection_time,
            "filtering_time": self.filtering_time,
            "reconstruction_time": self.reconstruction_time,

            "projection_time_fraction": self.projection_time / max(total_time, 1e-6),
            "filtering_time_fraction": self.filtering_time / max(total_time, 1e-6),
            "reconstruction_time_fraction": self.reconstruction_time / max(total_time, 1e-6),

            # Compression statistics
            "original_dimension": self.model_dim,
            "projected_dimension": self.config.projection_dim,
            "compression_ratio": self.model_dim / self.config.projection_dim,
            "communication_reduction": self.model_dim / self.config.projection_dim
        }


def calculate_model_dimension(model: nn.Module) -> int:
    """Calculate total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


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
    """Implementation of local model poisoning attacks for decentralized learning."""

    def __init__(self, num_nodes: int, attack_percentage: float, attack_type: str,
                 lambda_param: float = 1.0, seed: int = 42):
        self.num_nodes = num_nodes
        self.attack_percentage = attack_percentage
        self.attack_type = attack_type
        self.lambda_param = lambda_param

        # Deterministically select which nodes are compromised
        num_compromised = int(num_nodes * attack_percentage)
        if num_compromised == 0 and attack_percentage > 0:
            num_compromised = 1

        random.seed(seed)
        self.compromised_nodes = set(random.sample(range(num_nodes), min(num_compromised, num_nodes)))
        print(f"Attack: Compromised {len(self.compromised_nodes)}/{num_nodes} nodes: {sorted(self.compromised_nodes)}")

        self.previous_neighborhood_avgs = {}
        self.compromised_node_states = {}

    def update_compromised_states(self, round_num: int, node_states: List[Dict[str, torch.Tensor]]):
        """Collect states from all compromised nodes for global coordination."""
        self.compromised_node_states[round_num] = {}
        for node_id in self.compromised_nodes:
            self.compromised_node_states[round_num][node_id] = {k: v.clone() for k, v in node_states[node_id].items()}

    def estimate_global_directions_from_compromised(self, round_num: int) -> Optional[Dict[str, torch.Tensor]]:
        """Estimate global changing directions using only compromised nodes' data."""
        if round_num < 1 or round_num not in self.compromised_node_states:
            return None

        prev_round = round_num - 1
        if prev_round not in self.compromised_node_states:
            return None

        current_compromised = list(self.compromised_node_states[round_num].values())
        previous_compromised = list(self.compromised_node_states[prev_round].values())

        if not previous_compromised or not current_compromised:
            return None

        current_mean = average_states(current_compromised)
        previous_mean = average_states(previous_compromised)

        directions = {}
        for key in current_mean.keys():
            directions[key] = torch.where(
                current_mean[key] > previous_mean[key],
                torch.ones_like(current_mean[key]),
                -torch.ones_like(current_mean[key])
            )

        return directions

    def estimate_neighborhood_directions(self, node_id: int, current_neigh_states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Estimate directions using only neighborhood information."""
        if not current_neigh_states:
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
            directions = {}
            for key in current_neigh_avg.keys():
                directions[key] = torch.sign(torch.randn_like(current_neigh_avg[key]))
            self.previous_neighborhood_avgs[node_id] = {k: v.clone() for k, v in current_neigh_avg.items()}
            return directions

        directions = {}
        for key in current_neigh_avg.keys():
            directions[key] = torch.where(
                current_neigh_avg[key] > previous_neigh_avg[key],
                torch.ones_like(current_neigh_avg[key]),
                -torch.ones_like(current_neigh_avg[key])
            )

        self.previous_neighborhood_avgs[node_id] = {k: v.clone() for k, v in current_neigh_avg.items()}
        return directions

    def craft_malicious_models_decentralized(self, node_id: int, neighborhood_indices: List[int],
                                             honest_neigh_states: List[Dict[str, torch.Tensor]],
                                             round_num: int) -> List[Dict[str, torch.Tensor]]:
        """Craft malicious models for compromised nodes in this neighborhood."""
        num_compromised_in_neigh = len([i for i in neighborhood_indices if i in self.compromised_nodes])
        if num_compromised_in_neigh == 0:
            return []

        if self.attack_type == "random":
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

        # Directed deviation attack
        directions = self.estimate_global_directions_from_compromised(round_num)
        if directions is None:
            directions = self.estimate_neighborhood_directions(node_id, honest_neigh_states)

        if not directions:
            return []

        if honest_neigh_states:
            reference_state = average_states(honest_neigh_states)
        elif self.compromised_node_states and round_num in self.compromised_node_states:
            compromised_states = list(self.compromised_node_states[round_num].values())
            reference_state = average_states(compromised_states)
        else:
            return []

        malicious_states = []
        primary_malicious = {}
        for key in reference_state.keys():
            primary_malicious[key] = reference_state[key] - self.lambda_param * directions[key]

        malicious_states.append(primary_malicious)

        epsilon = 0.01
        for _ in range(num_compromised_in_neigh - 1):
            supporting_model = {}
            for key in primary_malicious.keys():
                noise = torch.randn_like(primary_malicious[key]) * epsilon
                supporting_model[key] = primary_malicious[key] + noise
            malicious_states.append(supporting_model)

        return malicious_states[:num_compromised_in_neigh]


# ---------------------------- Aggregation Functions ---------------------------- #

def balance_aggregation_step(models: List[nn.Module], graph: Graph,
                             balance_monitors: Dict[str, BALANCE],
                             round_num: int, attacker: Optional[LocalModelPoisoningAttacker] = None):
    """Perform one round of BALANCE aggregation."""
    states = [get_state(m) for m in models]

    if attacker:
        attacker.update_compromised_states(round_num, states)

    new_states = []

    for i in range(graph.n):
        node_id = str(i)
        neighbors = graph.neighbors[i]
        own_update = states[i]

        # Get neighbor updates with potential attacks
        if attacker and any(j in attacker.compromised_nodes for j in neighbors):
            compromised_in_neigh = [j for j in neighbors if j in attacker.compromised_nodes]
            honest_in_neigh = [j for j in neighbors if j not in attacker.compromised_nodes]
            honest_neigh_states = [states[j] for j in honest_in_neigh]

            malicious_states = attacker.craft_malicious_models_decentralized(
                i, [i] + neighbors, honest_neigh_states, round_num
            )

            neighbor_updates = {}
            malicious_idx = 0

            for j in neighbors:
                if j in attacker.compromised_nodes and malicious_idx < len(malicious_states):
                    neighbor_updates[str(j)] = malicious_states[malicious_idx]
                    malicious_idx += 1
                else:
                    neighbor_updates[str(j)] = states[j]
        else:
            neighbor_updates = {str(j): states[j] for j in neighbors}

        if neighbor_updates and node_id in balance_monitors:
            balance_monitor = balance_monitors[node_id]
            accepted_neighbors = balance_monitor.filter_neighbors(
                own_update, neighbor_updates, round_num
            )

            aggregated_update = balance_monitor.aggregate_updates(
                own_update, accepted_neighbors
            )

            new_states.append(aggregated_update)
        else:
            new_states.append(own_update)

    for model, state in zip(models, new_states):
        set_state(model, state)


def jl_balance_aggregation_step(models: List[nn.Module], graph: Graph,
                                jl_monitors: Dict[str, LightweightJLBALANCE],
                                round_num: int, attacker: Optional[LocalModelPoisoningAttacker] = None):
    """
    CORRECTED JL-BALANCE aggregation with proper attack model.

    Key fix: Attacks now happen at PARAMETER level before projection,
    ensuring fair comparison with other aggregation algorithms.
    """

    # Phase 1: Get current model states
    states = [get_state(m) for m in models]

    # Phase 2: Apply attacks BEFORE projection (same as other algorithms)
    if attacker:
        attacker.update_compromised_states(round_num, states)

        # Replace compromised node states with malicious parameter updates
        for i in range(graph.n):
            if i in attacker.compromised_nodes:
                neighbors = [i] + graph.neighbors[i]
                honest_neighbors = [j for j in neighbors if j not in attacker.compromised_nodes]
                honest_neigh_states = [states[j] for j in honest_neighbors]

                malicious_states = attacker.craft_malicious_models_decentralized(
                    i, neighbors, honest_neigh_states, round_num
                )

                if malicious_states:
                    states[i] = malicious_states[0]  # Replace with malicious parameters

    # Phase 3: Project ALL updates (honest and malicious) - no preferential treatment
    projected_updates = {}
    for i in range(graph.n):
        node_id = str(i)
        if node_id in jl_monitors:
            # Project whatever update node i has (honest or malicious)
            projected = jl_monitors[node_id].get_projected_update_for_sharing(states[i])
            projected_updates[i] = projected

    # Phase 4: Each node performs JL-BALANCE with received projections
    new_states = []
    for i in range(graph.n):
        node_id = str(i)
        neighbors = graph.neighbors[i]

        if node_id not in jl_monitors:
            # Fallback if no JL monitor
            new_states.append(states[i])
            continue

        # Get neighbor projected updates (includes projected malicious updates)
        neighbor_projected_dict = {str(j): projected_updates[j] for j in neighbors}

        # Perform JL-BALANCE round with projected malicious updates
        jl_monitor = jl_monitors[node_id]
        aggregated_update = jl_monitor.jl_balance_round(
            states[i], neighbor_projected_dict, round_num
        )

        new_states.append(aggregated_update)

    # Phase 5: Update models with aggregated states
    for model, state in zip(models, new_states):
        set_state(model, state)


def decentralized_fedavg_step(models: List[nn.Module], graph: Graph, round_num: int = 0,
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


def gossip_round(models: List[nn.Module], graph: Graph, steps: int = 1, round_num: int = 0, seed: int = 42,
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


def decentralized_krum_step(models: List[nn.Module], graph: Graph,
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

    # Initialize models first (needed for model dimension calculation)
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

    # Calculate model dimension for JL-BALANCE
    model_dim = calculate_model_dimension(models[0])

    # Initialize aggregation monitors
    balance_monitors = {}
    jl_monitors = {}

    if args.agg == "balance":
        balance_config = BALANCEConfig(
            gamma=args.balance_gamma,
            kappa=args.balance_kappa,
            alpha=args.balance_alpha
        )
        for i in range(args.num_nodes):
            balance_monitors[str(i)] = BALANCE(str(i), balance_config, args.rounds)
        print(f"BALANCE algorithm: {args.agg}")
        print(f"  - Gamma: {args.balance_gamma}")
        print(f"  - Kappa: {args.balance_kappa}")
        print(f"  - Alpha: {args.balance_alpha}")
        print(f"  - Model dimension: {model_dim:,} parameters")

    elif args.agg == "jl-balance":
        jl_config = JLBALANCEConfig(
            # BALANCE parameters
            gamma=args.balance_gamma,
            kappa=args.balance_kappa,
            alpha=args.balance_alpha,

            # JL parameters
            projection_dim=args.jl_projection_dim,
            network_seed=args.seed,  # Use training seed for shared projection
            epsilon=0.1,
            reconstruction_method=args.jl_reconstruction
        )

        for i in range(args.num_nodes):
            jl_monitors[str(i)] = LightweightJLBALANCE(str(i), jl_config, args.rounds, model_dim)

        compression_ratio = model_dim / args.jl_projection_dim
        print(f"🚀 JL-BALANCE algorithm initialized!")
        print(f"  - Model dimension: {model_dim:,} parameters")
        print(f"  - Projected dimension: {args.jl_projection_dim}")
        print(f"  - Compression ratio: {compression_ratio:.1f}x")
        print(f"  - Communication savings: {compression_ratio:.1f}x less bandwidth per message")
        print(f"  - Network seed: {args.seed} (shared across all nodes)")
        print(f"  - Reconstruction method: {args.jl_reconstruction}")
        print(f"  - BALANCE params: γ={args.balance_gamma}, κ={args.balance_kappa}, α={args.balance_alpha}")
        print(f"  - ✅ CORRECTED: Attacks applied at parameter level before projection")

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
            decentralized_fedavg_step(models, graph, r, attacker)
        elif args.agg == "gossip":
            gossip_round(models, graph, steps=args.gossip_steps,
                         round_num=r, seed=args.seed, attacker=attacker)
        elif args.agg == "krum":
            decentralized_krum_step(models, graph, args.pct_compromised, r, attacker)
        elif args.agg == "balance":
            balance_aggregation_step(models, graph, balance_monitors, r, attacker)
        elif args.agg == "jl-balance":
            jl_balance_aggregation_step(models, graph, jl_monitors, r, attacker)
        else:
            raise ValueError("agg must be 'd-fedavg', 'gossip', 'krum', 'balance', or 'jl-balance'")

        # Evaluation phase
        accs = []
        losses = []
        correct_totals = []
        for i, m in enumerate(models):
            acc, loss, correct, total = evaluate(m, test_loaders[i], dev)
            accs.append(acc)
            losses.append(loss)
            correct_totals.append((correct, total))

        print(f"Round {r:03d}: test acc mean={np.mean(accs):.4f} ± {np.std(accs):.4f} | "
              f"min={np.min(accs):.4f} max={np.max(accs):.4f}")
        print(f"         : test loss mean={np.mean(losses):.4f} ± {np.std(losses):.4f}")

        if args.verbose:
            acc_strs = [f"{acc:.6f}" for acc in accs]
            print(f"         : individual accs = {acc_strs}")
            print(f"         : correct/total = {correct_totals}")

            if attacker:
                compromised_accs = [accs[i] for i in attacker.compromised_nodes]
                honest_accs = [accs[i] for i in range(args.num_nodes) if i not in attacker.compromised_nodes]
                if compromised_accs and honest_accs:
                    print(f"         : compromised nodes: mean acc={np.mean(compromised_accs):.4f}")
                    print(f"         : honest nodes: mean acc={np.mean(honest_accs):.4f}")

            # Show algorithm-specific information
            if args.agg == "balance" and balance_monitors:
                balance_stats = []
                for node_id, monitor in balance_monitors.items():
                    stats = monitor.get_statistics()
                    balance_stats.append(f"Node {node_id}: acc_rate={stats['mean_acceptance_rate']:.3f}")
                print(f"         : balance stats = {balance_stats[:5]}...")

            elif args.agg == "jl-balance" and jl_monitors:
                jl_stats = []
                for node_id, monitor in jl_monitors.items():
                    stats = monitor.get_statistics()
                    jl_stats.append(f"Node {node_id}: acc_rate={stats['mean_acceptance_rate']:.3f}")
                print(f"         : jl-balance stats = {jl_stats[:3]}...")

    # Final evaluation and summary
    accs = []
    for i, m in enumerate(models):
        acc, _, _, _ = evaluate(m, test_loaders[i], dev)
        accs.append(acc)

    print("\n=== FINAL RESULTS ===")
    print(f"Dataset: {args.dataset}, Nodes: {args.num_nodes}, Graph: {args.graph}, Aggregation: {args.agg}")
    if attacker:
        print(f"Attack: {args.attack_type}, {args.attack_percentage*100:.1f}% compromised, lambda={args.attack_lambda}")
        compromised_accs = [accs[i] for i in attacker.compromised_nodes]
        honest_accs = [accs[i] for i in range(args.num_nodes) if i not in attacker.compromised_nodes]
        if compromised_accs and honest_accs:
            print(f"Final accuracy - Compromised: {np.mean(compromised_accs):.4f}, Honest: {np.mean(honest_accs):.4f}")
            print(f"Attack effectiveness: {max(0, np.mean(honest_accs) - np.mean(compromised_accs)):.4f} accuracy difference")
    else:
        print("No attack (clean run)")
    print(f"Overall test accuracy: mean={np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # Print algorithm-specific summary
    if args.agg == "balance" and balance_monitors:
        print(f"\n=== BALANCE ALGORITHM SUMMARY ===")
        all_acceptance_rates = []
        all_thresholds = []

        for node_id, monitor in balance_monitors.items():
            stats = monitor.get_statistics()
            all_acceptance_rates.append(stats["mean_acceptance_rate"])

            if monitor.threshold_history:
                all_thresholds.append(monitor.threshold_history[-1])

            # Show detailed stats for first few nodes
            if int(node_id) < 5:
                print(f"Node {node_id}: mean_acceptance={stats['mean_acceptance_rate']:.3f}, "
                      f"rounds_processed={stats['total_rounds_processed']}")

        if all_acceptance_rates:
            print(f"Network-wide acceptance statistics:")
            print(f"  - Mean acceptance rate: {np.mean(all_acceptance_rates):.3f} ± {np.std(all_acceptance_rates):.3f}")
            print(f"  - Min acceptance rate: {np.min(all_acceptance_rates):.3f}")
            print(f"  - Max acceptance rate: {np.max(all_acceptance_rates):.3f}")

        if all_thresholds:
            print(f"  - Final threshold values: mean={np.mean(all_thresholds):.3f} ± {np.std(all_thresholds):.3f}")

        print(f"BALANCE insights:")
        print(f"  - Uses uniform L2 distance for similarity measurement")
        print(f"  - Threshold decay: gamma={args.balance_gamma} * exp(-{args.balance_kappa} * t)")

    elif args.agg == "jl-balance" and jl_monitors:
        print(f"\n=== JL-BALANCE ALGORITHM SUMMARY ===")
        all_acceptance_rates = []
        all_thresholds = []
        all_compression_ratios = []
        total_proj_time = 0.0
        total_filter_time = 0.0
        total_recon_time = 0.0

        for node_id, monitor in jl_monitors.items():
            stats = monitor.get_statistics()
            all_acceptance_rates.append(stats["mean_acceptance_rate"])
            all_compression_ratios.append(stats["compression_ratio"])

            total_proj_time += stats["projection_time"]
            total_filter_time += stats["filtering_time"]
            total_recon_time += stats["reconstruction_time"]

            if monitor.threshold_history:
                all_thresholds.append(monitor.threshold_history[-1])

            # Show detailed stats for first few nodes
            if int(node_id) < 3:
                print(f"Node {node_id}: acceptance={stats['mean_acceptance_rate']:.3f}, "
                      f"compression={stats['compression_ratio']:.1f}x, "
                      f"proj_time={stats['projection_time']:.3f}s, "
                      f"filter_time={stats['filtering_time']:.3f}s")

        if all_acceptance_rates:
            print(f"\nNetwork-wide JL-BALANCE statistics:")
            print(f"  - Mean acceptance rate: {np.mean(all_acceptance_rates):.3f} ± {np.std(all_acceptance_rates):.3f}")
            print(f"  - Min acceptance rate: {np.min(all_acceptance_rates):.3f}")
            print(f"  - Max acceptance rate: {np.max(all_acceptance_rates):.3f}")
            print(f"  - Average compression ratio: {np.mean(all_compression_ratios):.1f}x")

        if all_thresholds:
            print(f"  - Final threshold values: mean={np.mean(all_thresholds):.3f} ± {np.std(all_thresholds):.3f}")

        # Performance breakdown
        total_time = total_proj_time + total_filter_time + total_recon_time
        if total_time > 0:
            print(f"\nPerformance breakdown (total across all nodes):")
            print(f"  - Projection time: {total_proj_time:.3f}s ({total_proj_time/total_time*100:.1f}%)")
            print(f"  - Filtering time: {total_filter_time:.3f}s ({total_filter_time/total_time*100:.1f}%)")
            print(f"  - Reconstruction time: {total_recon_time:.3f}s ({total_recon_time/total_time*100:.1f}%)")
            print(f"  - Total computation time: {total_time:.3f}s")

        print(f"\nJL-BALANCE insights:")
        print(f"  - Uses Johnson-Lindenstrauss projection for {model_dim} → {args.jl_projection_dim} compression")
        print(f"  - Communication savings: {model_dim//args.jl_projection_dim:.0f}x less bandwidth per message")
        print(f"  - Threshold decay in projected space: gamma={args.balance_gamma} * exp(-{args.balance_kappa} * t)")
        print(f"  - Reconstruction method: {args.jl_reconstruction}")
        print(f"  - ✅ FAIR EVALUATION: Attacks applied at parameter level (same as other algorithms)")

    # Performance analysis
    if attacker and args.agg in ["balance", "jl-balance"]:
        monitors = balance_monitors if args.agg == "balance" else jl_monitors
        if monitors and honest_accs and compromised_accs:
            avg_acceptance = np.mean(all_acceptance_rates)
            if avg_acceptance < 0.3:
                print(f"\n⚠️  WARNING: Very low acceptance rate ({avg_acceptance:.3f}) may indicate:")
                print(f"   - Aggressive filtering (possibly good against attacks)")
                print(f"   - Network instability or poor parameter tuning")
            elif avg_acceptance > 0.9:
                print(f"\n📝 NOTE: High acceptance rate ({avg_acceptance:.3f}) suggests:")
                print(f"   - Either weak attacks or highly similar model updates")
                print(f"   - Consider increasing attack strength for robustness testing")

            attack_mitigation = max(0, np.mean(honest_accs) - np.mean(accs))
            if attack_mitigation > 0.05:
                print(f"\n✅ ANALYSIS: Algorithm successfully mitigated attack impact by {attack_mitigation:.4f} accuracy")
                print(f"   - Attack detection and filtering appears effective")
            else:
                print(f"\n❌ ANALYSIS: Limited attack mitigation detected")
                print(f"   - Consider tuning algorithm parameters or using stronger defenses")

    # JL-BALANCE specific insights
    if args.agg == "jl-balance":
        print(f"\n🚀 JL-BALANCE PERFORMANCE SUMMARY:")
        estimated_orig_ops = args.num_nodes * len(graph.neighbors[0]) * model_dim * args.rounds
        estimated_jl_ops = args.num_nodes * len(graph.neighbors[0]) * args.jl_projection_dim * args.rounds
        theoretical_speedup = estimated_orig_ops / estimated_jl_ops

        print(f"  - Theoretical distance computation speedup: {theoretical_speedup:.1f}x")
        print(f"  - Actual filtering time per node: {total_filter_time/args.num_nodes:.4f}s")
        print(f"  - Communication bandwidth saved: {(model_dim - args.jl_projection_dim) * args.num_nodes * len(graph.neighbors[0]) * args.rounds:,} fewer parameters transmitted")

        if total_filter_time > 0 and total_proj_time > 0:
            efficiency_ratio = total_filter_time / (total_proj_time + total_recon_time)
            if efficiency_ratio > 1.0:
                print(f"  - ✅ Filtering is {efficiency_ratio:.1f}x faster than projection+reconstruction overhead")
            else:
                print(f"  - ⚠️  Projection+reconstruction overhead is {1/efficiency_ratio:.1f}x the filtering time")
                print(f"     Consider: JL-BALANCE trades computation for communication efficiency")

        print(f"  - 🎯 KEY INSIGHT: JL-BALANCE optimizes for communication/memory, not raw computation")
        print(f"     Best suited for bandwidth-limited or memory-constrained environments")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Decentralized Learning Simulator with BALANCE and JL-BALANCE")

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
                   choices=["d-fedavg", "gossip", "krum", "balance", "jl-balance"],
                   default="d-fedavg",
                   help="Aggregation algorithm")
    p.add_argument("--gossip-steps", type=int, default=10,
                   help="Number of random edge gossips per round (for gossip aggregation)")
    p.add_argument("--pct-compromised", type=float, default=0.0,
                   help="Max percentage of compromised models per neighborhood for Krum")

    # BALANCE algorithm parameters (shared by both BALANCE and JL-BALANCE)
    p.add_argument("--balance-gamma", type=float, default=2.0,
                   help="BALANCE similarity threshold multiplier")
    p.add_argument("--balance-kappa", type=float, default=1.0,
                   help="BALANCE threshold decay rate")
    p.add_argument("--balance-alpha", type=float, default=0.5,
                   help="BALANCE weight for own vs neighbor updates")

    # JL-BALANCE specific parameters
    p.add_argument("--jl-projection-dim", type=int, default=200,
                   help="JL-BALANCE projected dimension k (lower = more compression)")
    p.add_argument("--jl-reconstruction", type=str, choices=["pinv", "iterative"], default="pinv",
                   help="JL-BALANCE reconstruction method")

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
