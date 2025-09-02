#!/usr/bin/env python3
"""
Decentralized Learning Simulator with Local Model Poisoning Attacks

Supports three aggregation strategies over a peer graph:
  1) Decentralized FedAvg (synchronous neighbor averaging per round)
  2) Gossip averaging (asynchronous-style, simulated with K random edge gossips per round)
  3) Decentralized Krum (Byzantine-robust aggregation, selects most similar model in neighborhood)

Attack Implementation:
  - Local model poisoning attacks based on "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
  - Partial knowledge assumption (attacker only knows compromised nodes' data)
  - Global coordination among compromised nodes
  - Neighborhood-based direction estimation for decentralized setting

LEAF Datasets:
  - FEMNIST: Handwritten characters by writer (non-IID, natural federated dataset)
  - CelebA: Celebrity face attributes classification (non-IID, natural federated dataset)

Features:
  - Uses LEAF's natural client partitioning (writer-based for FEMNIST)
  - Client-specific train/test splits following LEAF methodology
  - LEAF's standard model architectures (CNN for FEMNIST, CNN for CelebA)

Example usage:
  # Clean run without attacks
  python decentralized_fl_sim.py \
      --dataset femnist --num-nodes 8 --rounds 20 --local-epochs 1 \
      --agg krum --graph ring --lr 0.01

  # Attack 25% of nodes with directed deviation attack
  python decentralized_fl_sim.py \
      --dataset femnist --num-nodes 8 --rounds 20 --local-epochs 1 \
      --agg krum --graph ring --attack-percentage 0.25 \
      --attack-type directed_deviation --attack-lambda 2.0

  # Gossip with attack
  python decentralized_fl_sim.py \
      --dataset celeba --num-nodes 10 --rounds 30 --local-epochs 1 \
      --agg gossip --gossip-steps 50 --graph erdos --p 0.3 \
      --attack-percentage 0.3 --attack-type directed_deviation

IMPORTANT: Use responsibly for research purposes only (security evaluation, robustness testing, etc.)

Notes:
  - Requires LEAF dataset preprocessing (see leaf/data/*/preprocess.sh)
  - Uses writer-based non-IID partitioning for realistic federated learning simulation
  - Attack implementation adapted from centralized to decentralized setting
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

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
    # Ensure deterministic behavior for CUDA operations
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
    """
    Create different graph topologies for decentralized learning.

    Args:
        n: Number of nodes
        kind: Graph type ('ring', 'fully', 'erdos')
        p: Edge probability for Erdős–Rényi random graph

    Returns:
        Graph object with adjacency list and edge list
    """
    kind = kind.lower()
    neighbors = [[] for _ in range(n)]
    edges: List[Tuple[int, int]] = []

    if kind == "ring":
        # Ring topology: each node connects to next node
        for i in range(n):
            j = (i + 1) % n
            neighbors[i].append(j)
            neighbors[j].append(i)
            edges.append((min(i, j), max(i, j)))
    elif kind == "fully":
        # Fully connected: each node connects to all others
        for i in range(n):
            for j in range(i + 1, n):
                neighbors[i].append(j)
                neighbors[j].append(i)
                edges.append((i, j))
    elif kind in ("erdos", "er"):
        # Erdős–Rényi random graph G(n, p)
        rng = random.Random(12345)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                    edges.append((i, j))
        # Ensure connectivity: connect isolated nodes to their next neighbor
        for i in range(n):
            if not neighbors[i]:
                j = (i + 1) % n
                neighbors[i].append(j)
                neighbors[j].append(i)
                edges.append((min(i, j), max(i, j)))
    else:
        raise ValueError(f"Unknown graph kind: {kind}")

    # Clean up: deduplicate neighbor lists and edges
    neighbors = [sorted(set(ns)) for ns in neighbors]
    edges = sorted(set(tuple(sorted(e)) for e in edges))
    return Graph(n=n, neighbors=neighbors, edges=edges)


# ---------------------------- Training helpers ---------------------------- #

def evaluate(model: nn.Module, loader: DataLoader, device_: torch.device) -> Tuple[float, float, int, int]:
    """
    Evaluate model performance on a dataset.

    Returns:
        (accuracy, average_loss, correct_count, total_count)
    """
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
    """
    Compute weighted average of model states.

    Args:
        states: List of model state dictionaries
        weights: Optional weights for averaging (default: uniform)

    Returns:
        Averaged model state
    """
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
    """
    Select model using Krum algorithm.

    Krum selects the model that has the smallest sum of distances to its closest models,
    making it robust to Byzantine failures.

    Args:
        model_states: List of model state dictionaries
        num_compromised: Maximum number of compromised models (c parameter)

    Returns:
        Index of selected model
    """
    m = len(model_states)
    c = num_compromised

    if c >= (m - 2) / 2:
        raise ValueError(f"Krum requires c < (m-2)/2, got c={c}, m={m}")

    # Compute pairwise distances between all models
    distances = []
    for i in range(m):
        model_distances = []
        for j in range(m):
            if i != j:
                dist = compute_model_distance(model_states[i], model_states[j])
                model_distances.append(dist)
        distances.append(model_distances)

    # For each model, compute score as sum of distances to m-c-2 closest models
    scores = []
    for i in range(m):
        sorted_distances = sorted(distances[i])
        num_closest = max(1, m - c - 2)  # Number of closest models to consider
        closest_distances = sorted_distances[:num_closest]
        score = sum(closest_distances)
        scores.append(score)

    # Select model with smallest score (most similar to others)
    selected_idx = scores.index(min(scores))
    return selected_idx


# ---------------------------- Attack Implementation ---------------------------- #

class LocalModelPoisoningAttacker:
    """
    Implementation of local model poisoning attacks for decentralized learning.

    Based on "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
    but adapted for decentralized setting with the following assumptions:
    - Partial knowledge: attacker only knows compromised nodes' data
    - Global coordination: compromised nodes can coordinate (attacker's privilege)
    - Neighborhood communication: each node only sees its immediate neighbors
    """

    def __init__(self, num_nodes: int, attack_percentage: float, attack_type: str,
                 lambda_param: float = 1.0, seed: int = 42):
        """
        Initialize attacker.

        Args:
            num_nodes: Total number of nodes in the network
            attack_percentage: Fraction of nodes to compromise (0.0-1.0)
            attack_type: Type of attack ('directed_deviation' or 'random')
            lambda_param: Attack strength parameter (λ from paper)
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


# ---------------------------- Enhanced Aggregation Functions ---------------------------- #

def decentralized_fedavg_step_with_attack(models: List[nn.Module], graph: Graph, round_num: int = 0,
                                          attacker: Optional[LocalModelPoisoningAttacker] = None):
    """
    Decentralized FedAvg with local model poisoning attack capability.

    Each node averages with its neighbors. If attacker is present, compromised nodes
    send crafted malicious models instead of their honest local models.
    """
    states = [get_state(m) for m in models]

    # Update attacker with compromised states for global coordination
    if attacker:
        attacker.update_compromised_states(round_num, states)

    new_states = []
    for i in range(graph.n):
        neigh = [i] + graph.neighbors[i]  # Include self in neighborhood

        # Apply attack if there are compromised nodes in this neighborhood
        if attacker and any(j in attacker.compromised_nodes for j in neigh):
            compromised_in_neigh = [j for j in neigh if j in attacker.compromised_nodes]
            honest_in_neigh = [j for j in neigh if j not in attacker.compromised_nodes]

            # Get honest neighbor states (partial knowledge assumption)
            honest_neigh_states = [states[j] for j in honest_in_neigh]

            # Craft malicious models using attack algorithm
            malicious_states = attacker.craft_malicious_models_decentralized(
                i, neigh, honest_neigh_states, round_num
            )

            # Replace compromised nodes' states with malicious ones
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
            # No attack: use honest states
            neigh_states = [states[j] for j in neigh]

        # Compute average normally (FedAvg aggregation)
        w = [1.0 / len(neigh_states)] * len(neigh_states)
        new_states.append(average_states(neigh_states, w))

    # Update all models with new averaged states
    for model, st in zip(models, new_states):
        set_state(model, st)


def gossip_round_with_attack(models: List[nn.Module], graph: Graph, steps: int = 1,
                             round_num: int = 0, seed: int = 42,
                             attacker: Optional[LocalModelPoisoningAttacker] = None):
    """
    Gossip averaging with local model poisoning attack capability.

    In gossip, pairs of nodes randomly exchange and average their models.
    Attack is applied by modifying compromised nodes' states before gossip begins.
    """
    if not graph.edges:
        return

    states = [get_state(m) for m in models]

    # Apply attacks before gossip process begins
    if attacker and attacker.compromised_nodes:
        attacker.update_compromised_states(round_num, states)

        # For each compromised node, craft malicious state
        for comp_idx in attacker.compromised_nodes:
            # Get compromised node's neighborhood (for partial knowledge)
            neighbors_of_comp = graph.neighbors[comp_idx] + [comp_idx]
            honest_neighbors = [j for j in neighbors_of_comp if j not in attacker.compromised_nodes]
            honest_neigh_states = [states[j] for j in honest_neighbors]

            # Create malicious state for this compromised node
            malicious_states = attacker.craft_malicious_models_decentralized(
                comp_idx, neighbors_of_comp, honest_neigh_states, round_num
            )

            if malicious_states:
                set_state(models[comp_idx], malicious_states[0])

    # Regular gossip process: randomly select edges and average their endpoints
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
    """
    Decentralized Krum with local model poisoning attack capability.

    Each node applies Krum selection within its neighborhood. The attack crafts
    multiple similar malicious models to increase the probability that Krum
    selects the malicious model as most "representative" of the neighborhood.
    """
    states = [get_state(m) for m in models]

    # Update attacker with current compromised node states for global coordination
    if attacker:
        attacker.update_compromised_states(round_num, states)

    new_states = []

    for i in range(graph.n):
        neigh = [i] + graph.neighbors[i]  # Include self in neighborhood
        original_neigh_states = [states[j] for j in neigh]

        # Apply attack if there are compromised nodes in this neighborhood
        if attacker and any(j in attacker.compromised_nodes for j in neigh):
            compromised_in_neigh = [j for j in neigh if j in attacker.compromised_nodes]
            honest_in_neigh = [j for j in neigh if j not in attacker.compromised_nodes]

            # Get honest neighbor states (partial knowledge assumption)
            honest_neigh_states = [states[j] for j in honest_in_neigh]

            # Craft malicious models using attack algorithm
            # For Krum, this creates multiple similar malicious models to game the selection
            malicious_states = attacker.craft_malicious_models_decentralized(
                i, neigh, honest_neigh_states, round_num
            )

            # Replace compromised nodes' states with crafted malicious ones
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
            # No attack: use honest states
            neigh_states = original_neigh_states

        # Apply Krum selection within (possibly modified) neighborhood
        if len(neigh_states) > 1:
            try:
                # Calculate neighborhood-specific c parameter
                neighborhood_size = len(neigh_states)
                c = int(neighborhood_size * pct_compromised)
                selected_idx = krum_select(neigh_states, c)
                selected_state = neigh_states[selected_idx]
                new_states.append(selected_state)
            except ValueError:
                # Fallback to averaging if Krum conditions not met
                print(f"Warning: Krum failed for node {i}, falling back to averaging")
                w = [1.0 / len(neigh_states)] * len(neigh_states)
                new_states.append(average_states(neigh_states, w))
        else:
            # Single model in neighborhood
            new_states.append(neigh_states[0])

    # Update all models with Krum-selected states
    for model, st in zip(models, new_states):
        set_state(model, st)


# ---------------------------- Original Aggregation Functions (No Attack) ---------------------------- #

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
                # Calculate neighborhood-specific c parameter
                neighborhood_size = len(neigh_states)
                c = int(neighborhood_size * pct_compromised)
                selected_idx = krum_select(neigh_states, c)
                selected_state = neigh_states[selected_idx]
                new_states.append(selected_state)
            except ValueError:
                # Fallback to averaging
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

    # Load LEAF dataset with appropriate model architecture
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

    # Create client partitions using LEAF's natural user groupings
    train_partitions, test_partitions = create_leaf_client_partitions(train_ds, test_ds, args.num_nodes, seed=args.seed)
    parts = [Subset(train_ds, indices) for indices in train_partitions]
    test_parts = [Subset(test_ds, indices) for indices in test_partitions]

    # Dataloaders configuration - optimized for different hardware
    num_workers = 4  # Good for M3 Pro and similar multicore systems
    pin_memory = dev.type != "cpu"  # Use pin_memory for GPU/MPS

    use_sampling = args.max_samples is not None
    if use_sampling:
        print(f"Will sample {args.max_samples} samples per client per epoch (resampled each round)")

    # Create client-specific test loaders (following LEAF methodology)
    test_loaders = [DataLoader(tp, batch_size=512, shuffle=False,
                               num_workers=0, pin_memory=False) for tp in test_parts]

    # Create graph topology for decentralized communication
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
        print(f"Attack verification: {len(attacker.compromised_nodes)} compromised nodes out of {args.num_nodes}")

    # Initialize node models using LEAF architectures
    models = []
    for i in range(args.num_nodes):
        # Set deterministic seed for each model initialization
        torch.manual_seed(args.seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + i)

        if args.dataset.lower() == "femnist":
            model = LEAFFEMNISTModel(num_classes=num_classes).to(dev)
        elif args.dataset.lower() == "celeba":
            model = LEAFCelebAModel(num_classes=num_classes, image_size=image_size).to(dev)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        # Compile model for faster execution (PyTorch 2.0+) if available
        if dev.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass  # Fallback if torch.compile not available
        models.append(model)

    # Evaluate initial performance on client-specific test sets
    with torch.no_grad():
        base_accs = []
        for i, m in enumerate(models):
            acc, _, _, _ = evaluate(m, test_loaders[i], dev)
            base_accs.append(acc)
        print(f"Initial test acc across nodes: mean={np.mean(base_accs):.4f} ± {np.std(base_accs):.4f}")

    # Main training loop
    for r in range(1, args.rounds + 1):
        # Create data loaders with round-specific seeds for reproducible but varying data
        if use_sampling:
            # Sample subset of data each round
            loaders = []
            for i, p in enumerate(parts):
                num_samples = min(args.max_samples, len(p))
                round_seed = args.seed + r * 1000 + i  # Unique seed per round and client
                sampler = RandomSampler(p, replacement=False, num_samples=num_samples,
                                        generator=torch.Generator().manual_seed(round_seed))
                loader = DataLoader(p, batch_size=args.batch_size, sampler=sampler,
                                    num_workers=num_workers, pin_memory=pin_memory)
                loaders.append(loader)
        else:
            # Use full data with shuffling
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

        # Local training phase: each node trains on its local data
        for i, (m, ld) in enumerate(zip(models, loaders)):
            local_train(m, ld, epochs=args.local_epochs, lr=args.lr, device_=dev)

        # Communication/aggregation phase: choose algorithm based on args
        if args.agg == "d-fedadj":
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
        else:
            raise ValueError("agg must be 'd-fedadj', 'gossip', or 'krum'")

        # Evaluation phase: each client tests on their own user's test data
        accs = []
        losses = []
        correct_totals = []
        for i, m in enumerate(models):
            acc, loss, correct, total = evaluate(m, test_loaders[i], dev)
            accs.append(acc)
            losses.append(loss)
            correct_totals.append((correct, total))

        # Print detailed round statistics
        print(f"Round {r:03d}: test acc mean={np.mean(accs):.4f} ± {np.std(accs):.4f} | "
              f"min={np.min(accs):.4f} max={np.max(accs):.4f}")
        print(f"         : test loss mean={np.mean(losses):.4f} ± {np.std(losses):.4f}")
        
        # Individual accuracies
        acc_strs = [f"{acc:.6f}" for acc in accs]
        print(f"         : individual accs = {acc_strs}")
        
        # Correct/total counts
        print(f"         : correct/total = {correct_totals}")
        
        # Show compromised vs honest node performance if under attack
        if attacker and args.verbose:
            compromised_accs = [accs[i] for i in attacker.compromised_nodes]
            honest_accs = [accs[i] for i in range(args.num_nodes) if i not in attacker.compromised_nodes]
            print(f"         : compromised nodes {attacker.compromised_nodes}: mean acc={np.mean(compromised_accs):.4f}")
            print(f"         : honest nodes: mean acc={np.mean(honest_accs):.4f}")

    # Final evaluation and summary
    accs = []
    for i, m in enumerate(models):
        acc, _, _, _ = evaluate(m, test_loaders[i], dev)
        accs.append(acc)

    print("\n=== FINAL RESULTS ===")
    print(f"Dataset: {args.dataset}, Nodes: {args.num_nodes}, Graph: {args.graph}, Agg: {args.agg}")
    if attacker:
        print(f"Attack: {args.attack_type}, {args.attack_percentage*100:.1f}% compromised, lambda={args.attack_lambda}")
    else:
        print("No attack (clean run)")
    print(f"Test accuracy: mean={np.mean(accs):.4f} ± {np.std(accs):.4f}")


# ---------------------------- Command Line Interface ---------------------------- #

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Decentralized Learning Simulator with Attack Capability")

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
                   help="Max samples per client per epoch (for large datasets like CelebA)")

    # Aggregation algorithm parameters
    p.add_argument("--agg", type=str, choices=["d-fedadj", "gossip", "krum"], default="d-fedadj",
                   help="Aggregation algorithm: d-fedadj=FedAvg-style, gossip=random pairs, krum=Byzantine-robust")
    p.add_argument("--gossip-steps", type=int, default=10,
                   help="Number of random edge gossips per round (for gossip aggregation)")
    p.add_argument("--pct-compromised", type=float, default=0.0,
                   help="Max percentage of compromised models per neighborhood for Krum (c = pct * neighborhood_size)")

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
                   help="Attack type: directed_deviation (from paper) or random baseline")
    p.add_argument("--attack-lambda", type=float, default=1.0,
                   help="Lambda parameter controlling attack strength")
    
    # Debug/verbose
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose output for debugging")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sim(args)