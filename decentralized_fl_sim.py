#!/usr/bin/env python3
"""
Decentralized Learning Simulator with BALANCE

Supports aggregation strategies over a peer graph:
  1) Decentralized FedAvg (synchronous neighbor averaging per round)
  2) Gossip averaging (asynchronous-style, simulated with K random edge gossips per round)
  3) Decentralized Krum (Byzantine-robust aggregation, selects most similar model)
  4) BALANCE (Byzantine-robust averaging through local similarity)

Example usage:
  # Clean run with BALANCE
  python decentralized_fl_sim.py \
      --dataset femnist --num-nodes 8 --rounds 20 --local-epochs 1 \
      --agg balance --graph ring --lr 0.01

  # BALANCE under attack
  python decentralized_fl_sim.py \
      --dataset femnist --num-nodes 10 --rounds 30 --local-epochs 1 \
      --agg balance --graph erdos --p 0.3 \
      --attack-percentage 0.3 --attack-type directed_deviation
"""
from __future__ import annotations

import argparse
import random
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


# ---------------------------- BALANCE Aggregation Function ---------------------------- #

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


# ---------------------------- Original Aggregation Functions ---------------------------- #

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

    # Initialize BALANCE monitors
    balance_monitors = {}
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
            decentralized_fedavg_step(models, graph, r, attacker)
        elif args.agg == "gossip":
            gossip_round(models, graph, steps=args.gossip_steps,
                         round_num=r, seed=args.seed, attacker=attacker)
        elif args.agg == "krum":
            decentralized_krum_step(models, graph, args.pct_compromised, r, attacker)
        elif args.agg == "balance":
            balance_aggregation_step(models, graph, balance_monitors, r, attacker)
        else:
            raise ValueError("agg must be 'd-fedavg', 'gossip', 'krum', or 'balance'")

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

            # Show BALANCE information
            if args.agg == "balance" and balance_monitors:
                balance_stats = []
                for node_id, monitor in balance_monitors.items():
                    stats = monitor.get_statistics()
                    balance_stats.append(f"Node {node_id}: acc_rate={stats['mean_acceptance_rate']:.3f}")
                print(f"         : balance stats = {balance_stats[:5]}...")

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

    # Print BALANCE summary
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

        # Algorithm insights
        print(f"BALANCE insights:")
        print(f"  - Uses uniform L2 distance for similarity measurement")
        print(f"  - Threshold decay: gamma={args.balance_gamma} * exp(-{args.balance_kappa} * t)")

        # Performance analysis
        if attacker and honest_accs and compromised_accs:
            avg_acceptance = np.mean(all_acceptance_rates)
            if avg_acceptance < 0.3:
                print(f"WARNING: Very low acceptance rate ({avg_acceptance:.3f}) may indicate aggressive filtering")
            elif avg_acceptance > 0.9:
                print(f"NOTE: High acceptance rate ({avg_acceptance:.3f}) suggests either weak attacks or ineffective filtering")

            attack_mitigation = max(0, np.mean(honest_accs) - np.mean(accs))
            if attack_mitigation > 0.05:
                print(f"ANALYSIS: Algorithm successfully mitigated attack impact by {attack_mitigation:.4f} accuracy")
            else:
                print(f"ANALYSIS: Limited attack mitigation - consider tuning parameters or stronger defenses")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Decentralized Learning Simulator with BALANCE")

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
                   choices=["d-fedavg", "gossip", "krum", "balance"],
                   default="d-fedavg",
                   help="Aggregation algorithm")
    p.add_argument("--gossip-steps", type=int, default=10,
                   help="Number of random edge gossips per round (for gossip aggregation)")
    p.add_argument("--pct-compromised", type=float, default=0.0,
                   help="Max percentage of compromised models per neighborhood for Krum")

    # BALANCE algorithm parameters
    p.add_argument("--balance-gamma", type=float, default=2.0,
                   help="BALANCE similarity threshold multiplier")
    p.add_argument("--balance-kappa", type=float, default=1.0,
                   help="BALANCE threshold decay rate")
    p.add_argument("--balance-alpha", type=float, default=0.5,
                   help="BALANCE weight for own vs neighbor updates")

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
