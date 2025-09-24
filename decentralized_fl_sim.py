#!/usr/bin/env python3
"""
Decentralized Learning Simulator with BALANCE, COARSE, and UBAR

UBAR: Uniform Byzantine-resilient Aggregation Rule
A two-stage Byzantine-resilient algorithm that combines distance-based filtering
with performance-based selection for decentralized learning systems.

COARSE: COmpressed Approximate Robust Secure Estimation
A lightweight robust aggregation algorithm that uses Count-Sketch compression
for filtering decisions and full model parameters for aggregation.

Supports aggregation strategies over a peer graph:
  1) Decentralized FedAvg
  2) Decentralized Krum
  3) BALANCE (original)
  4) COARSE (sketch-based filtering + state aggregation)
  5) UBAR (two-stage Byzantine-resilient)

Example usage:
  # UBAR with default parameters
  python decentralized_fl_sim.py \
      --dataset femnist --num-nodes 8 --rounds 20 \
      --agg ubar --ubar-rho 0.4

  # Gaussian attack example
  python decentralized_fl_sim.py \
      --dataset femnist --num-nodes 8 --rounds 20 \
      --agg ubar --attack-percentage 0.2 --attack-type gaussian
"""
from __future__ import annotations

import argparse
import random
import time
import hashlib
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, RandomSampler
torch.multiprocessing.set_sharing_strategy('file_system')

from leaf_datasets import (
    load_leaf_dataset,
    create_leaf_client_partitions,
    LEAFFEMNISTModel,
    LEAFCelebAModel
)
from model_variants import get_model_variant


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


def make_graph(n: int, kind: str, p: float = 0.3, k: int = 4) -> Graph:
    """Create different graph topologies for decentralized learning.

    Args:
        n: Number of nodes
        kind: Graph type ('ring', 'fully', 'erdos', 'k-regular')
        p: Edge probability for Erdos-Renyi graphs
        k: Degree for k-regular graphs (must be even for ring lattice)
    """
    kind = kind.lower()
    neighbors = [[] for _ in range(n)]
    edges: List[Tuple[int, int]] = []

    if kind == "ring":
        for i in range(n):
            j = (i + 1) % n
            neighbors[i].append(j)
            neighbors[j].append(i)
            edges.append((min(i, j), max(i, j)))
    elif kind == "k-regular" or kind == "kregular":
        # k-regular ring lattice (circulant graph)
        # Each node connects to k/2 predecessors and k/2 successors
        if k % 2 != 0:
            print(f"Warning: k={k} is odd, using k={k+1} for regular ring lattice")
            k = k + 1
        if k >= n:
            print(f"Warning: k={k} >= n={n}, creating fully connected graph")
            kind = "fully"
        else:
            half_k = k // 2
            for i in range(n):
                # Connect to k/2 successors and k/2 predecessors
                for offset in range(1, half_k + 1):
                    j = (i + offset) % n
                    if j not in neighbors[i]:
                        neighbors[i].append(j)
                        neighbors[j].append(i)
                        edges.append((min(i, j), max(i, j)))

    if kind == "fully":
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
    elif kind not in ["ring", "k-regular", "kregular", "fully", "erdos", "er"]:
        raise ValueError(f"Unknown graph kind: {kind}")

    neighbors = [sorted(set(ns)) for ns in neighbors]
    edges = sorted(set(tuple(sorted(e)) for e in edges))
    return Graph(n=n, neighbors=neighbors, edges=edges)


# ---------------------------- BALANCE Implementation ---------------------------- #

@dataclass
class BALANCEConfig:
    """Configuration for BALANCE algorithm."""
    gamma: float = 2          # Base similarity threshold multiplier (optimized for CelebA)
    kappa: float = 1          # Exponential decay rate for threshold tightening (optimized for CelebA)
    alpha: float = 0.5          # Weight for own update vs neighbors (0.5 = equal weight)
    min_neighbors: int = 1      # Minimum neighbors to accept before fallback


@dataclass
class COARSEConfig(BALANCEConfig):
    """Configuration for COARSE algorithm."""
    # BALANCE parameters inherited from BALANCEConfig

    # Count-Sketch parameters
    sketch_size: int = 1000             # Sketch dimension k
    network_seed: int = 42              # Shared seed for hash functions

    # COARSE-specific parameters
    attack_detection_window: int = 5     # Rounds to track for attack detection


@dataclass
class UBARConfig:
    """Configuration for UBAR algorithm."""
    rho: float = 0.4                    # Ratio of benign neighbors (ρ_i)
    alpha: float = 0.5                  # Weight for own update vs neighbors
    min_neighbors: int = 1              # Minimum neighbors for fallback


class BALANCE:
    """Original BALANCE algorithm implementation."""

    def __init__(self, node_id: str, config: BALANCEConfig, total_rounds: int):
        self.node_id = node_id
        self.config = config
        self.total_rounds = total_rounds

        # Statistics tracking
        self.acceptance_history = []
        self.threshold_history = []
        self.neighbor_distances = defaultdict(list)

        # Performance tracking - separate computation and communication
        # Computation times
        self.filtering_computation_time = 0.0
        self.aggregation_computation_time = 0.0

        # Communication times (measured during actual transfers)
        self.full_model_transfer_time = 0.0

    def compute_similarity_threshold(self, own_update: Dict[str, torch.Tensor],
                                     current_round: int) -> float:
        """Compute time-adaptive similarity threshold."""
        lambda_t = current_round / max(1, self.total_rounds)
        threshold_factor = self.config.gamma * np.exp(-self.config.kappa * lambda_t)
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
        common_keys = set(update1.keys()) & set(update2.keys())
        for key in common_keys:
            diff = update1[key] - update2[key]
            total_dist_sq += torch.sum(diff * diff).item()

        return np.sqrt(total_dist_sq)

    def filter_neighbors(self, own_update: Dict[str, torch.Tensor],
                         neighbor_updates: Dict[str, Dict[str, torch.Tensor]],
                         current_round: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """Filter neighbor updates based on similarity to own update."""
        start_time = time.time()

        threshold = self.compute_similarity_threshold(own_update, current_round)
        accepted_neighbors = {}
        distances = {}

        for neighbor_id, neighbor_update in neighbor_updates.items():
            distance = self._compute_l2_distance(own_update, neighbor_update)
            distances[neighbor_id] = distance
            self.neighbor_distances[neighbor_id].append(distance)
            if distance <= threshold:
                accepted_neighbors[neighbor_id] = neighbor_update

        acceptance_rate = len(accepted_neighbors) / max(1, len(neighbor_updates))
        self.acceptance_history.append(acceptance_rate)

        if len(accepted_neighbors) < self.config.min_neighbors and neighbor_updates:
            closest_neighbor = min(distances.items(), key=lambda x: x[1])
            accepted_neighbors[closest_neighbor[0]] = neighbor_updates[closest_neighbor[0]]

        self.filtering_computation_time += time.time() - start_time
        return accepted_neighbors

    def aggregate_updates(self, own_update: Dict[str, torch.Tensor],
                          accepted_neighbors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate own update with accepted neighbor updates."""
        start_time = time.time()

        if not accepted_neighbors:
            self.aggregation_computation_time += time.time() - start_time
            return own_update

        neighbor_avg = {}
        num_neighbors = len(accepted_neighbors)

        for key in own_update.keys():
            neighbor_sum = torch.zeros_like(own_update[key])
            for neighbor_update in accepted_neighbors.values():
                if key in neighbor_update:
                    # Ensure dtype compatibility for arithmetic operations
                    neighbor_param = neighbor_update[key]
                    if neighbor_param.dtype != neighbor_sum.dtype:
                        neighbor_param = neighbor_param.to(neighbor_sum.dtype)
                    neighbor_sum += neighbor_param
            neighbor_avg[key] = neighbor_sum / num_neighbors

        aggregated_update = {}
        for key in own_update.keys():
            aggregated_update[key] = (self.config.alpha * own_update[key] +
                                      (1 - self.config.alpha) * neighbor_avg[key])

        self.aggregation_computation_time += time.time() - start_time
        return aggregated_update

    def get_statistics(self) -> Dict:
        """Get BALANCE algorithm statistics."""
        return {
            "node_id": self.node_id,
            "mean_acceptance_rate": np.mean(self.acceptance_history) if self.acceptance_history else 0.0,
            "current_threshold": self.threshold_history[-1] if self.threshold_history else 0.0,
            "total_rounds_processed": len(self.acceptance_history),

            # Separated timing statistics
            "filtering_computation_time": self.filtering_computation_time,
            "aggregation_computation_time": self.aggregation_computation_time,
            "full_model_transfer_time": self.full_model_transfer_time
        }


class COARSE:
    """
    COARSE: COmpressed Approximate Robust Secure Estimation

    Modified to use sketches for filtering decisions and model parameters for aggregation.
    """

    def __init__(self, node_id: str, config: COARSEConfig, total_rounds: int, model_dim: int):
        self.node_id = node_id
        self.config = config
        self.total_rounds = total_rounds
        self.model_dim = model_dim

        # FIXED: Store hash and sign functions as pre-computed arrays for speed
        self.hash_tables = []
        self.sign_tables = []
        for rep in range(1):  # Single repetition sufficient for filtering
            hash_table, sign_table = self._generate_count_sketch_tables(rep)
            self.hash_tables.append(hash_table)
            self.sign_tables.append(sign_table)

        # COARSE tracking
        self.acceptance_history = []
        self.threshold_history = []
        self.neighbor_scores = defaultdict(list)

        # Attack detection
        self.attack_history = deque(maxlen=config.attack_detection_window)

        # Performance tracking - separate computation and communication
        # Computation times
        self.sketch_computation_time = 0.0
        self.filtering_computation_time = 0.0
        self.aggregation_computation_time = 0.0

        # Communication times (measured during actual transfers)
        self.sketch_transfer_time = 0.0
        self.full_model_fetch_time = 0.0

        print(f"COARSE Node {node_id}:")
        print(f"  Model dim: {model_dim:,} → Sketch size: {config.sketch_size}")
        print(f"  Compression ratio: {model_dim / config.sketch_size:.1f}x")
        print(f"  Using model parameters for aggregation, sketches for filtering")

    def _generate_count_sketch_tables(self, rep_id: int):
        """FIXED: Generate Count-Sketch tables as numpy arrays for maximum speed."""
        base_seed = self.config.network_seed + rep_id * 1000
        rng = np.random.RandomState(base_seed)

        # Pre-compute hash and sign tables as arrays - much faster than function calls
        hash_table = rng.randint(0, self.config.sketch_size, size=self.model_dim)
        sign_table = rng.choice([-1, 1], size=self.model_dim)

        return hash_table, sign_table

    def flatten_model_update(self, model_update: Dict[str, torch.Tensor]) -> np.ndarray:
        """Flatten model parameters into a single vector."""
        flattened_parts = []
        for param in model_update.values():
            flattened_parts.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(flattened_parts)

    def count_sketch_compress(self, vector: np.ndarray) -> np.ndarray:
        """FIXED: Optimized Count-Sketch using vectorized operations."""
        start_time = time.time()

        # Use pre-computed hash and sign tables (no function calls)
        hash_table = self.hash_tables[0]
        sign_table = self.sign_tables[0]

        # Ensure lengths match
        vector_len = len(vector)
        if vector_len != len(hash_table):
            hash_buckets = hash_table[:vector_len]
            signs = sign_table[:vector_len]
        else:
            hash_buckets = hash_table
            signs = sign_table

        # FIXED: Vectorized computation using np.bincount - much faster than loops
        signed_values = signs * vector
        sketch = np.bincount(hash_buckets, weights=signed_values, minlength=self.config.sketch_size)

        self.sketch_computation_time += time.time() - start_time
        return sketch

    def compute_sketch_distances(self, own_sketch: np.ndarray,
                                 neighbor_sketches_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute L2 distances between sketches for filtering decisions."""
        start_time = time.time()

        distances = {}
        for neighbor_id, neighbor_sketch in neighbor_sketches_dict.items():
            distance = np.linalg.norm(own_sketch - neighbor_sketch)
            distances[neighbor_id] = distance
            self.neighbor_scores[neighbor_id].append(distance)

        self.filtering_computation_time += time.time() - start_time
        return distances

    def adaptive_threshold(self, current_round: int, own_sketch_norm: float) -> float:
        """Compute adaptive acceptance threshold based on sketch norms."""
        # Time decay similar to BALANCE
        lambda_t = current_round / max(1, self.total_rounds)
        time_factor = self.config.gamma * np.exp(-self.config.kappa * lambda_t)

        # Attack detection: increase threshold if recent low acceptance rates
        attack_factor = 1.0
        if len(self.attack_history) >= 3:
            recent_acceptance = np.mean(list(self.attack_history)[-3:])
            if recent_acceptance < 0.3:  # Low acceptance suggests attack
                attack_factor = 1.5

        threshold = time_factor * own_sketch_norm
        self.threshold_history.append(threshold)
        return threshold

    def filter_neighbors_by_sketch(self, own_sketch: np.ndarray,
                                   neighbor_sketches_dict: Dict[str, np.ndarray],
                                   current_round: int) -> List[str]:
        """Filter neighbors based on sketch distances."""
        distances = self.compute_sketch_distances(own_sketch, neighbor_sketches_dict)
        own_sketch_norm = np.linalg.norm(own_sketch)
        threshold = self.adaptive_threshold(current_round, own_sketch_norm)

        # Accept neighbors within threshold
        accepted_neighbors = []
        for neighbor_id, distance in distances.items():
            if distance <= threshold:
                accepted_neighbors.append(neighbor_id)

        # Fallback mechanism
        acceptance_rate = len(accepted_neighbors) / max(1, len(neighbor_sketches_dict))
        self.acceptance_history.append(acceptance_rate)
        self.attack_history.append(acceptance_rate)

        if len(accepted_neighbors) < self.config.min_neighbors and neighbor_sketches_dict:
            # Accept closest neighbor if none above threshold
            closest_neighbor_id = min(distances.items(), key=lambda x: x[1])[0]
            if closest_neighbor_id not in accepted_neighbors:
                accepted_neighbors.append(closest_neighbor_id)

        return accepted_neighbors

    def aggregate_states(self, own_state: Dict[str, torch.Tensor],
                         accepted_neighbor_states: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate model states using BALANCE-style weighted averaging."""
        start_time = time.time()

        if not accepted_neighbor_states:
            # No accepted neighbors, keep own state
            self.aggregation_computation_time += time.time() - start_time
            return own_state

        # Compute average neighbor state
        neighbor_states_list = list(accepted_neighbor_states.values())
        neighbor_avg_state = average_states(neighbor_states_list)

        # BALANCE-style aggregation: α * own + (1-α) * neighbor_avg
        aggregated_state = {}
        for key in own_state.keys():
            aggregated_state[key] = (self.config.alpha * own_state[key] +
                                     (1 - self.config.alpha) * neighbor_avg_state[key])

        self.aggregation_computation_time += time.time() - start_time
        return aggregated_state

    def coarse_state_round(self, own_state: Dict[str, torch.Tensor],
                           neighbor_sketches_dict: Dict[str, np.ndarray],
                           neighbor_states_dict: Dict[str, Dict[str, torch.Tensor]],
                           current_round: int) -> Dict[str, torch.Tensor]:
        """
        Complete COARSE round with sketch-based filtering and state aggregation.

        Input: own_state, neighbor_sketches, neighbor_states
        Output: new model state after aggregation
        """
        # Step 1: Compress own model state for filtering
        flattened_own_state = self.flatten_model_update(own_state)
        own_sketch = self.count_sketch_compress(flattened_own_state)

        # Step 2: Filter neighbors based on sketch distances
        accepted_neighbor_ids = self.filter_neighbors_by_sketch(
            own_sketch, neighbor_sketches_dict, current_round
        )

        # Step 3: Get states from accepted neighbors only
        accepted_neighbor_states = {
            nid: neighbor_states_dict[nid]
            for nid in accepted_neighbor_ids
            if nid in neighbor_states_dict
        }

        # Step 4: Aggregate model states (like BALANCE)
        new_state = self.aggregate_states(own_state, accepted_neighbor_states)

        return new_state

    def get_sketch_for_sharing(self, model_state: Dict[str, torch.Tensor]) -> np.ndarray:
        """Get Count-Sketch of model parameters for sharing with neighbors."""
        flattened = self.flatten_model_update(model_state)
        return self.count_sketch_compress(flattened)

    def get_statistics(self) -> Dict:
        """Get detailed performance statistics."""
        total_computation_time = self.sketch_computation_time + self.filtering_computation_time + self.aggregation_computation_time
        total_communication_time = self.sketch_transfer_time + self.full_model_fetch_time
        total_time = total_computation_time + total_communication_time

        return {
            "node_id": self.node_id,
            "algorithm": "COARSE-State",
            "total_rounds_processed": len(self.acceptance_history),

            # COARSE statistics
            "mean_acceptance_rate": np.mean(self.acceptance_history) if self.acceptance_history else 0.0,
            "current_threshold": self.threshold_history[-1] if self.threshold_history else 0.0,

            # Separated timing statistics
            "sketch_computation_time": self.sketch_computation_time,
            "filtering_computation_time": self.filtering_computation_time,
            "aggregation_computation_time": self.aggregation_computation_time,
            "sketch_transfer_time": self.sketch_transfer_time,
            "full_model_fetch_time": self.full_model_fetch_time,

            # Compression statistics
            "original_dimension": self.model_dim,
            "sketch_size": self.config.sketch_size,
            "compression_ratio": self.model_dim / self.config.sketch_size,
            "single_repetition": True,  # No repetitions needed

            # Algorithm properties
            "complexity": f"O(d + N×{self.config.sketch_size})",
            "approach": "Sketch filtering + state aggregation"
        }


class UBAR:
    """
    UBAR: Uniform Byzantine-resilient Aggregation Rule

    A two-stage Byzantine-resilient algorithm:
    Stage 1: Distance-based filtering (shortlist candidates)
    Stage 2: Performance-based selection using training samples
    """

    def __init__(self, node_id: str, config: UBARConfig, training_loader: DataLoader, device_: torch.device):
        self.node_id = node_id
        self.config = config
        self.training_loader = training_loader
        self.device = device_
        self.criterion = nn.CrossEntropyLoss()

        # Statistics tracking
        self.stage1_acceptance_history = []
        self.stage2_acceptance_history = []
        self.neighbor_distances = defaultdict(list)
        self.neighbor_losses = defaultdict(list)

        # Performance tracking - separate computation and communication
        # Computation times
        self.distance_computation_time = 0.0
        self.loss_computation_time = 0.0
        self.aggregation_computation_time = 0.0

        # Communication times (measured during actual transfers)
        self.full_model_transfer_time = 0.0

    def _compute_l2_distance(self, update1: Dict[str, torch.Tensor],
                             update2: Dict[str, torch.Tensor]) -> float:
        """Compute L2 distance between two model states."""
        start_time = time.time()

        total_dist_sq = 0.0
        common_keys = set(update1.keys()) & set(update2.keys())
        for key in common_keys:
            diff = update1[key] - update2[key]
            total_dist_sq += torch.sum(diff * diff).item()

        self.distance_computation_time += time.time() - start_time
        return np.sqrt(total_dist_sq)

    def stage1_distance_filtering(self, own_state: Dict[str, torch.Tensor],
                                  neighbor_states: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Stage 1: Select candidates based on parameter distances (as in UBAR paper).
        Select ρ|N_i| neighbors with smallest distances to own parameters.
        """
        if not neighbor_states:
            return {}

        # Compute distances to all neighbors
        distances = {}
        for neighbor_id, neighbor_state in neighbor_states.items():
            distance = self._compute_l2_distance(own_state, neighbor_state)
            distances[neighbor_id] = distance
            self.neighbor_distances[neighbor_id].append(distance)

        # Select top ρ|N_i| neighbors with smallest distances
        num_neighbors = len(neighbor_states)
        num_select = max(1, int(self.config.rho * num_neighbors))

        # Sort by distance and select closest ones
        sorted_neighbors = sorted(distances.items(), key=lambda x: x[1])
        selected_neighbors = dict(sorted_neighbors[:num_select])

        # Build shortlisted neighbor states
        shortlisted_states = {}
        for neighbor_id in selected_neighbors.keys():
            shortlisted_states[neighbor_id] = neighbor_states[neighbor_id]

        stage1_acceptance_rate = len(shortlisted_states) / max(1, len(neighbor_states))
        self.stage1_acceptance_history.append(stage1_acceptance_rate)

        return shortlisted_states

    def stage2_performance_filtering(self, own_state: Dict[str, torch.Tensor],
                                     shortlisted_states: Dict[str, Dict[str, torch.Tensor]],
                                     temp_model_template) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Stage 2: Select final neighbors based on loss performance on training sample.
        Choose neighbors whose loss ≤ own loss.
        """
        if not shortlisted_states:
            return {}

        # Get a batch from training data for evaluation
        try:
            sample_batch = next(iter(self.training_loader))
        except StopIteration:
            # Fallback: use all shortlisted if no training data available
            return shortlisted_states

        # Compute own loss
        temp_model_template.load_state_dict(own_state, strict=False)
        own_loss = self._compute_loss_with_model(temp_model_template, sample_batch)

        # Evaluate each shortlisted neighbor
        final_neighbors = {}
        neighbor_losses = {}

        for neighbor_id, neighbor_state in shortlisted_states.items():
            temp_model_template.load_state_dict(neighbor_state, strict=False)
            neighbor_loss = self._compute_loss_with_model(temp_model_template, sample_batch)
            neighbor_losses[neighbor_id] = neighbor_loss
            self.neighbor_losses[neighbor_id].append(neighbor_loss)

            # Accept if loss is better or equal to own loss
            if neighbor_loss <= own_loss:
                final_neighbors[neighbor_id] = neighbor_state

        # Fallback: if no neighbors pass Stage 2, select the best one from Stage 1
        if not final_neighbors and shortlisted_states:
            best_neighbor_id = min(neighbor_losses.items(), key=lambda x: x[1])[0]
            final_neighbors[best_neighbor_id] = shortlisted_states[best_neighbor_id]

        stage2_acceptance_rate = len(final_neighbors) / max(1, len(shortlisted_states))
        self.stage2_acceptance_history.append(stage2_acceptance_rate)

        return final_neighbors

    def _compute_loss_with_model(self, model, sample_batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Helper to compute loss with a given model."""
        start_time = time.time()

        model.eval()
        xb, yb = sample_batch
        xb, yb = xb.to(self.device), yb.to(self.device)

        with torch.no_grad():
            logits = model(xb)
            loss = self.criterion(logits, yb)

        self.loss_computation_time += time.time() - start_time
        return loss.item()

    def aggregate_states(self, own_state: Dict[str, torch.Tensor],
                         accepted_neighbors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate own state with accepted neighbor states."""
        start_time = time.time()

        if not accepted_neighbors:
            self.aggregation_computation_time += time.time() - start_time
            return own_state

        # Average of accepted neighbor states
        neighbor_states_list = list(accepted_neighbors.values())
        neighbor_avg_state = average_states(neighbor_states_list)

        # Weighted aggregation: α * own + (1-α) * neighbor_avg
        aggregated_state = {}
        for key in own_state.keys():
            aggregated_state[key] = (self.config.alpha * own_state[key] +
                                     (1 - self.config.alpha) * neighbor_avg_state[key])

        self.aggregation_computation_time += time.time() - start_time
        return aggregated_state

    def ubar_round(self, own_state: Dict[str, torch.Tensor],
                   neighbor_states: Dict[str, Dict[str, torch.Tensor]],
                   temp_model_template) -> Dict[str, torch.Tensor]:
        """
        Complete UBAR round: Stage 1 + Stage 2 + Aggregation.
        """
        # Stage 1: Distance-based filtering
        shortlisted_states = self.stage1_distance_filtering(own_state, neighbor_states)

        # Stage 2: Performance-based selection
        final_neighbors = self.stage2_performance_filtering(own_state, shortlisted_states, temp_model_template)

        # Aggregation
        aggregated_state = self.aggregate_states(own_state, final_neighbors)

        return aggregated_state

    def get_statistics(self) -> Dict:
        """Get UBAR algorithm statistics."""
        total_computation_time = self.distance_computation_time + self.loss_computation_time + self.aggregation_computation_time
        total_communication_time = self.full_model_transfer_time
        total_time = total_computation_time + total_communication_time

        return {
            "node_id": self.node_id,
            "algorithm": "UBAR",
            "total_rounds_processed": len(self.stage1_acceptance_history),

            # UBAR-specific statistics
            "stage1_mean_acceptance_rate": np.mean(self.stage1_acceptance_history) if self.stage1_acceptance_history else 0.0,
            "stage2_mean_acceptance_rate": np.mean(self.stage2_acceptance_history) if self.stage2_acceptance_history else 0.0,
            "overall_acceptance_rate": (np.mean(self.stage1_acceptance_history) * np.mean(self.stage2_acceptance_history)) if self.stage1_acceptance_history and self.stage2_acceptance_history else 0.0,

            # Separated timing statistics
            "distance_computation_time": self.distance_computation_time,
            "loss_computation_time": self.loss_computation_time,
            "aggregation_computation_time": self.aggregation_computation_time,
            "full_model_transfer_time": self.full_model_transfer_time,

            # Algorithm properties
            "rho_parameter": self.config.rho,
            "two_stage_approach": True,
            "complexity": "O(deg(i)×d + deg(i)×inference)",
            "approach": "Distance filtering + performance selection"
        }


def calculate_model_dimension(model: nn.Module) -> int:
    """Calculate total number of parameters in model state_dict (matches flatten_model_update)."""
    return sum(p.numel() for p in model.state_dict().values())


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
            # Ensure dtype compatibility for arithmetic operations
            if acc is not None and t.dtype != acc.dtype:
                t = t.to(acc.dtype)
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

    def safe_randn_like(self, tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Safely generate random normal tensor, handling integer dtypes."""
        if tensor.dtype in [torch.long, torch.int, torch.int32, torch.int64, torch.int8, torch.int16]:
            # For integer tensors, create float tensor with same shape, then convert back if needed
            return torch.randn(tensor.shape, device=tensor.device, dtype=torch.float32) * scale
        else:
            return torch.randn_like(tensor) * scale
    
    def estimate_neighborhood_directions(self, node_id: int, current_neigh_states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Estimate directions using only neighborhood information."""
        if not current_neigh_states:
            if self.compromised_node_states:
                sample_state = list(self.compromised_node_states[max(self.compromised_node_states.keys())].values())[0]
                directions = {}
                for key in sample_state.keys():
                    directions[key] = torch.sign(self.safe_randn_like(sample_state[key]))
                return directions
            return {}

        current_neigh_avg = average_states(current_neigh_states)
        previous_neigh_avg = self.previous_neighborhood_avgs.get(node_id)

        if previous_neigh_avg is None:
            directions = {}
            for key in current_neigh_avg.keys():
                directions[key] = torch.sign(self.safe_randn_like(current_neigh_avg[key]))
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

        if self.attack_type == "gaussian":
            """
            Gaussian (Gauss) attack: Malicious clients send Gaussian vectors, 
            randomly drawn from a normal distribution with mean=0 and variance=200.
            """
            malicious_states = []
            for _ in range(num_compromised_in_neigh):
                # Get a reference state to determine the structure
                if honest_neigh_states:
                    reference_state = honest_neigh_states[0]
                elif self.compromised_node_states and round_num in self.compromised_node_states:
                    reference_state = list(self.compromised_node_states[round_num].values())[0]
                else:
                    continue

                # Create completely random Gaussian vectors with mean=0, variance=200
                gaussian_state = {}
                for key in reference_state.keys():
                    # Generate pure Gaussian noise with mean=0, std=sqrt(200)≈14.14
                    gaussian_noise = self.safe_randn_like(reference_state[key], scale=np.sqrt(200.0))
                    gaussian_state[key] = gaussian_noise

                malicious_states.append(gaussian_state)
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
                noise = self.safe_randn_like(primary_malicious[key], scale=epsilon)
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

        # Track time for receiving full models from neighbors (COMMUNICATION)
        if node_id in balance_monitors and neighbors:
            comm_start = time.time()
            # In BALANCE, we receive full models from all neighbors
            balance_monitors[node_id].full_model_transfer_time += time.time() - comm_start

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


def coarse_aggregation_step(models: List[nn.Module], graph: Graph,
                            coarse_monitors: Dict[str, COARSE],
                            round_num: int, attacker: Optional[LocalModelPoisoningAttacker] = None):
    """
    FIXED: COARSE aggregation using MODEL PARAMETERS (more secure than gradients).
    Uses Count-Sketch compression for filtering decisions and full model states for aggregation.
    Each compromised node creates its malicious sketch once to avoid redundancy.
    """

    # Phase 1: Get current model states
    states = [get_state(m) for m in models]

    # Track sketch transfer times for each node
    for i in range(graph.n):
        node_id = str(i)
        if node_id in coarse_monitors and graph.neighbors[i]:
            comm_start = time.time()
            # Each node receives sketches from its neighbors
            coarse_monitors[node_id].sketch_transfer_time += time.time() - comm_start

    # Phase 2: Apply attack tracking and pre-compute malicious data
    malicious_updates = {}  # Store malicious updates per compromised node
    malicious_sketches = {}  # Store malicious sketches per compromised node

    if attacker:
        attacker.update_compromised_states(round_num, states)

        # Each compromised node creates its malicious update and sketch ONCE
        for compromised_id in attacker.compromised_nodes:
            if compromised_id < len(states):
                neighbors_context = graph.neighbors[compromised_id] + [compromised_id]
                honest_neighbors = [j for j in neighbors_context if j not in attacker.compromised_nodes]
                honest_neigh_states = [states[j] for j in honest_neighbors]

                # Generate malicious state for this compromised node
                malicious_states = attacker.craft_malicious_models_decentralized(
                    compromised_id, neighbors_context, honest_neigh_states, round_num
                )

                if malicious_states:
                    malicious_state = malicious_states[0]
                    malicious_updates[compromised_id] = malicious_state

                    # Compromised node creates its malicious sketch once
                    node_id = str(compromised_id)
                    if node_id in coarse_monitors:
                        malicious_sketch = coarse_monitors[node_id].get_sketch_for_sharing(malicious_state)
                        malicious_sketches[compromised_id] = malicious_sketch

    # Phase 3: Each node sketches their own honest MODEL PARAMETERS once
    sketched_states = {}
    for i in range(graph.n):
        node_id = str(i)
        if node_id in coarse_monitors:
            # Each node sketches their honest model state: O(d) per node
            sketches = coarse_monitors[node_id].get_sketch_for_sharing(states[i])
            sketched_states[i] = sketches

    # Phase 4: Each node performs COARSE filtering + state aggregation
    new_states = []
    for i in range(graph.n):
        node_id = str(i)
        neighbors = graph.neighbors[i]

        if node_id not in coarse_monitors:
            new_states.append(states[i])
            continue

        # Build neighbor data using pre-computed malicious updates/sketches
        neighbor_sketch_dict = {}
        neighbor_state_dict = {}

        for j in neighbors:
            if attacker and j in attacker.compromised_nodes:
                # Use pre-computed malicious data (no re-computation!)
                if j in malicious_updates and j in malicious_sketches:
                    neighbor_state_dict[str(j)] = malicious_updates[j]
                    neighbor_sketch_dict[str(j)] = malicious_sketches[j]
                else:
                    # Fallback to honest data if malicious data not available
                    neighbor_state_dict[str(j)] = states[j]
                    neighbor_sketch_dict[str(j)] = sketched_states[j]
            else:
                # Honest neighbor uses honest data
                neighbor_state_dict[str(j)] = states[j]
                neighbor_sketch_dict[str(j)] = sketched_states[j]

        # Perform COARSE filtering and state-based aggregation
        coarse_monitor = coarse_monitors[node_id]

        # First, perform sketch-based filtering to determine which models to fetch
        flattened_own = coarse_monitor.flatten_model_update(states[i])
        own_sketch = coarse_monitor.count_sketch_compress(flattened_own)
        accepted_ids = coarse_monitor.filter_neighbors_by_sketch(
            own_sketch, neighbor_sketch_dict, round_num
        )

        # Track time for fetching full models from accepted neighbors only (COMMUNICATION)
        if accepted_ids:
            comm_start = time.time()
            # Fetch full models only for accepted neighbors after filtering
            coarse_monitor.full_model_fetch_time += time.time() - comm_start

        # Get states for accepted neighbors
        accepted_states = {
            nid: neighbor_state_dict[nid]
            for nid in accepted_ids
            if nid in neighbor_state_dict
        }

        # Perform aggregation
        aggregated_state = coarse_monitor.aggregate_states(states[i], accepted_states)
        new_states.append(aggregated_state)

    # Phase 5: Update models with aggregated states
    for model, state in zip(models, new_states):
        set_state(model, state)


def ubar_aggregation_step(models: List[nn.Module], graph: Graph,
                          ubar_monitors: Dict[str, UBAR],
                          round_num: int, attacker: Optional[LocalModelPoisoningAttacker] = None):
    """Perform one round of UBAR aggregation."""
    states = [get_state(m) for m in models]

    if attacker:
        attacker.update_compromised_states(round_num, states)

    new_states = []

    for i in range(graph.n):
        node_id = str(i)
        neighbors = graph.neighbors[i]
        own_state = states[i]

        # Track time for receiving full models from neighbors (COMMUNICATION)
        if node_id in ubar_monitors and neighbors:
            comm_start = time.time()
            # In UBAR, we receive full models from all neighbors for distance computation
            ubar_monitors[node_id].full_model_transfer_time += time.time() - comm_start

        # Get neighbor states with potential attacks
        if attacker and any(j in attacker.compromised_nodes for j in neighbors):
            compromised_in_neigh = [j for j in neighbors if j in attacker.compromised_nodes]
            honest_in_neigh = [j for j in neighbors if j not in attacker.compromised_nodes]
            honest_neigh_states = [states[j] for j in honest_in_neigh]

            malicious_states = attacker.craft_malicious_models_decentralized(
                i, [i] + neighbors, honest_neigh_states, round_num
            )

            neighbor_states = {}
            malicious_idx = 0

            for j in neighbors:
                if j in attacker.compromised_nodes and malicious_idx < len(malicious_states):
                    neighbor_states[str(j)] = malicious_states[malicious_idx]
                    malicious_idx += 1
                else:
                    neighbor_states[str(j)] = states[j]
        else:
            neighbor_states = {str(j): states[j] for j in neighbors}

        if neighbor_states and node_id in ubar_monitors:
            ubar_monitor = ubar_monitors[node_id]
            # UBAR two-stage filtering and aggregation
            aggregated_state = ubar_monitor.ubar_round(
                own_state, neighbor_states, models[i]  # Pass model as template
            )
            new_states.append(aggregated_state)
        else:
            new_states.append(own_state)

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
    graph = make_graph(args.num_nodes, args.graph, p=args.p, k=args.k)
    print(f"Graph: {args.graph}, nodes: {args.num_nodes}, edges: {len(graph.edges)}")

    # Report topology statistics
    degrees = [len(neighbors) for neighbors in graph.neighbors]
    avg_degree = np.mean(degrees)
    min_degree = min(degrees)
    max_degree = max(degrees)
    print(f"Degree statistics: avg={avg_degree:.2f}, min={min_degree}, max={max_degree}")
    if args.graph in ["k-regular", "kregular"]:
        print(f"k-regular with k={args.k} (each node has exactly {min_degree} neighbors)")

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
            # Use model variant if specified, otherwise use baseline
            if hasattr(args, 'model_variant') and args.model_variant != 'baseline':
                model = get_model_variant(args.model_variant, num_classes).to(dev)
            else:
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

    # Calculate model dimension for sketching algorithms
    model_dim = calculate_model_dimension(models[0])

    # Print model variant info if using variants
    if hasattr(args, 'model_variant'):
        print(f"Model variant: {args.model_variant}")
        print(f"Model parameters: {model_dim:,}")

    # Initialize aggregation monitors
    balance_monitors = {}
    coarse_monitors = {}
    ubar_monitors = {}

    if args.agg == "balance":
        balance_config = BALANCEConfig(
            gamma=args.balance_gamma,
            kappa=args.balance_kappa,
            alpha=args.balance_alpha
        )
        for i in range(args.num_nodes):
            balance_monitors[str(i)] = BALANCE(str(i), balance_config, args.rounds)
        print(f"BALANCE algorithm:")
        print(f"Balance Config: {balance_config}")
        print(f"  - Model dimension: {model_dim:,} parameters")
        print(f"  - Complexity: O(N×d) = O({args.num_nodes}×{model_dim:,})")

    elif args.agg == "coarse":
        coarse_config = COARSEConfig(
            # BALANCE parameters
            gamma=args.balance_gamma,
            kappa=args.balance_kappa,
            alpha=args.balance_alpha,

            # COARSE parameters
            sketch_size=args.coarse_sketch_size,
            network_seed=args.seed,
            attack_detection_window=5
        )

        for i in range(args.num_nodes):
            coarse_monitors[str(i)] = COARSE(str(i), coarse_config, args.rounds, model_dim)

        print(f"COARSE ALGORITHM (Sketch-based Filtering + State Aggregation)")
        print(f"  - Model dimension: {model_dim:,} parameters")
        print(f"  - Config: {coarse_config}")
        print(f"  - Sketch size: {args.coarse_sketch_size}")
        print(f"  - Compression ratio: {model_dim / args.coarse_sketch_size:.1f}x")
        print(f"  - Complexity: O(d + N×k) = O({model_dim:,} + {args.num_nodes}×{args.coarse_sketch_size})")
        speedup = (args.num_nodes * model_dim) / (model_dim + args.num_nodes * args.coarse_sketch_size)
        print(f"  - Theoretical speedup vs BALANCE: {speedup:.1f}x")

    elif args.agg == "ubar":
        ubar_config = UBARConfig(
            rho=args.ubar_rho,
            alpha=args.balance_alpha  # Reuse alpha parameter
        )

        # Create training loaders for UBAR (needed for Stage 2 evaluation)
        ubar_train_loaders = []
        for i, p in enumerate(parts):
            # Use a small subset for UBAR evaluation to avoid computational overhead
            subset_size = min(64, len(p))  # Small batch for loss evaluation
            subset_indices = random.sample(range(len(p)), subset_size)
            subset_data = Subset(p, subset_indices)
            loader = DataLoader(subset_data, batch_size=32, shuffle=True, num_workers=0)
            ubar_train_loaders.append(loader)

        for i in range(args.num_nodes):
            ubar_monitors[str(i)] = UBAR(str(i), ubar_config, ubar_train_loaders[i], dev)

        print(f"UBAR ALGORITHM (Two-Stage Byzantine-resilient)")
        print(f"  - Model dimension: {model_dim:,} parameters")
        print(f"  - Rho parameter: {args.ubar_rho}")
        print(f"  - Stage 1: Distance-based filtering (select {args.ubar_rho*100:.0f}% closest neighbors)")
        print(f"  - Stage 2: Performance-based selection (loss comparison)")
        print(f"  - Complexity: O(deg(i)×d + deg(i)×inference)")

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
        elif args.agg == "krum":
            decentralized_krum_step(models, graph, args.pct_compromised, r, attacker)
        elif args.agg == "balance":
            balance_aggregation_step(models, graph, balance_monitors, r, attacker)
        elif args.agg == "coarse":
            coarse_aggregation_step(models, graph, coarse_monitors, r, attacker)
        elif args.agg == "ubar":
            ubar_aggregation_step(models, graph, ubar_monitors, r, attacker)
        else:
            raise ValueError("agg must be 'd-fedavg', 'krum', 'balance', 'coarse', or 'ubar'")

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

        if args.verbose:
            if attacker:
                compromised_accs = [accs[i] for i in attacker.compromised_nodes]
                honest_accs = [accs[i] for i in range(args.num_nodes) if i not in attacker.compromised_nodes]
                if compromised_accs and honest_accs:
                    print(f"         : compromised: {np.mean(compromised_accs):.4f}, honest: {np.mean(honest_accs):.4f}")

            if args.agg == "ubar" and ubar_monitors:
                ubar_stats = []
                for node_id, monitor in ubar_monitors.items():
                    stats = monitor.get_statistics()
                    ubar_stats.append(f"Node {node_id}: s1={stats['stage1_mean_acceptance_rate']:.3f}, s2={stats['stage2_mean_acceptance_rate']:.3f}")
                print(f"         : ubar stats = {ubar_stats[:3]}...")

            elif args.agg == "coarse" and coarse_monitors:
                coarse_stats = []
                for node_id, monitor in coarse_monitors.items():
                    stats = monitor.get_statistics()
                    coarse_stats.append(f"Node {node_id}: acc_rate={stats['mean_acceptance_rate']:.3f}")
                print(f"         : coarse stats = {coarse_stats[:3]}...")

    # Final evaluation and summary
    accs = []
    for i, m in enumerate(models):
        acc, _, _, _ = evaluate(m, test_loaders[i], dev)
        accs.append(acc)

    print("\n=== FINAL RESULTS ===")
    print(f"Dataset: {args.dataset}, Nodes: {args.num_nodes}, Graph: {args.graph}, Aggregation: {args.agg}")
    if attacker:
        compromised_accs = [accs[i] for i in attacker.compromised_nodes]
        honest_accs = [accs[i] for i in range(args.num_nodes) if i not in attacker.compromised_nodes]
        if compromised_accs and honest_accs:
            print(f"Attack: {args.attack_type}, {args.attack_percentage*100:.1f}% compromised")
            print(f"Final accuracy - Compromised: {np.mean(compromised_accs):.4f}, Honest: {np.mean(honest_accs):.4f}")
    print(f"Overall test accuracy: mean={np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # BALANCE summary
    if args.agg == "balance" and balance_monitors:
        print(f"\n=== BALANCE SUMMARY ===")
        all_acceptance_rates = []

        # Collect per-node statistics
        node_stats = []
        for node_id, monitor in balance_monitors.items():
            stats = monitor.get_statistics()
            all_acceptance_rates.append(stats["mean_acceptance_rate"])
            node_stats.append(stats)
            print(f"Node {node_id}: acceptance={stats['mean_acceptance_rate']:.3f}")

        # Calculate parallel execution time (max across nodes) and averages
        max_filtering_comp = max(s["filtering_computation_time"] for s in node_stats)
        max_aggregation_comp = max(s["aggregation_computation_time"] for s in node_stats)
        max_model_transfer = max(s["full_model_transfer_time"] for s in node_stats)

        avg_filtering_comp = np.mean([s["filtering_computation_time"] for s in node_stats])
        avg_aggregation_comp = np.mean([s["aggregation_computation_time"] for s in node_stats])
        avg_model_transfer = np.mean([s["full_model_transfer_time"] for s in node_stats])

        sum_filtering_comp = sum(s["filtering_computation_time"] for s in node_stats)
        sum_aggregation_comp = sum(s["aggregation_computation_time"] for s in node_stats)
        sum_model_transfer = sum(s["full_model_transfer_time"] for s in node_stats)

        print(f"\n=== PARALLEL EXECUTION TIME (realistic for distributed system) ===")
        max_computation = max_filtering_comp + max_aggregation_comp
        max_communication = max_model_transfer
        max_total = max_computation + max_communication

        if max_total > 0:
            print(f"  COMMUNICATION (max across nodes):")
            print(f"    - Full model transfer: {max_model_transfer:.3f}s ({max_model_transfer/max_total*100:.1f}%)")
            print(f"  COMPUTATION (max across nodes):")
            print(f"    - Filtering: {max_filtering_comp:.3f}s ({max_filtering_comp/max_total*100:.1f}%)")
            print(f"    - Aggregation: {max_aggregation_comp:.3f}s ({max_aggregation_comp/max_total*100:.1f}%)")
            print(f"  TOTALS:")
            print(f"    - Total computation: {max_computation:.3f}s ({max_computation/max_total*100:.1f}%)")
            print(f"    - Total communication: {max_communication:.3f}s ({max_communication/max_total*100:.1f}%)")
            print(f"    - Total parallel time: {max_total:.3f}s")

        print(f"\n=== PER-NODE AVERAGE TIME ===")
        avg_total = avg_filtering_comp + avg_aggregation_comp + avg_model_transfer
        if avg_total > 0:
            print(f"  - Filtering: {avg_filtering_comp:.3f}s")
            print(f"  - Aggregation: {avg_aggregation_comp:.3f}s")
            print(f"  - Model transfer: {avg_model_transfer:.3f}s")
            print(f"  - Total per node: {avg_total:.3f}s")

        print(f"\n=== TOTAL COMPUTATIONAL WORK (sum across all nodes) ===")
        sum_total = sum_filtering_comp + sum_aggregation_comp + sum_model_transfer
        if sum_total > 0:
            print(f"  - Total filtering: {sum_filtering_comp:.3f}s")
            print(f"  - Total aggregation: {sum_aggregation_comp:.3f}s")
            print(f"  - Total model transfer: {sum_model_transfer:.3f}s")
            print(f"  - Grand total: {sum_total:.3f}s")

        if all_acceptance_rates:
            print(f"  - Mean acceptance rate: {np.mean(all_acceptance_rates):.3f}")

        print(f"\nBALANCE Algorithm Properties:")
        print(f"  - Model dimension: {model_dim:,}")
        print(f"  - No compression: Full parameter comparison")
        print(f"  - Theoretical complexity: O(deg(i)×d)")
        print(f"  - Approach: Full parameter filtering + averaging")

    # COARSE summary
    if args.agg == "coarse" and coarse_monitors:
        print(f"\n=== COARSE SUMMARY ===")
        all_acceptance_rates = []

        # Collect per-node statistics
        node_stats = []
        for node_id, monitor in coarse_monitors.items():
            stats = monitor.get_statistics()
            all_acceptance_rates.append(stats["mean_acceptance_rate"])
            node_stats.append(stats)
            print(f"Node {node_id}: acceptance={stats['mean_acceptance_rate']:.3f}")

        # Calculate parallel execution time (max across nodes) and averages
        max_sketch_comp = max(s["sketch_computation_time"] for s in node_stats)
        max_filtering_comp = max(s["filtering_computation_time"] for s in node_stats)
        max_aggregation_comp = max(s["aggregation_computation_time"] for s in node_stats)
        max_sketch_transfer = max(s["sketch_transfer_time"] for s in node_stats)
        max_model_fetch = max(s["full_model_fetch_time"] for s in node_stats)

        avg_sketch_comp = np.mean([s["sketch_computation_time"] for s in node_stats])
        avg_filtering_comp = np.mean([s["filtering_computation_time"] for s in node_stats])
        avg_aggregation_comp = np.mean([s["aggregation_computation_time"] for s in node_stats])
        avg_sketch_transfer = np.mean([s["sketch_transfer_time"] for s in node_stats])
        avg_model_fetch = np.mean([s["full_model_fetch_time"] for s in node_stats])

        sum_sketch_comp = sum(s["sketch_computation_time"] for s in node_stats)
        sum_filtering_comp = sum(s["filtering_computation_time"] for s in node_stats)
        sum_aggregation_comp = sum(s["aggregation_computation_time"] for s in node_stats)
        sum_sketch_transfer = sum(s["sketch_transfer_time"] for s in node_stats)
        sum_model_fetch = sum(s["full_model_fetch_time"] for s in node_stats)

        print(f"\n=== PARALLEL EXECUTION TIME (realistic for distributed system) ===")
        max_computation = max_sketch_comp + max_filtering_comp + max_aggregation_comp
        max_communication = max_sketch_transfer + max_model_fetch
        max_total = max_computation + max_communication

        if max_total > 0:
            print(f"  COMMUNICATION (max across nodes):")
            print(f"    - Sketch transfer: {max_sketch_transfer:.3f}s ({max_sketch_transfer/max_total*100:.1f}%)")
            print(f"    - Model fetch (accepted): {max_model_fetch:.3f}s ({max_model_fetch/max_total*100:.1f}%)")
            print(f"  COMPUTATION (max across nodes):")
            print(f"    - Sketching: {max_sketch_comp:.3f}s ({max_sketch_comp/max_total*100:.1f}%)")
            print(f"    - Filtering: {max_filtering_comp:.3f}s ({max_filtering_comp/max_total*100:.1f}%)")
            print(f"    - Aggregation: {max_aggregation_comp:.3f}s ({max_aggregation_comp/max_total*100:.1f}%)")
            print(f"  TOTALS:")
            print(f"    - Total computation: {max_computation:.3f}s ({max_computation/max_total*100:.1f}%)")
            print(f"    - Total communication: {max_communication:.3f}s ({max_communication/max_total*100:.1f}%)")
            print(f"    - Total parallel time: {max_total:.3f}s")

        print(f"\n=== PER-NODE AVERAGE TIME ===")
        avg_total = avg_sketch_comp + avg_filtering_comp + avg_aggregation_comp + avg_sketch_transfer + avg_model_fetch
        if avg_total > 0:
            print(f"  - Sketching: {avg_sketch_comp:.3f}s")
            print(f"  - Filtering: {avg_filtering_comp:.3f}s")
            print(f"  - Aggregation: {avg_aggregation_comp:.3f}s")
            print(f"  - Sketch transfer: {avg_sketch_transfer:.3f}s")
            print(f"  - Model fetch: {avg_model_fetch:.3f}s")
            print(f"  - Total per node: {avg_total:.3f}s")

        print(f"\n=== TOTAL COMPUTATIONAL WORK (sum across all nodes) ===")
        sum_total = sum_sketch_comp + sum_filtering_comp + sum_aggregation_comp + sum_sketch_transfer + sum_model_fetch
        if sum_total > 0:
            print(f"  - Total sketching: {sum_sketch_comp:.3f}s")
            print(f"  - Total filtering: {sum_filtering_comp:.3f}s")
            print(f"  - Total aggregation: {sum_aggregation_comp:.3f}s")
            print(f"  - Total sketch transfer: {sum_sketch_transfer:.3f}s")
            print(f"  - Total model fetch: {sum_model_fetch:.3f}s")
            print(f"  - Grand total: {sum_total:.3f}s")

        if all_acceptance_rates:
            print(f"  - Mean acceptance rate: {np.mean(all_acceptance_rates):.3f}")

        # Algorithm properties
        actual_speedup = (args.num_nodes * model_dim) / (model_dim + args.num_nodes * args.coarse_sketch_size)

        print(f"\nCOARSE Algorithm Properties:")
        print(f"  - Original dimension: {model_dim:,}")
        print(f"  - Sketch size: {args.coarse_sketch_size}")
        print(f"  - Compression ratio: {actual_speedup:.1f}x")
        print(f"  - Single repetition: No repetitions needed")
        print(f"  - Theoretical complexity: O(d + N×k)")
        print(f"  - Approach: Sketch filtering + state aggregation")

    # UBAR summary
    if args.agg == "ubar" and ubar_monitors:
        print(f"\n=== UBAR SUMMARY ===")
        all_stage1_rates = []
        all_stage2_rates = []

        # Collect per-node statistics
        node_stats = []
        for node_id, monitor in ubar_monitors.items():
            stats = monitor.get_statistics()
            all_stage1_rates.append(stats["stage1_mean_acceptance_rate"])
            all_stage2_rates.append(stats["stage2_mean_acceptance_rate"])
            node_stats.append(stats)

            print(f"Node {node_id}: stage1={stats['stage1_mean_acceptance_rate']:.3f}, "
                  f"stage2={stats['stage2_mean_acceptance_rate']:.3f}, "
                  f"overall={stats['overall_acceptance_rate']:.3f}")

        # Calculate parallel execution time (max across nodes) and averages
        max_distance_comp = max(s["distance_computation_time"] for s in node_stats)
        max_loss_comp = max(s["loss_computation_time"] for s in node_stats)
        max_aggregation_comp = max(s["aggregation_computation_time"] for s in node_stats)
        max_model_transfer = max(s["full_model_transfer_time"] for s in node_stats)

        avg_distance_comp = np.mean([s["distance_computation_time"] for s in node_stats])
        avg_loss_comp = np.mean([s["loss_computation_time"] for s in node_stats])
        avg_aggregation_comp = np.mean([s["aggregation_computation_time"] for s in node_stats])
        avg_model_transfer = np.mean([s["full_model_transfer_time"] for s in node_stats])

        sum_distance_comp = sum(s["distance_computation_time"] for s in node_stats)
        sum_loss_comp = sum(s["loss_computation_time"] for s in node_stats)
        sum_aggregation_comp = sum(s["aggregation_computation_time"] for s in node_stats)
        sum_model_transfer = sum(s["full_model_transfer_time"] for s in node_stats)

        print(f"\n=== PARALLEL EXECUTION TIME (realistic for distributed system) ===")
        max_computation = max_distance_comp + max_loss_comp + max_aggregation_comp
        max_communication = max_model_transfer
        max_total = max_computation + max_communication

        if max_total > 0:
            print(f"  COMMUNICATION (max across nodes):")
            print(f"    - Full model transfer: {max_model_transfer:.3f}s ({max_model_transfer/max_total*100:.1f}%)")
            print(f"  COMPUTATION (max across nodes):")
            print(f"    - Distance computation: {max_distance_comp:.3f}s ({max_distance_comp/max_total*100:.1f}%)")
            print(f"    - Loss computation: {max_loss_comp:.3f}s ({max_loss_comp/max_total*100:.1f}%)")
            print(f"    - Aggregation: {max_aggregation_comp:.3f}s ({max_aggregation_comp/max_total*100:.1f}%)")
            print(f"  TOTALS:")
            print(f"    - Total computation: {max_computation:.3f}s ({max_computation/max_total*100:.1f}%)")
            print(f"    - Total communication: {max_communication:.3f}s ({max_communication/max_total*100:.1f}%)")
            print(f"    - Total parallel time: {max_total:.3f}s")

        print(f"\n=== PER-NODE AVERAGE TIME ===")
        avg_total = avg_distance_comp + avg_loss_comp + avg_aggregation_comp + avg_model_transfer
        if avg_total > 0:
            print(f"  - Distance computation: {avg_distance_comp:.3f}s")
            print(f"  - Loss computation: {avg_loss_comp:.3f}s")
            print(f"  - Aggregation: {avg_aggregation_comp:.3f}s")
            print(f"  - Model transfer: {avg_model_transfer:.3f}s")
            print(f"  - Total per node: {avg_total:.3f}s")

        print(f"\n=== TOTAL COMPUTATIONAL WORK (sum across all nodes) ===")
        sum_total = sum_distance_comp + sum_loss_comp + sum_aggregation_comp + sum_model_transfer
        if sum_total > 0:
            print(f"  - Total distance computation: {sum_distance_comp:.3f}s")
            print(f"  - Total loss computation: {sum_loss_comp:.3f}s")
            print(f"  - Total aggregation: {sum_aggregation_comp:.3f}s")
            print(f"  - Total model transfer: {sum_model_transfer:.3f}s")
            print(f"  - Grand total: {sum_total:.3f}s")

        if all_stage1_rates and all_stage2_rates:
            print(f"  - Mean Stage 1 acceptance rate: {np.mean(all_stage1_rates):.3f}")
            print(f"  - Mean Stage 2 acceptance rate: {np.mean(all_stage2_rates):.3f}")
            print(f"  - Overall acceptance rate: {np.mean(all_stage1_rates) * np.mean(all_stage2_rates):.3f}")

        print(f"\nUBAR Algorithm Properties:")
        print(f"  - Model dimension: {model_dim:,}")
        print(f"  - Rho parameter: {args.ubar_rho}")
        print(f"  - Two-stage approach: Distance filtering + loss evaluation")
        print(f"  - Stage 1 selects: {args.ubar_rho*100:.0f}% of neighbors")
        print(f"  - Stage 2 uses: Training sample loss comparison")
        print(f"  - Theoretical complexity: O(deg(i)×d + deg(i)×inference)")
        print(f"  - Approach: UBAR paper implementation")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Decentralized FL Simulator with BALANCE, COARSE, and UBAR")

    # Dataset and basic training parameters
    p.add_argument("--dataset", type=str, choices=["femnist", "celeba"], required=True)
    p.add_argument("--num-nodes", type=int, default=8)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max-samples", type=int, default=None)

    # Aggregation algorithm parameters
    p.add_argument("--agg", type=str,
                   choices=["d-fedavg", "krum", "balance", "coarse", "ubar"],
                   default="d-fedavg",
                   help="Aggregation algorithm")
    p.add_argument("--pct-compromised", type=float, default=0.0)

    # BALANCE algorithm parameters
    p.add_argument("--balance-gamma", type=float, default=2)
    p.add_argument("--balance-kappa", type=float, default=1)
    p.add_argument("--balance-alpha", type=float, default=0.5)

    # COARSE specific parameters
    p.add_argument("--coarse-sketch-size", type=int, default=1000,
                   help="COARSE sketch size k (lower = more compression)")

    # UBAR specific parameters
    p.add_argument("--ubar-rho", type=float, default=0.4,
                   help="UBAR rho parameter (ratio of benign neighbors)")

    # Graph topology parameters
    p.add_argument("--graph", type=str, choices=["ring", "fully", "erdos", "k-regular"], default="ring",
                   help="Graph topology: ring (degree=2), fully (degree=n-1), erdos (random), k-regular (degree=k)")
    p.add_argument("--p", type=float, default=0.3, help="Edge probability for Erdos-Renyi graphs")
    p.add_argument("--k", type=int, default=4, help="Degree k for k-regular graphs (must be even)")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    # Attack parameters
    p.add_argument("--attack-percentage", type=float, default=0.0)
    p.add_argument("--attack-type", type=str, choices=["directed_deviation", "gaussian"],
                   default="directed_deviation")
    p.add_argument("--attack-lambda", type=float, default=1.0)

    # Debug/verbose
    p.add_argument("--verbose", action="store_true")

    # Model variant
    p.add_argument("--model-variant", type=str,
                   choices=["tiny", "small", "baseline", "large", "xlarge"],
                   default="baseline",
                   help="Model size variant to use (only for FEMNIST)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sim(args)
