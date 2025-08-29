#!/usr/bin/env python3
"""
Decentralized Learning Simulator for LEAF Datasets (PyTorch)

Supports two aggregation strategies over a peer graph:
  1) Decentralized FedAvg (synchronous neighbor averaging per round)
  2) Gossip averaging (asynchronous-style, simulated with K random edge gossips per round)

LEAF Datasets:
  - FEMNIST: Handwritten characters by writer (non-IID, natural federated dataset)
  - Sent140: Twitter sentiment analysis by user (non-IID, natural federated dataset)

Features:
  - Uses LEAF's natural client partitioning (writer-based for FEMNIST)
  - Client-specific train/test splits following LEAF methodology  
  - LEAF's standard model architectures (CNN for FEMNIST, LSTM for Sent140)

Example usage:
  python decentralized_fl_sim.py \
      --dataset femnist --num-nodes 8 --rounds 20 --local-epochs 1 \
      --agg d-fedadj --graph ring --lr 0.01

  python decentralized_fl_sim.py \
      --dataset sent140 --num-nodes 10 --rounds 30 --local-epochs 1 \
      --agg gossip --gossip-steps 50 --graph erdos --p 0.3

Notes:
  - Requires LEAF dataset preprocessing (see leaf/data/*/preprocess.sh)
  - Uses writer-based non-IID partitioning for realistic federated learning simulation
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, RandomSampler

from leaf_datasets import (
    load_leaf_dataset, 
    create_leaf_client_partitions,
    LEAFFEMNISTModel,
    LEAFSent140Model
)


# ---------------------------- Utilities ---------------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@dataclass
class Graph:
    n: int
    neighbors: List[List[int]]  # adjacency list, include self if desired in logic
    edges: List[Tuple[int, int]]


def make_graph(n: int, kind: str, p: float = 0.3) -> Graph:
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
    elif kind in ("erdos", "er"):  # G(n, p)
        rng = random.Random(12345)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                    edges.append((i, j))
        # ensure connectivity minimally; if isolated nodes, connect to nearest by index
        for i in range(n):
            if not neighbors[i]:
                j = (i + 1) % n
                neighbors[i].append(j)
                neighbors[j].append(i)
                edges.append((min(i, j), max(i, j)))
    else:
        raise ValueError(f"Unknown graph kind: {kind}")

    # Deduplicate neighbor lists
    neighbors = [sorted(set(ns)) for ns in neighbors]
    edges = sorted(set(tuple(sorted(e)) for e in edges))
    return Graph(n=n, neighbors=neighbors, edges=edges)


# All dataset loading is now handled by leaf_datasets.py


# ---------------------------- Training helpers ---------------------------- #

# All partitioning is now handled by create_leaf_client_partitions in leaf_datasets.py


def evaluate(model: nn.Module, loader: DataLoader, device_: torch.device) -> Tuple[float, float, int, int]:
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
    # More efficient state extraction
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def set_state(model: nn.Module, state: Dict[str, torch.Tensor]):
    # Faster loading with strict=False (since we know the structure matches)
    model.load_state_dict(state, strict=False)


def average_states(states: List[Dict[str, torch.Tensor]], weights: List[float] | None = None) -> Dict[str, torch.Tensor]:
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


# ---------------------------- Aggregation strategies ---------------------------- #

def decentralized_fedavg_step(models: List[nn.Module], graph: Graph):
    # Each node averages with its neighbors (including itself) equally
    states = [get_state(m) for m in models]
    new_states = []
    for i in range(graph.n):
        neigh = [i] + graph.neighbors[i]
        neigh_states = [states[j] for j in neigh]
        w = [1.0 / len(neigh_states)] * len(neigh_states)
        new_states.append(average_states(neigh_states, w))
    for model, st in zip(models, new_states):
        set_state(model, st)


def gossip_round(models: List[nn.Module], graph: Graph, steps: int = 1):
    # Randomly select edges; endpoints average their parameters
    if not graph.edges:
        return
    for _ in range(steps):
        i, j = random.choice(graph.edges)
        si, sj = get_state(models[i]), get_state(models[j])
        avg = average_states([si, sj], [0.5, 0.5])
        set_state(models[i], avg)
        set_state(models[j], avg)


# ---------------------------- Simulator ---------------------------- #

def run_sim(args):
    set_seed(args.seed)
    dev = device()
    print(f"Device: {dev}")

    # Load LEAF dataset with appropriate model architecture
    if args.dataset.lower() == "femnist":
        data_path = "./leaf/data/femnist/data"
        train_ds, test_ds, model_template, num_classes, input_size = load_leaf_dataset("femnist", data_path)
        image_size = input_size
    elif args.dataset.lower() == "sent140": 
        data_path = "./leaf/data/sent140/data"
        train_ds, test_ds, model_template, num_classes, input_size = load_leaf_dataset("sent140", data_path)
        image_size = input_size
    else:
        raise ValueError(f"Dataset {args.dataset} not supported. Use 'femnist' or 'sent140'")

    # Create client partitions using LEAF's natural user groupings
    train_partitions, test_partitions = create_leaf_client_partitions(train_ds, test_ds, args.num_nodes, seed=args.seed)
    parts = [Subset(train_ds, indices) for indices in train_partitions]
    test_parts = [Subset(test_ds, indices) for indices in test_partitions]
    
    # Print partition info and quality  
    print("Using non-IID partitions (writer-based for FEMNIST, user-based for Sent140)")
    # Show partition sizes and class diversity
    for i in range(args.num_nodes):
        class_counts = {}
        # Check ALL samples to get accurate class distribution
        for idx in parts[i].indices:
            _, label = train_ds[idx]
            class_counts[label] = class_counts.get(label, 0) + 1
        unique_classes = len(class_counts)
        class_dist = ", ".join([f"class {k}: {v}" for k, v in sorted(class_counts.items())])
        print(f"  Client {i}: {len(parts[i])} samples, {unique_classes} unique classes [{class_dist}]")

    # Dataloaders per node - optimized for M3 Pro
    num_workers = 4  # M3 Pro has good multiprocessing capabilities
    pin_memory = dev.type != "cpu"  # Only use pin_memory for GPU
    
    # Create data loaders with optional sampling
    if args.max_samples:
        # Sample a subset of data per epoch
        loaders = []
        for p in parts:
            num_samples = min(args.max_samples, len(p))
            sampler = RandomSampler(p, replacement=False, num_samples=num_samples)
            loader = DataLoader(p, batch_size=args.batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory, 
                              persistent_workers=True, prefetch_factor=2)
            loaders.append(loader)
        print(f"Sampling {args.max_samples} samples per client per epoch")
    else:
        loaders = [DataLoader(p, batch_size=args.batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=pin_memory, 
                             persistent_workers=True, prefetch_factor=2) for p in parts]
    
    # Create client-specific test loaders (following LEAF methodology)
    test_loaders = [DataLoader(tp, batch_size=512, shuffle=False, 
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=True, prefetch_factor=2) for tp in test_parts]

    # Graph
    graph = make_graph(args.num_nodes, args.graph, p=args.p)
    print(f"Graph kind={args.graph}, edges={len(graph.edges)}")

    # Initialize node models using LEAF architectures
    models = []
    for _ in range(args.num_nodes):
        if args.dataset.lower() == "femnist":
            model = LEAFFEMNISTModel(num_classes=num_classes).to(dev)
        elif args.dataset.lower() == "sent140":
            model = LEAFSent140Model(vocab_size=train_ds.vocab_size).to(dev)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        # Compile model for faster execution (PyTorch 2.0+) - skip for MPS as it's not supported yet
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

    # Training rounds
    for r in range(1, args.rounds + 1):
        # Local training at each node
        # Track if models are actually changing
        pre_train_params = [get_state(m) for m in models]
        
        for i, (m, ld) in enumerate(zip(models, loaders)):
            local_train(m, ld, epochs=args.local_epochs, lr=args.lr, device_=dev)
        
        # Check if any models changed
        param_changes = []
        for i, m in enumerate(models):
            post_params = get_state(m)
            change = sum((post_params[k] - pre_train_params[i][k]).abs().sum().item() 
                        for k in post_params.keys())
            param_changes.append(change)
        print(f"Round {r}: parameter changes = {[f'{change:.6f}' for change in param_changes[:3]]}...")

        # Communication / aggregation
        if args.agg == "d-fedadj":
            decentralized_fedavg_step(models, graph)
        elif args.agg == "gossip":
            gossip_round(models, graph, steps=args.gossip_steps)
        else:
            raise ValueError("agg must be 'd-fedadj' or 'gossip'")

        # Evaluation: each client tests on their own user's test data
        accs = []
        losses = []
        corrects = []
        totals = []
        for i, m in enumerate(models):
            acc, loss, correct, total = evaluate(m, test_loaders[i], dev)
            accs.append(acc)
            losses.append(loss)
            corrects.append(correct)
            totals.append(total)
        
        print(f"Round {r:03d}: test acc mean={np.mean(accs):.4f} ± {np.std(accs):.4f} | min={np.min(accs):.4f} max={np.max(accs):.4f}")
        print(f"         : test loss mean={np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"         : individual accs = {[f'{acc:.6f}' for acc in accs]}")
        print(f"         : correct/total = {[(c, t) for c, t in zip(corrects[:3], totals[:3])]}...")

    # Final summary - each client evaluates on their own test data
    accs = []
    for i, m in enumerate(models):
        acc, _, _, _ = evaluate(m, test_loaders[i], dev)
        accs.append(acc)
    print("\n=== FINAL ===")
    print(f"Nodes: {args.num_nodes}, Graph: {args.graph}, Agg: {args.agg}")
    print(f"Test accuracy across nodes: mean={np.mean(accs):.4f} ± {np.std(accs):.4f}")


# ---------------------------- CLI ---------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Decentralized Learning Simulator (FedAvg-style & Gossip)")
    p.add_argument("--dataset", type=str, choices=["femnist", "sent140"], required=True)
    p.add_argument("--num-nodes", type=int, default=8)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max-samples", type=int, default=None, 
                   help="Max samples per client per epoch (for large datasets like Sent140)")
    p.add_argument("--agg", type=str, choices=["d-fedadj", "gossip"], default="d-fedadj",
                   help="d-fedadj = average with neighbors synchronously; gossip = random edge averaging")
    p.add_argument("--gossip-steps", type=int, default=10, help="number of random edge gossips per round")
    p.add_argument("--graph", type=str, choices=["ring", "fully", "erdos"], default="ring")
    p.add_argument("--p", type=float, default=0.3, help="edge prob for erdos graph")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_sim(args)
