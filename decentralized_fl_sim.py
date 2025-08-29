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


# ---------------------------- Data ---------------------------- #

# Global variables to store computed normalization stats
_norm_stats = {}

class LEAFFEMNISTDataset(torch.utils.data.Dataset):
    """Dataset class for LEAF FEMNIST data."""
    
    def __init__(self, data_list, targets_list, transform=None):
        self.data = []
        self.targets = []
        
        # Flatten client data into a single dataset
        for client_data, client_targets in zip(data_list, targets_list):
            self.data.extend(client_data)
            self.targets.extend(client_targets)
            
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # LEAF FEMNIST data comes as 28x28 pixel arrays
        img_array = np.array(self.data[idx], dtype=np.float32).reshape(28, 28)
        img = Image.fromarray(img_array)
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

class LEAFFEMNISTClientDataset(torch.utils.data.Dataset):
    """Dataset class for individual LEAF FEMNIST clients."""
    
    def __init__(self, client_data, client_targets, transform=None):
        self.data = client_data
        self.targets = client_targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # LEAF FEMNIST data comes as 28x28 pixel arrays
        img_array = np.array(self.data[idx], dtype=np.float32).reshape(28, 28)
        img = Image.fromarray(img_array)
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

def download_leaf_femnist():
    """Download LEAF FEMNIST dataset if not already present."""
    data_dir = "./data/leaf_femnist"
    os.makedirs(data_dir, exist_ok=True)
    
    # LEAF FEMNIST download URLs (using smaller sample for faster download)
    urls = {
        "train": "https://raw.githubusercontent.com/TalwalkarLab/leaf/master/data/femnist/data/train/all_data_0_niid_05_keep_0_train_9.json",
        "test": "https://raw.githubusercontent.com/TalwalkarLab/leaf/master/data/femnist/data/test/all_data_0_niid_05_keep_0_test_9.json"
    }
    
    files = {}
    for split, url in urls.items():
        file_path = os.path.join(data_dir, f"{split}.json")
        if not os.path.exists(file_path):
            print(f"Downloading LEAF FEMNIST {split} data...")
            urllib.request.urlretrieve(url, file_path)
        files[split] = file_path
    
    return files

def load_leaf_femnist_data():
    """Load LEAF FEMNIST data from JSON files."""
    files = download_leaf_femnist()
    
    # Load train data
    with open(files["train"], 'r') as f:
        train_data = json.load(f)
    
    # Load test data  
    with open(files["test"], 'r') as f:
        test_data = json.load(f)
    
    return train_data, test_data

def compute_normalization_stats(dataset, dataset_name: str):
    """Compute mean and std for dataset normalization from torchvision dataset."""
    if dataset_name in _norm_stats:
        return _norm_stats[dataset_name]
    
    print(f"Computing normalization statistics for {dataset_name}...")
    
    # Sample a subset for efficiency (use first 5000 images)
    sample_size = min(5000, len(dataset))
    indices = list(range(sample_size))
    
    pixel_values = []
    for i in indices:
        img, _ = dataset[i]  # torchvision dataset returns (img, label)
        if hasattr(img, "mode"):  # PIL Image
            tensor_img = T.ToTensor()(img)
        else:
            tensor_img = img
        pixel_values.append(tensor_img.view(tensor_img.size(0), -1))
    
    # Stack all images and compute statistics
    all_pixels = torch.cat(pixel_values, dim=1)  # Shape: (3, total_pixels)
    mean = all_pixels.mean(dim=1).tolist()
    std = all_pixels.std(dim=1).tolist()
    
    _norm_stats[dataset_name] = {"mean": mean, "std": std}
    print(f"{dataset_name} - Mean: {mean}, Std: {std}")
    return _norm_stats[dataset_name]

def load_dataset(name: str):
    """Load dataset using torchvision instead of HuggingFace."""
    name = name.lower()
    if name == "cifar10":
        # Load raw datasets without transforms first
        train_ds_raw = datasets.CIFAR10(root="./data", train=True, download=True, transform=None)
        test_ds_raw = datasets.CIFAR10(root="./data", train=False, download=True, transform=None)
        
        # Compute normalization statistics
        compute_normalization_stats(train_ds_raw, "cifar10")
        
        # Create datasets with computed transforms
        stats = _norm_stats["cifar10"]
        normalize = T.Normalize(mean=stats["mean"], std=stats["std"])
        
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
        
        test_transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])
        
        train_ds = datasets.CIFAR10(root="./data", train=True, download=False, transform=train_transform)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=False, transform=test_transform)
        
        num_classes = 10
        image_size = 32
        
    elif name == "femnist":
        # Load LEAF FEMNIST data with natural client partitions
        train_data, test_data = load_leaf_femnist_data()
        
        # Extract all client data for normalization stats computation
        all_train_data = []
        all_train_targets = []
        for client in train_data["users"]:
            all_train_data.extend(train_data["user_data"][client]["x"])
            all_train_targets.extend(train_data["user_data"][client]["y"])
        
        # Create temporary dataset for normalization computation
        temp_ds = LEAFFEMNISTDataset([all_train_data], [all_train_targets], transform=None)
        compute_normalization_stats(temp_ds, "femnist")
        
        # Create datasets with computed transforms
        stats = _norm_stats["femnist"]
        normalize = T.Normalize(mean=stats["mean"], std=stats["std"])
        
        train_transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])
        
        test_transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])
        
        # Create train dataset
        all_test_data = []
        all_test_targets = []
        for client in test_data["users"]:
            all_test_data.extend(test_data["user_data"][client]["x"])
            all_test_targets.extend(test_data["user_data"][client]["y"])
        
        train_ds = LEAFFEMNISTDataset([all_train_data], [all_train_targets], transform=train_transform)
        test_ds = LEAFFEMNISTDataset([all_test_data], [all_test_targets], transform=test_transform)
        
        # Store client data for partitioning
        train_ds.client_data = train_data
        test_ds.client_data = test_data
        
        num_classes = 62  # FEMNIST has 62 classes (10 digits + 52 letters)
        image_size = 28
    
    else:
        raise ValueError("dataset must be 'cifar10' or 'femnist'")

    return train_ds, test_ds, num_classes, image_size


# ---------------------------- Model ---------------------------- #
class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, image_size: int = 32):
        super().__init__()
        # Conv stack
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # size/2

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # size/4
        )
        feat_size = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feat_size * feat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------- Training helpers ---------------------------- #

def create_cifar_iid_partitions(dataset: torch.utils.data.Dataset, num_nodes: int, seed: int) -> List[Subset]:
    """Create truly IID partitions for CIFAR-10 by splitting each class equally across clients."""
    rng = np.random.default_rng(seed)
    
    # Group indices by class
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Shuffle indices within each class
    for label in class_indices:
        rng.shuffle(class_indices[label])
    
    # Split each class equally across nodes
    node_indices = [[] for _ in range(num_nodes)]
    for label, indices in class_indices.items():
        class_parts = np.array_split(indices, num_nodes)
        for node_id, part in enumerate(class_parts):
            node_indices[node_id].extend(part.tolist())
    
    # Shuffle each node's indices to mix classes
    for node_id in range(num_nodes):
        rng.shuffle(node_indices[node_id])
    
    return [Subset(dataset, indices) for indices in node_indices]

def create_femnist_writer_partitions(dataset: torch.utils.data.Dataset, num_nodes: int, seed: int) -> List[Subset]:
    """Create writer-based non-IID partitions using LEAF FEMNIST's natural client structure."""
    rng = np.random.default_rng(seed)
    
    # Use natural LEAF client partitions
    if hasattr(dataset, 'client_data'):
        client_data = dataset.client_data
        users = client_data["users"]
        
        print(f"LEAF FEMNIST: Found {len(users)} natural writers/clients")
        
        # If we have fewer nodes than natural clients, group clients
        # If we have more nodes than natural clients, split large clients
        if len(users) >= num_nodes:
            # Select a subset of natural clients
            rng.shuffle(users)
            selected_users = users[:num_nodes]
            
            node_indices = []
            current_idx = 0
            
            for user in selected_users:
                user_data_len = len(client_data["user_data"][user]["x"])
                user_indices = list(range(current_idx, current_idx + user_data_len))
                node_indices.append(user_indices)
                current_idx += user_data_len
            
            print(f"Using {len(selected_users)} natural clients directly")
            
        else:
            # Group multiple clients per node
            clients_per_node = len(users) // num_nodes
            remaining_clients = len(users) % num_nodes
            
            node_indices = [[] for _ in range(num_nodes)]
            current_idx = 0
            user_idx = 0
            
            for node_id in range(num_nodes):
                # Number of clients for this node
                num_clients_for_node = clients_per_node + (1 if node_id < remaining_clients else 0)
                
                for _ in range(num_clients_for_node):
                    if user_idx < len(users):
                        user = users[user_idx]
                        user_data_len = len(client_data["user_data"][user]["x"])
                        user_indices = list(range(current_idx, current_idx + user_data_len))
                        node_indices[node_id].extend(user_indices)
                        current_idx += user_data_len
                        user_idx += 1
            
            print(f"Grouped {len(users)} natural clients into {num_nodes} nodes")
        
        return [Subset(dataset, indices) for indices in node_indices]
    
    else:
        # Fallback to class-based partitioning if no client data available
        print("No LEAF client data found, falling back to class-based partitioning")
        return create_cifar_iid_partitions(dataset, num_nodes, seed)

def create_partitions(dataset: torch.utils.data.Dataset, dataset_name: str, num_nodes: int, seed: int) -> List[Subset]:
    """Create appropriate partitions based on dataset type."""
    if dataset_name.lower() == "cifar10":
        return create_cifar_iid_partitions(dataset, num_nodes, seed)
    elif dataset_name.lower() == "femnist":
        return create_femnist_writer_partitions(dataset, num_nodes, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate(model: nn.Module, loader: DataLoader, device_: torch.device) -> Tuple[float, float]:
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
    return correct / max(1, total), loss_sum / max(1, total)


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
            acc, _ = evaluate(m, test_loaders[i], dev)
            base_accs.append(acc)
        print(f"Initial test acc across nodes: mean={np.mean(base_accs):.4f} ± {np.std(base_accs):.4f}")

    # Training rounds
    for r in range(1, args.rounds + 1):
        # Local training at each node
        for i, (m, ld) in enumerate(zip(models, loaders)):
            local_train(m, ld, epochs=args.local_epochs, lr=args.lr, device_=dev)

        # Communication / aggregation
        if args.agg == "d-fedadj":
            decentralized_fedavg_step(models, graph)
        elif args.agg == "gossip":
            gossip_round(models, graph, steps=args.gossip_steps)
        else:
            raise ValueError("agg must be 'd-fedadj' or 'gossip'")

        # Evaluation: each client tests on their own user's test data
        accs = []
        for i, m in enumerate(models):
            acc, _ = evaluate(m, test_loaders[i], dev)
            accs.append(acc)
        print(f"Round {r:03d}: test acc mean={np.mean(accs):.4f} ± {np.std(accs):.4f} | min={np.min(accs):.4f} max={np.max(accs):.4f}")

    # Final summary - each client evaluates on their own test data
    accs = []
    for i, m in enumerate(models):
        acc, _ = evaluate(m, test_loaders[i], dev)
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
