"""Tiny synthetic model + dataset for download-free smoke tests (esp. the distributed backend)."""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from murmura.data import DatasetAdapter


def synthetic_mlp(input_dim: int = 10, hidden: int = 64, num_classes: int = 2) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))


def synthetic_adapter(num_nodes: int = 4, samples_per_node: int = 64, input_dim: int = 10,
                      num_classes: int = 2, seed: int = 0) -> DatasetAdapter:
    g = torch.Generator().manual_seed(seed)
    n = num_nodes * samples_per_node
    x = torch.randn(n, input_dim, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    parts = [list(range(i * samples_per_node, (i + 1) * samples_per_node)) for i in range(num_nodes)]
    return DatasetAdapter(TensorDataset(x, y), parts)
