#!/usr/bin/env python3
"""
LEAF dataset loaders and model architectures for federated learning research.
Implements PyTorch versions of LEAF's standard models and data loading.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Tuple, Optional

# Add leaf directory to path
sys.path.append('./leaf')
sys.path.append('./leaf/data')


class LEAFFEMNISTDataset(Dataset):
    """LEAF FEMNIST Dataset - Handwritten digits and letters by writer."""
    
    def __init__(self, data_path: str, split: str = "train", transform=None):
        self.split = split
        self.transform = transform
        
        # Load all LEAF JSON files from the train/test directory
        split_dir = os.path.join(data_path, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"LEAF FEMNIST {split} directory not found at {split_dir}")
        
        # Find all JSON files in the split directory
        json_files = [f for f in os.listdir(split_dir) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {split_dir}")
        
        print(f"Loading {len(json_files)} LEAF FEMNIST {split} files...")
        
        # Combine data from all JSON files
        self.users = []
        self.user_data = {}
        self.num_samples = []
        
        for json_file in sorted(json_files):
            file_path = os.path.join(split_dir, json_file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Merge data from this file
            self.users.extend(data['users'])
            self.user_data.update(data['user_data'])
            self.num_samples.extend(data['num_samples'])
        
        # Flatten all data for centralized access
        self.all_data = []
        self.all_targets = []
        self.user_indices = {}  # Maps user -> list of indices
        
        current_idx = 0
        for user in self.users:
            user_x = self.user_data[user]['x']
            user_y = self.user_data[user]['y']
            
            start_idx = current_idx
            self.all_data.extend(user_x)
            self.all_targets.extend(user_y) 
            end_idx = current_idx + len(user_x)
            
            self.user_indices[user] = list(range(start_idx, end_idx))
            current_idx = end_idx
            
        print(f"LEAF FEMNIST {split}: {len(self.users)} users, {len(self.all_data)} samples")
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        # LEAF FEMNIST images are 28x28 pixels, values in [0,1]
        img_data = np.array(self.all_data[idx], dtype=np.float32).reshape(28, 28)
        img = Image.fromarray((img_data * 255).astype(np.uint8))
        target = self.all_targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target
    
    def get_user_data(self, user: str) -> List[int]:
        """Get indices for a specific user's data."""
        return self.user_indices.get(user, [])


class LEAFSent140Dataset(Dataset):
    """LEAF Sent140 Dataset - Sentiment classification from Twitter."""
    
    def __init__(self, data_path: str, split: str = "train", max_seq_len: int = 25):
        self.split = split
        self.max_seq_len = max_seq_len
        
        # Load LEAF JSON data from multiple files (similar to FEMNIST)
        split_dir = os.path.join(data_path, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"LEAF Sent140 {split} directory not found at {split_dir}")
            
        json_files = [f for f in os.listdir(split_dir) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {split_dir}")
        
        print(f"Loading {len(json_files)} LEAF Sent140 {split} files...")
        
        # Combine data from all JSON files
        self.users = []
        self.user_data = {}
        self.num_samples = []
        
        for json_file in sorted(json_files):
            file_path = os.path.join(split_dir, json_file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Merge data from this file
            self.users.extend(data['users'])
            self.user_data.update(data['user_data'])
            self.num_samples.extend(data['num_samples'])
        
        # Load vocabulary
        vocab_file = os.path.join(data_path, "embs.json")
        if os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                vocab_data = json.load(f)
                self.word_to_idx = vocab_data.get('vocab', {})
        else:
            # Build vocabulary if not available
            self.word_to_idx = self._build_vocab()
        
        self.vocab_size = len(self.word_to_idx)
        
        # Flatten all data
        self.all_data = []
        self.all_targets = []
        self.user_indices = {}
        
        current_idx = 0
        for user in self.users:
            user_x = self.user_data[user]['x'] 
            user_y = self.user_data[user]['y']
            
            start_idx = current_idx
            self.all_data.extend(user_x)
            self.all_targets.extend(user_y)
            end_idx = current_idx + len(user_x)
            
            self.user_indices[user] = list(range(start_idx, end_idx))
            current_idx = end_idx
            
        print(f"LEAF Sent140 {split}: {len(self.users)} users, {len(self.all_data)} samples")
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from all text data."""
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for user in self.users:
            for tweet in self.user_data[user]['x']:
                # tweet[4] contains the text
                words = tweet[4].lower().split()
                for word in words:
                    if word not in vocab:
                        vocab[word] = len(vocab)
        return vocab
    
    def _text_to_indices(self, text: str) -> List[int]:
        """Convert text to sequence of word indices."""
        words = text.lower().split()[:self.max_seq_len]
        indices = [self.word_to_idx.get(word, self.word_to_idx["<UNK>"]) for word in words]
        # Pad to max_seq_len
        while len(indices) < self.max_seq_len:
            indices.append(self.word_to_idx["<PAD>"])
        return indices
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        tweet_data = self.all_data[idx]
        text = tweet_data[4]  # Text is in position 4
        target = int(self.all_targets[idx])
        
        # Convert text to indices
        text_indices = torch.tensor(self._text_to_indices(text), dtype=torch.long)
        
        return text_indices, target
    
    def get_user_data(self, user: str) -> List[int]:
        """Get indices for a specific user's data."""
        return self.user_indices.get(user, [])


# LEAF Model Architectures (PyTorch implementations)

class LEAFFEMNISTModel(nn.Module):
    """LEAF FEMNIST CNN Model - PyTorch version of LEAF's TensorFlow model."""
    
    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.num_classes = num_classes
        
        # Match LEAF architecture exactly
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2) 
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # After pooling: 28->14->7, so 7x7x64
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        # Input: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class LEAFSent140Model(nn.Module):
    """LEAF Sent140 Stacked LSTM Model - PyTorch version."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 300, hidden_dim: int = 256, 
                 num_classes: int = 2, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, hidden_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim)
        
        # Take last output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # FC layers
        x = F.relu(self.fc1(last_output))
        x = self.fc2(x)
        
        return x


def create_leaf_client_partitions(train_dataset, test_dataset, num_nodes: int, seed: int = 42) -> Tuple[List[List[int]], List[List[int]]]:
    """Create client partitions using LEAF's natural user groupings.
    
    Returns:
        train_partitions, test_partitions - matching user assignments
    """
    rng = np.random.RandomState(seed)
    
    if not hasattr(train_dataset, 'user_indices') or not hasattr(test_dataset, 'user_indices'):
        raise ValueError("Both datasets must have user_indices for LEAF partitioning")
    
    # Use training users as the reference (some users might not have test data)
    train_users = set(train_dataset.user_indices.keys())
    test_users = set(test_dataset.user_indices.keys())
    common_users = list(train_users.intersection(test_users))
    
    # Sort users by training sample count (largest first)
    user_sample_counts = []
    for user in common_users:
        train_samples = len(train_dataset.user_indices[user])
        user_sample_counts.append((user, train_samples))
    
    # Sort by sample count in descending order
    user_sample_counts.sort(key=lambda x: x[1], reverse=True)
    sorted_users = [user for user, _ in user_sample_counts]
    
    print(f"Found {len(train_users)} train users, {len(test_users)} test users, {len(common_users)} common users")
    print(f"User sample counts range: {user_sample_counts[0][1]} (max) to {user_sample_counts[-1][1]} (min)")
    
    if len(sorted_users) >= num_nodes:
        # For Sent140, group multiple users per client to ensure class diversity
        # Check if this is likely Sent140 (many users with few samples each)
        avg_samples = np.mean([count for _, count in user_sample_counts])
        
        if avg_samples < 10 and len(sorted_users) >= num_nodes * 20:
            # Sent140 case: group users per client for class diversity
            # Use enough users to get good class mix, but not too many
            users_per_node = min(100, max(20, len(sorted_users) // (num_nodes * 10)))
            total_users_to_use = min(len(sorted_users), num_nodes * users_per_node)
            
            train_partitions = [[] for _ in range(num_nodes)]
            test_partitions = [[] for _ in range(num_nodes)]
            
            # Distribute users round-robin starting with largest
            for i, user in enumerate(sorted_users[:total_users_to_use]):
                node_id = i % num_nodes
                train_partitions[node_id].extend(train_dataset.user_indices[user])
                test_partitions[node_id].extend(test_dataset.user_indices[user])
            
            # Show partition info
            partition_sizes = [len(partition) for partition in train_partitions]
            print(f"Grouped {total_users_to_use} users into {num_nodes} clients (~{users_per_node} users per client)")
            print(f"Final partition sizes: {partition_sizes}")
        else:
            # FEMNIST case: use top users with most samples
            selected_users = sorted_users[:num_nodes]
            train_partitions = [train_dataset.user_indices[user] for user in selected_users]
            test_partitions = [test_dataset.user_indices[user] for user in selected_users]
            
            # Show sample counts for selected users
            sample_counts = [len(train_dataset.user_indices[user]) for user in selected_users]
            print(f"Using {len(selected_users)} users directly as clients")
            print(f"Selected users sample counts: {sample_counts}")
    else:
        # Group multiple users per node (largest users get distributed first)
        train_partitions = [[] for _ in range(num_nodes)]
        test_partitions = [[] for _ in range(num_nodes)]
        
        for i, user in enumerate(sorted_users):
            node_id = i % num_nodes
            train_partitions[node_id].extend(train_dataset.user_indices[user])
            test_partitions[node_id].extend(test_dataset.user_indices[user])
        
        # Show final partition sizes
        partition_sizes = [len(partition) for partition in train_partitions]
        print(f"Grouped {len(sorted_users)} users into {num_nodes} clients")
        print(f"Final partition sizes: {partition_sizes}")
    
    return train_partitions, test_partitions


def load_leaf_dataset(dataset_name: str, data_path: str) -> Tuple[Dataset, Dataset, nn.Module, int, int]:
    """
    Load LEAF dataset and return train/test datasets, model, num_classes, input_size.
    
    Returns:
        train_dataset, test_dataset, model, num_classes, input_size
    """
    from torchvision import transforms as T
    
    dataset_name = dataset_name.lower()
    
    if dataset_name == "femnist":
        # Create transforms for FEMNIST (PIL Image -> Tensor)
        transform = T.Compose([
            T.ToTensor(),  # Convert PIL Image to tensor and scale to [0,1]
        ])
        
        train_ds = LEAFFEMNISTDataset(data_path, "train", transform=transform) 
        test_ds = LEAFFEMNISTDataset(data_path, "test", transform=transform)
        model = LEAFFEMNISTModel(num_classes=62)
        return train_ds, test_ds, model, 62, 28
        
    elif dataset_name == "sent140":
        train_ds = LEAFSent140Dataset(data_path, "train")
        test_ds = LEAFSent140Dataset(data_path, "test") 
        model = LEAFSent140Model(vocab_size=train_ds.vocab_size)
        return train_ds, test_ds, model, 2, 25  # Binary classification, seq_len=25
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Use 'femnist' or 'sent140'")