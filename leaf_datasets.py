#!/usr/bin/env python3
"""
LEAF dataset loaders and model architectures for federated learning research.
Implements PyTorch versions of LEAF's standard models and data loading.
"""

import json
import os
import sys
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Tuple, Optional
from collections import Counter

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


class LEAFCelebADataset(Dataset):
    """LEAF CelebA Dataset - Celebrity face attributes classification."""
    
    def __init__(self, data_path: str, split: str = "train", image_size: int = 84, transform=None):
        self.split = split
        self.image_size = image_size
        self.transform = transform
        # Construct path to images directory
        self.images_dir = os.path.join(data_path, 'raw', 'img_align_celeba')
        print(f"Looking for images in: {os.path.abspath(self.images_dir)}")
        
        # Load LEAF JSON data from multiple files (similar to FEMNIST)
        split_dir = os.path.join(data_path, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"LEAF CelebA {split} directory not found at {split_dir}")
            
        json_files = [f for f in os.listdir(split_dir) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {split_dir}")
        
        print(f"Loading {len(json_files)} LEAF CelebA {split} files...")
        
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
        
        
        # Flatten all data
        self.all_data = []  # Image filenames
        self.all_targets = []  # Binary labels (0 or 1)
        self.user_indices = {}
        
        current_idx = 0
        for user in self.users:
            user_x = self.user_data[user]['x']  # Image filenames
            user_y = self.user_data[user]['y']  # Binary labels
            
            start_idx = current_idx
            self.all_data.extend(user_x)
            self.all_targets.extend(user_y)
            end_idx = current_idx + len(user_x)
            
            self.user_indices[user] = list(range(start_idx, end_idx))
            current_idx = end_idx
            
        print(f"LEAF CelebA {split}: {len(self.users)} users (celebrities), {len(self.all_data)} samples")
    
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        # Load image from filename
        img_name = self.all_data[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Check if image exists and provide better error handling
        if not os.path.exists(img_path):
            # Try alternative paths
            alt_paths = [
                os.path.join(os.path.dirname(self.images_dir), img_name),  # raw/img_name
                os.path.join(os.path.dirname(os.path.dirname(self.images_dir)), 'raw', img_name),  # ../raw/img_name
            ]
            
            found = False
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    img_path = alt_path
                    found = True
                    break
            
            if not found:
                raise FileNotFoundError(f"Could not find image {img_name}. Tried paths:\n" +
                                      f"  {img_path}\n" + 
                                      "\n".join(f"  {p}" for p in alt_paths))
        
        img = Image.open(img_path)
        img = img.resize((self.image_size, self.image_size)).convert('RGB')
        
        target = int(self.all_targets[idx])  # Binary classification (0 or 1)
        
        if self.transform:
            img = self.transform(img)
        else:
            # Default transform to tensor
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.tensor(img).permute(2, 0, 1)  # HWC -> CHW
        
        return img, target
    
    def get_user_data(self, user: str) -> List[int]:
        """Get indices for a specific user's data."""
        return self.user_indices.get(user, [])


# ---------------------- Sent140 Text Utilities ---------------------- #

def split_line(line: str) -> List[str]:
    """Split text into words and punctuation tokens."""
    return re.findall(r"[\w']+|[.,!?;]", line.lower())


def build_vocabulary(texts: List[str], max_vocab_size: int = 10000, min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from list of texts.

    Args:
        texts: List of text strings
        max_vocab_size: Maximum vocabulary size
        min_freq: Minimum word frequency to include

    Returns:
        Dictionary mapping words to indices
    """
    word_counts = Counter()
    for text in texts:
        words = split_line(text)
        word_counts.update(words)

    # Filter by frequency and take top words
    filtered_words = [(word, count) for word, count in word_counts.items() if count >= min_freq]
    filtered_words.sort(key=lambda x: -x[1])  # Sort by count descending
    top_words = [word for word, _ in filtered_words[:max_vocab_size]]

    # Create word to index mapping (0 reserved for padding, vocab_size for unknown)
    word2idx = {word: idx + 1 for idx, word in enumerate(top_words)}
    return word2idx


def text_to_indices(text: str, word2idx: Dict[str, int], max_len: int = 25) -> List[int]:
    """Convert text to list of word indices.

    Args:
        text: Input text string
        word2idx: Word to index mapping
        max_len: Maximum sequence length

    Returns:
        List of word indices, padded or truncated to max_len
    """
    unk_idx = len(word2idx) + 1  # Unknown word index
    words = split_line(text)
    indices = [word2idx.get(w, unk_idx) for w in words[:max_len]]
    # Pad with zeros if needed
    indices += [0] * (max_len - len(indices))
    return indices


class LEAFSent140Dataset(Dataset):
    """LEAF Sent140 Dataset - Twitter sentiment classification (binary: negative/positive)."""

    def __init__(self, data_path: str, split: str = "train", max_seq_len: int = 25,
                 vocab: Optional[Dict[str, int]] = None, max_vocab_size: int = 10000):
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_vocab_size = max_vocab_size

        # Load LEAF JSON data
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

            self.users.extend(data['users'])
            self.user_data.update(data['user_data'])
            self.num_samples.extend(data['num_samples'])

        # Flatten all data
        self.all_texts = []  # Raw text strings
        self.all_targets = []  # Sentiment labels (0=negative, 1=positive)
        self.user_indices = {}

        current_idx = 0
        for user in self.users:
            user_x = self.user_data[user]['x']  # List of [id, date, query, user, text]
            user_y = self.user_data[user]['y']  # Sentiment labels

            start_idx = current_idx
            # Extract just the tweet text (index 4 in raw data)
            for sample in user_x:
                if isinstance(sample, list) and len(sample) >= 5:
                    self.all_texts.append(sample[4])  # Tweet text is at index 4
                else:
                    self.all_texts.append(str(sample))  # Fallback
            self.all_targets.extend(user_y)
            end_idx = current_idx + len(user_x)

            self.user_indices[user] = list(range(start_idx, end_idx))
            current_idx = end_idx

        print(f"LEAF Sent140 {split}: {len(self.users)} users, {len(self.all_texts)} samples")

        # Build or use provided vocabulary
        if vocab is not None:
            self.word2idx = vocab
        else:
            print(f"Building vocabulary (max_size={max_vocab_size})...")
            self.word2idx = build_vocabulary(self.all_texts, max_vocab_size=max_vocab_size)
            print(f"Vocabulary size: {len(self.word2idx)}")

        self.vocab_size = len(self.word2idx) + 2  # +1 for padding, +1 for unknown

        # Precompute indices for all texts
        self.all_data = [text_to_indices(text, self.word2idx, max_seq_len)
                         for text in self.all_texts]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        indices = torch.tensor(self.all_data[idx], dtype=torch.long)
        target = int(self.all_targets[idx])
        return indices, target

    def get_user_data(self, user: str) -> List[int]:
        """Get indices for a specific user's data."""
        return self.user_indices.get(user, [])

    def get_vocabulary(self) -> Dict[str, int]:
        """Return the vocabulary for sharing with test dataset."""
        return self.word2idx


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


class LEAFCelebAModel(nn.Module):
    """Simple LeNet-style CelebA CNN Model for stable training."""

    def __init__(self, num_classes: int = 2, image_size: int = 84):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        # Simple LeNet-style architecture adapted for CelebA
        # Input: 84×84×3 (vs original 28×28×1)

        # First convolution: 3×3×30 (adapted from 3×3×30)
        self.conv1 = nn.Conv2d(3, 30, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolution: 3×3×50 (same as original)
        self.conv2 = nn.Conv2d(30, 50, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Calculate size after pooling
        # 84 → 42 (after pool1) → 21 (after pool2)
        conv_output_size = 21 * 21 * 50

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 100)  # Hidden layer with 100 units
        self.fc2 = nn.Linear(100, num_classes)       # Output layer

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with small random values for stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (batch, 3, 84, 84)

        # First conv + pool
        x = self.conv1(x)           # → (batch, 30, 84, 84)
        x = torch.relu(x)
        x = self.pool1(x)           # → (batch, 30, 42, 42)

        # Second conv + pool
        x = self.conv2(x)           # → (batch, 50, 42, 42)
        x = torch.relu(x)
        x = self.pool2(x)           # → (batch, 50, 21, 21)

        # Flatten
        x = x.view(x.size(0), -1)   # → (batch, 50*21*21)

        # Fully connected layers
        x = self.fc1(x)             # → (batch, 100)
        x = torch.relu(x)
        x = self.fc2(x)             # → (batch, num_classes)

        return x


class LEAFSent140Model(nn.Module):
    """LEAF Sent140 Stacked LSTM Model - PyTorch version of LEAF's TensorFlow model.

    Architecture matches LEAF reference: embedding -> 2-layer LSTM -> FC -> output
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 100,
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Stacked LSTM (2 layers as in LEAF reference)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # x: (batch, seq_len) - word indices

        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch, seq_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)

        # Take the last hidden state from the top layer
        last_hidden = hidden[-1]  # (batch, hidden_dim)

        # Fully connected layers
        x = self.dropout(last_hidden)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def create_leaf_client_partitions(train_dataset, test_dataset, num_nodes: int, seed: int = 42) -> Tuple[List[List[int]], List[List[int]]]:
    """Create client partitions using LEAF's natural user groupings.
    
    Args:
        train_dataset: Training dataset with user_indices
        test_dataset: Test dataset with user_indices
        num_nodes: Number of clients/nodes to partition data across
        seed: Random seed for reproducible partitioning
    
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
    
    # Sort users deterministically for reproducibility
    common_users.sort()  # Sort alphabetically first for consistency
    
    # Sort users by training sample count (largest first)
    user_sample_counts = []
    for user in common_users:
        train_samples = len(train_dataset.user_indices[user])
        user_sample_counts.append((user, train_samples))
    
    # Sort by sample count in descending order, with secondary sort by user ID for determinism
    user_sample_counts.sort(key=lambda x: (-x[1], x[0]))  # Negative for descending, then by user ID
    sorted_users = [user for user, _ in user_sample_counts]
    
    # Shuffle users with the provided seed for different random assignments
    rng.shuffle(sorted_users)
    
    print(f"Found {len(train_users)} train users, {len(test_users)} test users, {len(common_users)} common users")
    print(f"User sample counts range: {user_sample_counts[0][1]} (max) to {user_sample_counts[-1][1]} (min)")
    
    # Always distribute ALL users equally across nodes (no min/max limits)
    train_partitions = [[] for _ in range(num_nodes)]
    test_partitions = [[] for _ in range(num_nodes)]
    client_class_counts = [{} for _ in range(num_nodes)]
    
    # Distribute users round-robin (deterministic based on seed)
    for i, user in enumerate(sorted_users):
        node_id = i % num_nodes  # Round-robin assignment
        user_train_indices = train_dataset.user_indices[user]
        train_partitions[node_id].extend(user_train_indices)
        test_partitions[node_id].extend(test_dataset.user_indices[user])
        
        # Track classes for this user
        for idx in user_train_indices:
            label = train_dataset.all_targets[idx]
            if label not in client_class_counts[node_id]:
                client_class_counts[node_id][label] = 0
            client_class_counts[node_id][label] += 1
    
    # Show partition info
    users_per_node = len(sorted_users) // num_nodes
    remainder_users = len(sorted_users) % num_nodes
    partition_sizes = [len(partition) for partition in train_partitions]
    test_partition_sizes = [len(partition) for partition in test_partitions]
    
    print(f"Distributed ALL {len(sorted_users)} users across {num_nodes} clients")
    print(f"Users per client: {users_per_node} (with {remainder_users} clients getting +1 user)")
    print(f"Train partition sizes: {partition_sizes}")
    print(f"Test partition sizes: {test_partition_sizes}")
    
    # Show class distribution for each client
    for i in range(num_nodes):
        unique_classes = len(client_class_counts[i])
        total_samples = sum(client_class_counts[i].values())
        print(f"  Client {i}: {total_samples} train samples, {unique_classes} unique classes")
    
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
        
    elif dataset_name == "celeba":
        # Create transforms for CelebA
        transform = T.Compose([
            T.ToTensor(),  # Convert PIL Image to tensor and scale to [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        train_ds = LEAFCelebADataset(data_path, "train", image_size=84, transform=transform)
        test_ds = LEAFCelebADataset(data_path, "test", image_size=84, transform=transform)
        model = LEAFCelebAModel(num_classes=2, image_size=84)  # Binary classification (smiling or not)
        return train_ds, test_ds, model, 2, 84

    elif dataset_name == "sent140":
        # Load Sent140 - build vocabulary from training data
        train_ds = LEAFSent140Dataset(data_path, "train", max_seq_len=25, max_vocab_size=10000)
        # Share vocabulary with test dataset
        test_ds = LEAFSent140Dataset(data_path, "test", max_seq_len=25, vocab=train_ds.get_vocabulary())
        model = LEAFSent140Model(vocab_size=train_ds.vocab_size, num_classes=2)
        return train_ds, test_ds, model, 2, 25  # 25 is max sequence length

    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Use 'femnist', 'celeba', or 'sent140'")