# SketchGuard: Scaling Byzantine-Robust Decentralized Federated Learning via Sketch-Based Screening

## Artifact Evaluation Documentation

This repository contains the implementation and experimental code for our WWW 2026 paper "SketchGuard: Scaling Byzantine-Robust Decentralized Federated Learning via Sketch-Based Screening." The codebase implements **SketchGuard**, a novel framework that uses Count Sketch compression for Byzantine-robust neighbor screening, compared against established baselines including D-FedAvg, Krum, BALANCE, and UBAR.

## Overview

SketchGuard addresses the fundamental scalability bottleneck in Byzantine-robust decentralized federated learning by decoupling Byzantine filtering from model aggregation through sketch-based neighbor screening. The framework reduces communication complexity from O(d|𝒩ᵢ|) to O(k|𝒩ᵢ| + d|𝒮ᵢ|) while maintaining identical robustness guarantees.

**Key Features:**
- **SketchGuard Framework**: Count Sketch-based compression for Byzantine filtering
- **Baseline Algorithms**: D-FedAvg, Krum, BALANCE, UBAR for comparison
- **Datasets**: FEMNIST (6.6M parameters), CelebA (2.2M parameters)
- **Network Topologies**: Ring, Fully-connected, Erdős-Rényi (p=0.2,0.45,0.6)
- **Attack Models**: Directed deviation, Gaussian noise injection
- **Scalability Analysis**: Networks up to 300 nodes, models up to 60M parameters

## Repository Structure

```
edgedrift/
├── decentralized_fl_sim.py     # Main simulation script
├── leaf_datasets.py            # LEAF dataset handling
├── model_variants.py           # Different model architectures
├── scripts/
│   └── visualization/          # Plotting and analysis scripts
├── raw_data/                   # Extracted experimental results
├── figures/                    # Generated plots and figures
└── README.md                   # This file
```

## Dependencies

The code requires Python 3.8+ with the following packages:

```bash
# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Data processing
pillow>=8.0.0
scikit-learn>=0.24.0
```

Install dependencies:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn pillow scikit-learn
```

## Quick Start

### Basic Usage

Run a simple experiment with SketchGuard on FEMNIST:
```bash
python decentralized_fl_sim.py \
    --dataset femnist \
    --num-nodes 20 \
    --rounds 10 \
    --agg sketchguard \
    --graph erdos \
    --p 0.2 \
    --sketchguard-sketch-size 1000
```

### With Byzantine Attacks

Simulate 30% Byzantine attackers using directed deviation:
```bash
python decentralized_fl_sim.py \
    --dataset femnist \
    --num-nodes 20 \
    --rounds 10 \
    --agg sketchguard \
    --attack-percentage 0.3 \
    --attack-type directed_deviation \
    --attack-lambda 1.0 \
    --sketchguard-sketch-size 1000
```

## Algorithm Parameters

### SketchGuard (Our main contribution)
- `--sketchguard-sketch-size`: Sketch size k for compression
- `--balance-gamma`: Distance threshold
- `--balance-kappa`: Decay parameter
- `--balance-alpha`: Mixing parameter

### BALANCE (Baseline)
- `--balance-gamma`: Distance threshold
- `--balance-kappa`: Decay parameter
- `--balance-alpha`: Mixing parameter

### UBAR (Baseline)
- `--ubar-rho`: Ratio of benign neighbors

## Dataset Setup

The simulator uses LEAF datasets (FEMNIST and CelebA). Datasets are automatically downloaded and preprocessed on first use. Ensure you have sufficient disk space (~2GB for FEMNIST, ~5GB for CelebA).

For manual dataset preparation:
1. Clone the LEAF repository
2. Follow LEAF preprocessing instructions
3. Update dataset paths in `leaf_datasets.py` if needed

## Expected Results

### Performance Metrics
- **Accuracy**: Final test accuracy after training
- **Convergence**: Rounds to reach target accuracy
- **Robustness**: Performance degradation under attacks

## Computational Requirements

### System Requirements
- **CPU**: Multi-core recommended (experiments use parallel training)
- **Memory**: 8GB+ RAM for larger experiments
- **Storage**: 10GB+ for datasets and results
- **Time**: Basic experiments run in 10-30 minutes; full reproduction may take several hours

### Resource Scaling
- Memory usage scales with: number of nodes × model size
- Runtime scales with: rounds × local epochs × dataset size
- Use `--max-samples` to limit dataset size for faster testing

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: The code defaults to CPU. For GPU, ensure PyTorch CUDA installation
2. **Memory Errors**: Reduce `--num-nodes`, `--max-samples`, or use smaller model variants
3. **Dataset Download**: Ensure internet connection for initial LEAF dataset download
4. **Import Errors**: Verify all dependencies are installed with correct versions

### Debug Mode
Use `--verbose` flag for detailed logging:
```bash
python decentralized_fl_sim.py --dataset femnist --verbose [other args]
```

## Contact

For questions about the artifact or reproduction issues, please contact: 
* Murtaza Rangwala: rangwalam@unimelb.edu.au
---

*Last updated: September 2024*