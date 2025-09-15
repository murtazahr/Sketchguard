# Analysis Scripts

This directory contains scripts for data extraction and visualization of federated learning experiment results.

## Directory Structure

```
scripts/
├── data_extraction/     # Scripts for extracting data from log files
│   ├── extract_accuracies.py      # Extract accuracy metrics from experiment logs
│   └── extract_timing_data.py     # Extract timing performance data
└── visualization/       # Scripts for creating publication-quality visualizations  
    ├── visualize_paper_style.py           # Main accuracy comparison figures
    ├── visualize_topology_style.py        # Topology-based comparison figures
    ├── visualize_timing_performance.py    # Overall timing analysis
    ├── visualize_scaling_analysis.py      # Percentage scaling analysis
    └── visualize_computational_complexity.py  # Core complexity scaling
```

## Usage

### Data Extraction

1. **Extract accuracy data** (run from project root or scripts/data_extraction):
   ```bash
   python scripts/data_extraction/extract_accuracies.py
   ```
   - Processes all `.log` files in `results/` directory
   - Outputs: `extracted_accuracies.csv`

2. **Extract timing data** (run from project root or scripts/data_extraction):
   ```bash
   python scripts/data_extraction/extract_timing_data.py
   ```
   - Processes timing information from log files
   - Outputs: `timing_performance_data.csv`

### Visualization

All visualization scripts can be run from the project root or from their directory:

1. **Main paper figures**:
   ```bash
   python scripts/visualization/visualize_paper_style.py
   ```

2. **Topology comparison**:
   ```bash
   python scripts/visualization/visualize_topology_style.py
   ```

3. **Timing performance analysis**:
   ```bash
   python scripts/visualization/visualize_timing_performance.py
   ```

4. **Scaling analysis**:
   ```bash
   python scripts/visualization/visualize_scaling_analysis.py
   ```

5. **Computational complexity**:
   ```bash
   python scripts/visualization/visualize_computational_complexity.py
   ```

## Output Files

All generated files are saved to the project root directory:
- PDF and PNG figures for publications
- CSV files with extracted data

## Dependencies

- pandas
- matplotlib
- numpy
- seaborn (for some visualizations)