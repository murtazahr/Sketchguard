# Data Extraction Scripts

These scripts extract experimental data from federated learning log files.

## Scripts

### extract_accuracies.py
Extracts final accuracy metrics from experiment log files.

**Usage:**
```bash
# From project root
python scripts/data_extraction/extract_accuracies.py

# From this directory
python extract_accuracies.py
```

**Input:** Log files in `../../results/` directory  
**Output:** `../../extracted_accuracies.csv`

**Extracted Data:**
- Experiment parameters (dataset, algorithm, graph topology, attack type)
- Final honest node accuracy
- Final compromised node accuracy  
- Attack percentage and other configuration

### extract_timing_data.py
Extracts timing performance data from experiment log files for complexity analysis.

**Usage:**
```bash
# From project root  
python scripts/data_extraction/extract_timing_data.py

# From this directory
python extract_timing_data.py
```

**Input:** Log files in `../../results/` directory  
**Output:** `../../timing_performance_data.csv`

**Extracted Data:**
- All experiment parameters
- Component timing breakdowns:
  - BALANCE: distance_time, filtering_time, aggregation_time
  - UBAR: distance_time, loss_time, aggregation_time  
  - SKETCHGUARD: sketching_time, filtering_time, aggregation_time
- Total execution time
- Model dimensions and compression ratios

## File Format Requirements

Log files should be named following the pattern:
```
{dataset}_{nodes}_{rounds}_{local_epochs}_{graph_type}_{graph_param}_{batch}_{samples}_{algorithm}_{attack}_{type}_{lambda}.log
```

Examples:
- `femnist_20_10_3_ring_64_10000_balance_40attack_gaussian_1lambda.log`
- `femnist_20_10_3_erdos_02_64_10000_sketchguard_30attack_1lambda.log`