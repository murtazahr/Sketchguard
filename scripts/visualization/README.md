# Visualization Scripts

These scripts create publication-quality figures from extracted experimental data.

## Scripts

### visualize_paper_style.py
Creates the main paper figures comparing algorithm performance.

**Usage:**
```bash
python visualize_paper_style.py
```

**Input:** `../../extracted_accuracies.csv`  
**Output:** 
- `../../femnist_paper_style.pdf/png`

**Features:**
- Two-panel figure: (a) Directed Deviation, (b) Gaussian attacks
- Inset zoom for overlapping algorithms (BALANCE, UBAR, SKETCHGUARD)  
- Publication-style formatting with legend at top

### visualize_topology_style.py
Creates topology comparison figures showing performance across different graph structures.

**Usage:**
```bash
python visualize_topology_style.py
```

**Input:** `../../extracted_accuracies.csv`  
**Output:**
- `../../femnist_topology_comparison.pdf/png`

**Features:**
- Five panels: Ring, Erdős p=0.2/0.45/0.6, Fully Connected
- Shows how algorithms perform across network topologies
- Individual insets for each topology

### visualize_timing_performance.py  
Creates overall timing comparison and component breakdown analysis.

**Usage:**
```bash
python visualize_timing_performance.py
```

**Input:** `../../timing_performance_data.csv`  
**Output:**
- `../../timing_comparison.pdf/png`

**Features:**
- Box plots of total execution times
- Stacked bar charts showing timing component breakdown
- Statistical analysis table

### visualize_scaling_analysis.py
Creates percentage scaling analysis for theoretical complexity validation.

**Usage:**
```bash
python visualize_scaling_analysis.py
```

**Input:** `../../timing_performance_data.csv`  
**Output:**
- `../../network_topology_scaling.pdf/png`
- `../../malicious_concentration_scaling.pdf/png`

**Features:**
- **Scenario 1:** Network topology scaling (increasing neighbors)
- **Scenario 2:** Malicious concentration scaling (10% → 80% malicious nodes)
- Percentage change analysis normalized to baseline

### visualize_computational_complexity.py
Focuses on pure computational complexity by isolating filtering/screening components.

**Usage:**
```bash
python visualize_computational_complexity.py
```

**Input:** `../../timing_performance_data.csv`  
**Output:**
- `../../computational_complexity_scaling.pdf/png`

**Features:**
- Isolates core algorithmic differences:
  - BALANCE: distance + filtering time  
  - UBAR: distance + loss computation time
  - SKETCHGUARD: sketching + filtering time
- Excludes common O(d×|S_i^t|) aggregation overhead
- Shows theoretical complexity advantage empirically

## Algorithm Name Mapping

All visualization scripts map internal names to publication names:
- `balance` → `BALANCE`
- `ubar` → `UBAR` 
- `coarse` → `SKETCHGUARD`

## Output Format

All scripts generate both PDF (for publications) and PNG (for presentations) versions with:
- High DPI (300) for publication quality
- Consistent color schemes and styling
- Publication-ready fonts (Times New Roman/serif)