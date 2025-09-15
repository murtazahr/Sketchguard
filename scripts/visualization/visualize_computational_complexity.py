#!/usr/bin/env python3
"""
Visualize computational complexity scaling by isolating the filtering/screening components.
This focuses purely on the theoretical complexity differences without aggregation overhead.

Components analyzed:
- BALANCE: distance_time + filtering_time (O(N×d))
- UBAR: distance_time + loss_time (O(N×d + N×inference))  
- SKETCHGUARD: sketching_time + filtering_time (O(d + k×|Ni|))
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.6,
    'lines.linewidth': 3.0,
    'lines.markersize': 10,
})

def get_data_path(filename):
    """Get the correct path to data files relative to script location."""
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, '..', '..', filename)

def get_output_path(filename):
    """Get the correct path for output files."""
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, '..', '..', filename)

def load_timing_data(csv_file='timing_performance_data.csv'):
    """Load timing performance data from CSV."""
    csv_path = get_data_path(csv_file)
    df = pd.read_csv(csv_path)
    
    # Convert algorithm names for display
    df['algorithm_display'] = df['algorithm'].map({
        'balance': 'BALANCE',
        'ubar': 'UBAR', 
        'coarse': 'SKETCHGUARD'
    })
    
    return df

def calculate_computational_complexity_time(df):
    """Calculate computational complexity time for each algorithm."""
    
    # Add computational complexity columns
    df['complexity_time'] = np.nan
    
    for idx, row in df.iterrows():
        if row['algorithm'] == 'balance':
            # BALANCE: distance + filtering time (O(N×d))
            if pd.notna(row['distance_time']) and pd.notna(row['filtering_time']):
                df.at[idx, 'complexity_time'] = row['distance_time'] + row['filtering_time']
        
        elif row['algorithm'] == 'ubar':
            # UBAR: distance + loss computation time (O(N×d + N×inference))
            if pd.notna(row['distance_time']) and pd.notna(row['loss_time']):
                df.at[idx, 'complexity_time'] = row['distance_time'] + row['loss_time']
        
        elif row['algorithm'] == 'coarse':
            # SKETCHGUARD: sketching + filtering time (O(d + k×|Ni|))
            if pd.notna(row['sketching_time']) and pd.notna(row['filtering_time']):
                df.at[idx, 'complexity_time'] = row['sketching_time'] + row['filtering_time']
    
    return df

def estimate_neighbor_count(graph_type, graph_param, num_nodes=20):
    """Estimate average number of neighbors for each topology."""
    if graph_type == 'ring':
        return 2  # Each node connects to 2 neighbors in a ring
    elif graph_type == 'fully':
        return num_nodes - 1  # Each node connects to all others
    elif graph_type == 'erdos':
        # Expected degree in Erdős-Rényi graph: p*(n-1)
        p = graph_param
        return p * (num_nodes - 1)
    else:
        return 0

def create_computational_complexity_scaling(df, save_prefix=''):
    """
    Create computational complexity scaling analysis focusing on filtering/screening components only.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate computational complexity times
    df = calculate_computational_complexity_time(df)
    
    # Define topology progression (increasing neighbor count)
    topology_progression = [
        ('ring', None, 'Ring'),
        ('erdos', 0.2, 'Erdős p=0.2'),
        ('erdos', 0.45, 'Erdős p=0.45'),
        ('erdos', 0.6, 'Erdős p=0.6'),
        ('fully', None, 'Fully Connected')
    ]
    
    # Colors for algorithms
    colors = {
        'BALANCE': '#2E8B57',      # Sea Green
        'UBAR': '#4169E1',         # Royal Blue  
        'SKETCHGUARD': '#DC143C'   # Crimson
    }
    
    markers = {
        'BALANCE': 'o',
        'UBAR': 's', 
        'SKETCHGUARD': '^'
    }
    
    algorithms = ['BALANCE', 'UBAR', 'SKETCHGUARD']
    
    # Prepare data for each algorithm
    topology_data = {}
    neighbor_counts = []
    
    for graph_type, graph_param, display_name in topology_progression:
        # Filter data for this topology
        if graph_type == 'erdos':
            topo_df = df[(df['graph_type'] == graph_type) & (df['graph_param'] == graph_param)]
        else:
            topo_df = df[df['graph_type'] == graph_type]
        
        # Calculate average neighbor count for this topology
        avg_neighbors = estimate_neighbor_count(graph_type, graph_param)
        neighbor_counts.append(avg_neighbors)
        
        topology_data[display_name] = {}
        topology_data[display_name]['neighbors'] = avg_neighbors
        
        for algo in algorithms:
            algo_df = topo_df[topo_df['algorithm_display'] == algo]
            if len(algo_df) > 0:
                # Use median time to reduce outlier impact
                complexity_times = algo_df['complexity_time'].dropna()
                if len(complexity_times) > 0:
                    topology_data[display_name][algo] = complexity_times.median()
                else:
                    topology_data[display_name][algo] = None
            else:
                topology_data[display_name][algo] = None
    
    # Plot 1: Absolute computational complexity timing
    x_labels = [topo[2] for topo in topology_progression]
    x_pos = np.arange(len(x_labels))
    
    for algo in algorithms:
        times = []
        for display_name in [topo[2] for topo in topology_progression]:
            time_val = topology_data[display_name].get(algo)
            times.append(time_val if time_val is not None else 0)
        
        # Only plot if we have data
        if any(t > 0 for t in times):
            ax1.plot(x_pos, times, marker=markers[algo], color=colors[algo], 
                    label=algo, linewidth=3.0, markersize=10, markeredgewidth=2, 
                    markeredgecolor='white')
    
    ax1.set_xlabel('Network Topology (Increasing Connectivity)', fontweight='bold')
    ax1.set_ylabel('Computational Complexity Time (seconds)', fontweight='bold')
    ax1.set_title('(a) Absolute Computational Complexity', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=20, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add neighbor count annotations
    for i, neighbors in enumerate(neighbor_counts):
        ax1.text(i, ax1.get_ylim()[1] * 0.95, f'{neighbors:.1f}\nneighbors', 
                ha='center', va='top', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Percentage scaling (normalized to ring topology)
    for algo in algorithms:
        times = []
        ring_time = None
        
        for display_name in [topo[2] for topo in topology_progression]:
            time_val = topology_data[display_name].get(algo)
            if time_val is not None:
                if ring_time is None:  # First non-null value (ring)
                    ring_time = time_val
                times.append(time_val)
            else:
                times.append(None)
        
        if ring_time is not None and ring_time > 0:
            # Calculate percentage increase from ring topology
            percent_increases = []
            for time_val in times:
                if time_val is not None:
                    percent_increase = ((time_val - ring_time) / ring_time) * 100
                    percent_increases.append(percent_increase)
                else:
                    percent_increases.append(None)
            
            # Filter out None values for plotting
            valid_indices = [i for i, val in enumerate(percent_increases) if val is not None]
            valid_percentages = [percent_increases[i] for i in valid_indices]
            valid_x_pos = [x_pos[i] for i in valid_indices]
            
            if valid_percentages:
                ax2.plot(valid_x_pos, valid_percentages, marker=markers[algo], 
                        color=colors[algo], label=algo, linewidth=3.0, markersize=10,
                        markeredgewidth=2, markeredgecolor='white')
    
    ax2.set_xlabel('Network Topology (Increasing Connectivity)', fontweight='bold')
    ax2.set_ylabel('Computational Complexity Increase from Ring (%)', fontweight='bold')
    ax2.set_title('(b) Percentage Scaling of Core Complexity', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=20, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add theoretical complexity annotations
    ax2.text(0.02, 0.98, 
            'Isolated Computational Complexity:\n'
            'BALANCE: O(N×d) - distance + filtering\n'
            'UBAR: O(N×d + N×inference) - distance + loss\n'
            'SKETCHGUARD: O(d + k×|N_i|) - sketching + filtering\n'
            '\n'
            'Excludes O(d×|S_i^t|) aggregation (common to all)',
            transform=ax2.transAxes, fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    
    plt.tight_layout()
    
    # Save figure
    filename = get_output_path(f'{save_prefix}computational_complexity_scaling.pdf')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {filename}")
    
    png_filename = get_output_path(f'{save_prefix}computational_complexity_scaling.png')
    plt.savefig(png_filename, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {png_filename}")
    
    plt.show()
    plt.close()
    
    return topology_data

def create_complexity_breakdown_table(df, topology_data):
    """Create detailed breakdown of computational complexity components."""
    
    print("\n" + "="*120)
    print("COMPUTATIONAL COMPLEXITY SCALING ANALYSIS")
    print("="*120)
    
    print("\nCOMPUTATIONAL COMPONENTS ANALYZED:")
    print("  BALANCE: distance_time + filtering_time (O(N×d) complexity)")
    print("  UBAR: distance_time + loss_time (O(N×d + N×inference) complexity)")
    print("  SKETCHGUARD: sketching_time + filtering_time (O(d + k×|N_i|) complexity)")
    print("\nNote: Excludes aggregation_time (O(d×|S_i^t|)) which is common to all algorithms")
    
    print(f"\nNETWORK TOPOLOGY SCALING (Computational Complexity Only):")
    print("  Topology".ljust(18) + "Neighbors".ljust(12) + "BALANCE".ljust(15) + "UBAR".ljust(15) + "SKETCHGUARD")
    print("-" * 75)
    
    topologies = ['Ring', 'Erdős p=0.2', 'Erdős p=0.45', 'Erdős p=0.6', 'Fully Connected']
    ring_times = {}
    
    for topo in topologies:
        if topo in topology_data:
            neighbors = topology_data[topo]['neighbors']
            balance_time = topology_data[topo].get('BALANCE', 0)
            ubar_time = topology_data[topo].get('UBAR', 0) 
            coarse_time = topology_data[topo].get('SKETCHGUARD', 0)
            
            if topo == 'Ring':
                ring_times = {'BALANCE': balance_time, 'UBAR': ubar_time, 'SKETCHGUARD': coarse_time}
                print(f"  {topo:<17} {neighbors:>8.1f}     {balance_time:>10.3f}s      {ubar_time:>10.3f}s      {coarse_time:>10.3f}s")
            else:
                balance_pct = ((balance_time - ring_times['BALANCE']) / ring_times['BALANCE'] * 100) if ring_times['BALANCE'] > 0 else 0
                ubar_pct = ((ubar_time - ring_times['UBAR']) / ring_times['UBAR'] * 100) if ring_times['UBAR'] > 0 else 0
                coarse_pct = ((coarse_time - ring_times['SKETCHGUARD']) / ring_times['SKETCHGUARD'] * 100) if ring_times['SKETCHGUARD'] > 0 else 0
                
                print(f"  {topo:<17} {neighbors:>8.1f}     {balance_pct:>9.1f}%       {ubar_pct:>9.1f}%       {coarse_pct:>9.1f}%")
    
    # Show component breakdown for fully connected (highest neighbor count)
    print(f"\nCOMPONENT BREAKDOWN (Fully Connected Topology):")
    fully_df = df[df['graph_type'] == 'fully']
    
    print("Algorithm".ljust(15) + "Sketching".ljust(12) + "Distance".ljust(12) + "Filtering".ljust(12) + "Loss".ljust(12) + "Total Complex.")
    print("-" * 75)
    
    for algo in ['BALANCE', 'UBAR', 'SKETCHGUARD']:
        algo_df = fully_df[fully_df['algorithm_display'] == algo]
        if len(algo_df) > 0:
            sketch_time = algo_df['sketching_time'].median() if 'sketching_time' in algo_df and pd.notna(algo_df['sketching_time']).any() else 0
            distance_time = algo_df['distance_time'].median() if 'distance_time' in algo_df and pd.notna(algo_df['distance_time']).any() else 0
            filtering_time = algo_df['filtering_time'].median() if 'filtering_time' in algo_df and pd.notna(algo_df['filtering_time']).any() else 0
            loss_time = algo_df['loss_time'].median() if 'loss_time' in algo_df and pd.notna(algo_df['loss_time']).any() else 0
            
            if algo == 'BALANCE':
                total_complex = distance_time + filtering_time
            elif algo == 'UBAR':
                total_complex = distance_time + loss_time
            else:  # SKETCHGUARD
                total_complex = sketch_time + filtering_time
            
            print(f"  {algo:<14} {sketch_time:>8.3f}s   {distance_time:>8.3f}s   {filtering_time:>8.3f}s   {loss_time:>8.3f}s   {total_complex:>8.3f}s")
    
    print(f"\nTHEORETICAL ANALYSIS:")
    model_dim = df['model_dimension'].iloc[0]
    sketch_size = df[df['algorithm'] == 'coarse']['sketch_size'].iloc[0] if len(df[df['algorithm'] == 'coarse']) > 0 else 1000
    
    print(f"  Model dimension (d): {model_dim:,}")
    print(f"  Sketch size (k): {sketch_size:,}")
    print(f"  Compression ratio (d/k): {model_dim/sketch_size:.1f}x")
    print(f"  Expected SKETCHGUARD advantage: O(d + k×|N_i|) vs O(N×d)")
    print(f"  At fully connected (19 neighbors): O({model_dim:,} + {sketch_size}×19) vs O(19×{model_dim:,})")
    
    theoretical_ratio = (19 * model_dim) / (model_dim + sketch_size * 19)
    print(f"  Theoretical speedup: {theoretical_ratio:.1f}x")

def main():
    """Generate computational complexity scaling analysis."""
    
    print("Loading timing performance data...")
    df = load_timing_data()
    
    print(f"Loaded {len(df)} timing records")
    print(f"Algorithms: {df['algorithm_display'].unique()}")
    
    print("\nGenerating computational complexity scaling analysis...")
    topology_data = create_computational_complexity_scaling(df)
    
    print("\nGenerating detailed complexity breakdown...")
    create_complexity_breakdown_table(df, topology_data)
    
    print("\n✅ Computational complexity analysis completed!")
    print("\nKey insights:")
    print("1. Isolated computational complexity (excluding aggregation)")
    print("2. Shows pure algorithmic scaling differences")
    print("3. SKETCHGUARD O(d + k×|N_i|) vs others O(N×d)")
    print("4. Theoretical advantage should be visible in network scaling")

if __name__ == "__main__":
    main()