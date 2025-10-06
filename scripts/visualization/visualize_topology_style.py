#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (12, 6),  # Wider for multiple panels
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 3,
})

def get_output_path(filename):
    """Get the correct path for output files."""
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, '..', '..', filename)

def load_and_prepare_data(csv_file='raw_data/extracted_accuracies.csv'):
    """Load CSV and prepare data for visualization."""
    # Get the correct path to data files relative to script location
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, '..', '..', csv_file)
    df = pd.read_csv(csv_path)
    df['attack_percentage'] = df['attack_percentage'].astype(int)
    df['honest_error_rate'] = 1 - df['final_honest_accuracy']
    df['compromised_error_rate'] = 1 - df['final_compromised_accuracy']
    return df

def create_topology_figure(df, save_prefix=''):
    """Create a multi-panel figure separated by graph topology, averaging across datasets."""

    # Get all topology combinations
    topology_combinations = [
        ('ring', 'NA', 'Ring'),
        ('erdos', '02', 'Erdos p=0.2'),
        ('erdos', '045', 'Erdos p=0.45'),
        ('erdos', '06', 'Erdos p=0.6'),
        ('fully', 'NA', 'Fully Connected')
    ]

    # Create figure with 5 subplots (1 row, 5 columns)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes = axes.flatten()  # Flatten for easier indexing

    node_type = 'honest'  # Focus on honest nodes
    attack_type = 'directed_deviation'  # Use directed deviation as the attack type

    # Define algorithms and their visual properties
    algorithms = ['d-fedavg', 'krum', 'balance', 'ubar', 'sketchguard']

    colors = {
        'd-fedavg': '#DC143C',  # Crimson red for FedAvg
        'krum': '#000000',      # Black
        'balance': '#0000FF',   # Blue
        'ubar': '#FF8C00',      # Orange
        'sketchguard': '#FF00FF'     # Magenta
    }

    line_styles = {
        'd-fedavg': '--',       # Dashed
        'krum': ':',            # Dotted
        'balance': '-.',        # Dash-dot
        'ubar': '-',            # Solid
        'sketchguard': (0, (1, 1))   # More dotted (1 pixel line, 1 pixel gap)
    }

    markers = {
        'd-fedavg': 'v',
        'krum': 'o',
        'balance': 's',
        'ubar': '^',
        'sketchguard': 'D'
    }

    # Create legend handles once for the entire figure
    legend_handles = []
    legend_labels = []

    for idx, (graph_type, graph_param, display_name) in enumerate(topology_combinations):
        ax = axes[idx]

        # Filter data for this specific topology (averaging across all datasets)
        if graph_param == 'NA':
            # For fully and ring topologies, graph_param is NaN
            df_filtered = df[
                (df['graph_type'] == graph_type) &
                (df['graph_param'].isna()) &
                (df['attack_type'] == attack_type)
            ].copy()
        else:
            # For erdos topology, graph_param has specific values
            df_filtered = df[
                (df['graph_type'] == graph_type) &
                (df['graph_param'] == int(graph_param)) &
                (df['attack_type'] == attack_type)
            ].copy()
        
        error_col = f'{node_type}_error_rate'
        
        # Check if we have data for this topology
        if len(df_filtered) == 0:
            ax.text(0.5, 0.5, f'No data for\n{display_name}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=8, style='italic')
            ax.set_xlim(0, 80)
            ax.set_ylim(0, 1.0)
            continue
        
        # First, find attack percentages that exist for BOTH datasets in this topology
        celeba_data = df_filtered[df_filtered['dataset'] == 'celeba']
        femnist_data = df_filtered[df_filtered['dataset'] == 'femnist']

        if len(celeba_data) > 0 and len(femnist_data) > 0:
            # Find attack percentage + algorithm combinations that exist in BOTH datasets
            celeba_combos = set(zip(celeba_data['attack_percentage'], celeba_data['algorithm']))
            femnist_combos = set(zip(femnist_data['attack_percentage'], femnist_data['algorithm']))
            common_combos = celeba_combos.intersection(femnist_combos)

            print(f"\n{display_name}:")
            print(f"  CelebA combos: {len(celeba_combos)}")
            print(f"  FEMNIST combos: {len(femnist_combos)}")
            print(f"  Common combos: {len(common_combos)}")

            # Extract just the attack percentages from common combinations
            common_attack_percentages = {combo[0] for combo in common_combos}
            print(f"  Common attack %: {sorted(common_attack_percentages)}")
        else:
            # If no CelebA data for this topology, skip it
            ax.text(0.5, 0.5, f'No CelebA data for\n{display_name}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=8, style='italic')
            ax.set_xlim(0, 80)
            ax.set_ylim(0, 1.0)
            continue

        # Plot lines for each algorithm
        for algo in algorithms:
            algo_data = df_filtered[df_filtered['algorithm'] == algo]

            if len(algo_data) > 0:
                # Filter to only include (attack_percentage, algorithm) combinations that exist in BOTH datasets
                algo_combos = [(row['attack_percentage'], row['algorithm']) for _, row in algo_data.iterrows()]
                valid_rows = [combo in common_combos for combo in algo_combos]
                algo_data = algo_data[valid_rows]

                if len(algo_data) > 0:
                    grouped = algo_data.groupby('attack_percentage')[error_col].mean().reset_index()
                    grouped = grouped.sort_values('attack_percentage')

                    # Set z-order: sketchguard (top), balance (middle), ubar (bottom), krum (separate), fedavg (back)
                    z_orders = {'sketchguard': 5, 'balance': 4, 'ubar': 3, 'krum': 2, 'd-fedavg': 1}

                    # Map algorithm names for display
                    if algo == 'sketchguard':
                        algo_display_name = 'SKETCHGUARD'
                    elif algo == 'd-fedavg':
                        algo_display_name = 'FEDAVG'
                    else:
                        algo_display_name = algo.upper()

                    line, = ax.plot(grouped['attack_percentage'],
                                   grouped[error_col],
                                   label=algo_display_name,
                                   linestyle=line_styles[algo],
                                   color=colors[algo],
                                   marker=markers[algo],
                                   markersize=3,
                                   linewidth=1.5,
                                   markeredgewidth=0.4,
                                   markeredgecolor=colors[algo],
                                   markevery=2,
                                   zorder=z_orders[algo])

                    # Collect legend handles only once (from first subplot with data)
                    if len(legend_handles) < len(algorithms):
                        legend_handles.append(line)
                        legend_labels.append(algo_display_name)
        
        # Create inset for overlapping algorithms (if we have data)
        if len(df_filtered) > 0:
            # Place inset in lower right for better visibility
            axins = inset_axes(ax, width="30%", height="30%",
                              loc='lower right',
                              bbox_to_anchor=(0, 0.25, 1, 1),
                              bbox_transform=ax.transAxes)
            
            inset_algorithms = ['balance', 'ubar', 'sketchguard']
            
            # Calculate y-limits for inset
            y_values = []
            for algo in inset_algorithms:
                algo_data = df_filtered[df_filtered['algorithm'] == algo]
                if len(algo_data) > 0:
                    # Filter to only include (attack_percentage, algorithm) combinations that exist in BOTH datasets
                    algo_combos = [(row['attack_percentage'], row['algorithm']) for _, row in algo_data.iterrows()]
                    valid_rows = [combo in common_combos for combo in algo_combos]
                    algo_data = algo_data[valid_rows]
                    if len(algo_data) > 0:
                        grouped = algo_data.groupby('attack_percentage')[error_col].mean().reset_index()
                        zoom_data = grouped[(grouped['attack_percentage'] >= 0) & (grouped['attack_percentage'] <= 80)]
                        if len(zoom_data) > 0:
                            y_values.extend(zoom_data[error_col].values)
            
            if y_values:
                y_min = min(y_values) - 0.005
                y_max = max(y_values) + 0.005
                
                for algo in inset_algorithms:
                    algo_data = df_filtered[df_filtered['algorithm'] == algo]

                    if len(algo_data) > 0:
                        # Filter to only include (attack_percentage, algorithm) combinations that exist in BOTH datasets
                        algo_combos = [(row['attack_percentage'], row['algorithm']) for _, row in algo_data.iterrows()]
                        valid_rows = [combo in common_combos for combo in algo_combos]
                        algo_data = algo_data[valid_rows]
                        if len(algo_data) > 0:
                            grouped = algo_data.groupby('attack_percentage')[error_col].mean().reset_index()
                            grouped = grouped.sort_values('attack_percentage')

                            zoom_data = grouped[(grouped['attack_percentage'] >= 0) & (grouped['attack_percentage'] <= 80)]

                            if len(zoom_data) > 0:
                                z_orders = {'sketchguard': 5, 'balance': 4, 'ubar': 3, 'krum': 2, 'd-fedavg': 1}
                                axins.plot(zoom_data['attack_percentage'], 
                                      zoom_data[error_col],
                                      linestyle=line_styles[algo],
                                      color=colors[algo],
                                      linewidth=1.2,
                                      marker=markers[algo],
                                      markersize=2,
                                      markeredgewidth=0.2,
                                      markeredgecolor=colors[algo],
                                      markevery=2,
                                      zorder=z_orders[algo])
                
                # Inset settings
                axins.set_xlim(0, 80)
                axins.set_ylim(y_min, y_max)
                axins.set_xticks([0, 20, 40, 60, 80])
                axins.set_xticklabels(['0', '20', '40', '60', '80'], fontsize=6)
                
                y_ticks = np.linspace(y_min, y_max, 3)
                axins.set_yticks(y_ticks)
                axins.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
                axins.tick_params(labelsize=6)
                axins.grid(True, alpha=0.3, linewidth=0.2)
                
                axins.spines['top'].set_visible(False)
                axins.spines['right'].set_visible(False)
                
                axins.text(0.5, 0.95, 'Zoom', transform=axins.transAxes, 
                           fontsize=6, ha='center', va='top', style='italic')
        
        # Main subplot settings
        ax.set_xlabel('Frac. of malicious clients (%)')
        ax.set_ylabel('Max TER')
        
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(0, 90, 20))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        ax.grid(True, alpha=0.4, linewidth=0.5)
        
        # Add subplot labels underneath the plot
        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
        ax.text(0.5, -0.25, f'{panel_labels[idx]} {display_name}',
                transform=ax.transAxes, fontsize=12, ha='center', va='top')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide the last empty subplot
    if len(topology_combinations) < len(axes):
        axes[-1].set_visible(False)
    
    # Add single legend at the top of the figure
    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1.02),
                   ncol=5,
                   frameon=False,
                   fancybox=False,
                   shadow=False,
                   borderpad=0.3,
                   columnspacing=2.0,
                   handlelength=2.0,
                   handletextpad=0.5)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.25, hspace=0.4, wspace=0.3)
    
    # Save figure
    filename = get_output_path(f'{save_prefix}topology_comparison.pdf')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {filename}")
    
    png_filename = get_output_path(f'{save_prefix}topology_comparison.png')
    plt.savefig(png_filename, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {png_filename}")
    
    plt.show()
    plt.close()

def main():
    print("Loading data from extracted_accuracies.csv...")
    df = load_and_prepare_data()

    print(f"Loaded {len(df)} experiments")

    # Check available datasets
    datasets = df['dataset'].unique()
    print(f"Available datasets: {', '.join(datasets)}")

    # Show statistics per dataset
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]
        print(f"\n{dataset.upper()}: {len(dataset_data)} experiments")

        # Check available topologies for this dataset
        print(f"Available topology combinations for {dataset.upper()}:")
        topology_counts = dataset_data.groupby(['graph_type', 'graph_param']).size().reset_index(name='count')
        for _, row in topology_counts.iterrows():
            print(f"  {row['graph_type']} (param={row['graph_param']}): {row['count']} experiments")

    # Generate single averaged figure across all datasets
    print(f"\nGenerating averaged topology comparison figure across all datasets...")
    create_topology_figure(df, save_prefix='averaged_')

    print("\nAveraged topology comparison figure generated!")
    print("Features:")
    print("  - Single 1x5 subplot layout")
    print("  - Separate panel for each graph topology")
    print("  - Averages performance across FEMNIST and CelebA datasets")
    print("  - Directed deviation attack type")
    print("  - Honest node error rates")
    print("  - Inset zoom for overlapping algorithms")
    print("  - Single legend for all panels")

if __name__ == "__main__":
    main()