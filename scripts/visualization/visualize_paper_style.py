#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

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
    'figure.figsize': (8, 3),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.8,
    'lines.markersize': 4,
})

def get_output_path(filename):
    """Get the correct path for output files."""
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, '..', '..', filename)

def load_and_prepare_data(csv_file='extracted_accuracies_v2.csv'):
    """Load CSV and prepare data for visualization."""
    # Get the correct path to data files relative to script location
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, '..', '..', csv_file)
    df = pd.read_csv(csv_path)

    # Convert attack_percentage to integer
    df['attack_percentage'] = df['attack_percentage'].astype(int)

    # Calculate error rates (1 - accuracy)
    df['honest_error_rate'] = 1 - df['final_honest_accuracy']
    df['compromised_error_rate'] = 1 - df['final_compromised_accuracy']

    return df

def create_combined_figure(df, save_prefix=''):
    """Create a multi-panel figure with insets matching the paper style."""

    # Define datasets and attack types
    datasets = ['femnist', 'celeba', 'sent140']
    attack_types = ['directed_deviation', 'gaussian', 'backdoor', 'krum']

    # Create figure with 3x4 grid (3 datasets x 4 attack types)
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))

    # Build subplot configs dynamically
    subplot_configs = []
    panel_idx = 0
    panel_labels = 'abcdefghijkl'
    dataset_display = {'femnist': 'FEMNIST', 'celeba': 'CelebA', 'sent140': 'Sent140'}
    attack_display = {
        'directed_deviation': 'Directed Deviation',
        'gaussian': 'Gaussian',
        'backdoor': 'Backdoor',
        'krum': 'Krum Attack'
    }

    for dataset in datasets:
        for attack_type in attack_types:
            label = f'({panel_labels[panel_idx]}) {dataset_display[dataset]} - {attack_display[attack_type]}'
            subplot_configs.append((dataset, attack_type, label))
            panel_idx += 1

    node_type = 'honest'  # Focus on honest nodes
    
    # Define algorithms and their visual properties matching the style
    algorithms = ['d-fedavg', 'krum', 'balance', 'ubar', 'sketchguard']

    # Define colors and styles to match the reference
    colors = {
        'd-fedavg': '#DC143C',  # Crimson red for FedAvg
        'krum': '#000000',      # Black with dots
        'balance': '#0000FF',   # Blue dash-dot
        'ubar': '#FF8C00',      # Orange solid
        'sketchguard': '#FF00FF'     # Magenta/Pink dashed
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
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    for idx, (ax, (dataset, attack_type, label)) in enumerate(zip(axes_flat, subplot_configs)):
        # Filter data for specific dataset and attack type
        df_filtered = df[(df['dataset'] == dataset) & (df['attack_type'] == attack_type)].copy()
        error_col = f'{node_type}_error_rate'

        # For gaussian attack at 0%, use directed_deviation data from the same dataset
        if attack_type == 'gaussian':
            df_zero = df[(df['dataset'] == dataset) & (df['attack_type'] == 'directed_deviation') & (df['attack_percentage'] == 0)].copy()
            df_zero['attack_type'] = 'gaussian'
            df_filtered = pd.concat([df_filtered, df_zero]).drop_duplicates(subset=['algorithm', 'attack_percentage'])
        
        # Plot lines for each algorithm
        for algo in algorithms:
            algo_data = df_filtered[df_filtered['algorithm'] == algo]
            
            if len(algo_data) > 0:
                grouped = algo_data.groupby('attack_percentage')[error_col].mean().reset_index()
                grouped = grouped.sort_values('attack_percentage')
                
                # Set z-order: sketchguard (top), balance (middle), ubar (bottom), krum (separate), fedavg (back)
                z_orders = {'sketchguard': 5, 'balance': 4, 'ubar': 3, 'krum': 2, 'd-fedavg': 1}

                # Map algorithm names for display
                if algo == 'sketchguard':
                    display_name = 'SKETCHGUARD'
                elif algo == 'd-fedavg':
                    display_name = 'FEDAVG'
                else:
                    display_name = algo.upper()
                
                line, = ax.plot(grouped['attack_percentage'], 
                               grouped[error_col],
                               label=display_name,
                               linestyle=line_styles[algo],
                               color=colors[algo],
                               marker=markers[algo],
                               markersize=4,
                               linewidth=1.8,
                               markeredgewidth=0.5,
                               markeredgecolor=colors[algo],
                               markevery=2,  # Show markers every other point
                               zorder=z_orders[algo])  # Control drawing order
                
                # Collect legend handles only once (from first subplot)
                if idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(display_name)
        
        # Create inset axes (smaller size) - top right for CelebA
        if dataset == 'celeba':
            axins = inset_axes(ax, width="35%", height="35%",
                              loc='upper right',
                              bbox_to_anchor=(0, 0, 1, 1),
                              bbox_transform=ax.transAxes)
        else:
            axins = inset_axes(ax, width="35%", height="35%",
                              loc='lower right',
                              bbox_to_anchor=(0, 0.3, 1, 1),
                              bbox_transform=ax.transAxes)
        
        # Plot in inset - only the three overlapping algorithms
        inset_algorithms = ['balance', 'ubar', 'sketchguard']
        
        for algo in inset_algorithms:
            algo_data = df_filtered[df_filtered['algorithm'] == algo]
            
            if len(algo_data) > 0:
                grouped = algo_data.groupby('attack_percentage')[error_col].mean().reset_index()
                grouped = grouped.sort_values('attack_percentage')
                
                # Focus on 10-80% range for inset
                zoom_data = grouped[(grouped['attack_percentage'] >= 10) & (grouped['attack_percentage'] <= 80)]
                
                if len(zoom_data) > 0:
                    z_orders = {'sketchguard': 5, 'balance': 4, 'ubar': 3, 'krum': 2, 'd-fedavg': 1}
                    axins.plot(zoom_data['attack_percentage'], 
                              zoom_data[error_col],
                              linestyle=line_styles[algo],
                              color=colors[algo],
                              linewidth=1.5,
                              marker=markers[algo],
                              markersize=3,
                              markeredgewidth=0.3,
                              markeredgecolor=colors[algo],
                              markevery=2,
                              zorder=z_orders[algo])
        
        # Inset settings
        axins.set_xlim(10, 80)
        
        # Calculate y-limits for inset
        y_values = []
        for algo in inset_algorithms:
            algo_data = df_filtered[df_filtered['algorithm'] == algo]
            if len(algo_data) > 0:
                grouped = algo_data.groupby('attack_percentage')[error_col].mean().reset_index()
                zoom_data = grouped[(grouped['attack_percentage'] >= 10) & (grouped['attack_percentage'] <= 80)]
                if len(zoom_data) > 0:
                    y_values.extend(zoom_data[error_col].values)
        
        if y_values:
            y_min = min(y_values) - 0.005
            y_max = max(y_values) + 0.005
        else:
            y_min, y_max = 0.16, 0.19
        
        axins.set_ylim(y_min, y_max)
        axins.set_xticks([10, 30, 50, 70])
        axins.set_xticklabels(['10', '30', '50', '70'], fontsize=7)
        
        # Format y-ticks for inset
        y_ticks = np.linspace(y_min, y_max, 3)
        axins.set_yticks(y_ticks)
        axins.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
        axins.tick_params(labelsize=7)
        axins.grid(True, alpha=0.3, linewidth=0.3)
        
        # Remove inset spines for cleaner look
        axins.spines['top'].set_visible(False)
        axins.spines['right'].set_visible(False)
        
        # Add "Zoom" label to inset
        axins.text(0.5, 0.95, 'Zoom', transform=axins.transAxes, 
                   fontsize=7, ha='center', va='top', style='italic')
        
        # Main subplot settings
        ax.set_xlabel('Frac. of malicious clients (%)')
        ax.set_ylabel('Max TER')  # Test Error Rate - on both subplots
        
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(0, 90, 10))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        ax.grid(True, alpha=0.4, linewidth=0.5)
        
        # Add subplot labels underneath the plot
        ax.text(0.5, -0.25, label,
                transform=ax.transAxes, fontsize=12, ha='center', va='top')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add single legend at the top of the figure
    fig.legend(legend_handles, legend_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.05),
               ncol=5,  # All algorithms in one row (now 5 with FedAvg)
               frameon=False,
               fancybox=False,
               shadow=False,
               borderpad=0.3,
               columnspacing=2.0,
               handlelength=2.0,
               handletextpad=0.5)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.45, wspace=0.3)  # Make room for legend and labels
    
    # Save figure
    filename = get_output_path(f'{save_prefix}paper_style.pdf')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {filename}")
    
    png_filename = get_output_path(f'{save_prefix}paper_style.png')
    plt.savefig(png_filename, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {png_filename}")
    
    plt.show()
    plt.close()

def main():
    print("Loading data from extracted_accuracies_v2.csv...")
    df = load_and_prepare_data()

    print(f"Loaded {len(df)} experiments")

    # Check available datasets
    datasets = df['dataset'].unique()
    print(f"Available datasets: {', '.join(datasets)}")

    # Check available attack types
    attack_types = df['attack_type'].unique()
    print(f"Available attack types: {', '.join(attack_types)}")

    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]
        print(f"{dataset.upper()}: {len(dataset_data)} experiments")

    print(f"\nGenerating combined paper-style figure with all datasets and attack types...")
    create_combined_figure(df, 'combined_')

    print("✅ Combined paper-style figure generated!")
    print("Features:")
    print("  - 3x4 subplot layout (3 datasets x 4 attack types)")
    print("  - Datasets: FEMNIST, CelebA, Sent140")
    print("  - Attack types: Directed Deviation, Gaussian, Backdoor, Krum Attack")
    print("  - Single legend at the top for all panels")
    print("  - Inset zoom from 10-80% on x-axis")
    print("  - Matching publication style from reference")

if __name__ == "__main__":
    main()