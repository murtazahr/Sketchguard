#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
})

def get_output_path(filename):
    """Get the correct path for output files."""
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, '..', '..', 'figures', filename)

def load_scalability_data(csv_file='extracted_time_network_scaling.csv'):
    """Load timing data and prepare for scalability visualization."""
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, '..', '..', csv_file)
    df = pd.read_csv(csv_path)

    # Rename 'coarse' to 'sketchguard' for display
    df['algorithm'] = df['aggregation-algorithm'].replace('coarse', 'sketchguard')

    return df

def create_scalability_plot(df):
    """Create scalability comparison plot."""

    # Create figure with paper-style dimensions
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    # Define algorithms and their visual properties matching paper style
    algorithms = ['balance', 'ubar', 'sketchguard']

    # Define colors and styles exactly matching the paper figures
    colors = {
        'balance': '#0000FF',    # Blue
        'ubar': '#FF8C00',       # Orange
        'sketchguard': '#FF00FF' # Magenta/Pink
    }

    line_styles = {
        'balance': '-.',             # Dash-dot
        'ubar': '-',                 # Solid
        'sketchguard': (0, (1, 1))   # More dotted
    }

    markers = {
        'balance': 's',      # Square
        'ubar': '^',         # Triangle up
        'sketchguard': 'D'   # Diamond
    }

    # Create legend handles
    legend_handles = []
    legend_labels = []

    # Plot lines for each algorithm
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo]

        if len(algo_data) > 0:
            # Sort by node degree for proper line plotting
            algo_data = algo_data.sort_values('node-degree')

            display_name = algo.upper()

            line, = ax.plot(algo_data['node-degree'],
                           algo_data['compute-time'],
                           label=display_name,
                           linestyle=line_styles[algo],
                           color=colors[algo],
                           marker=markers[algo],
                           markersize=4,
                           linewidth=1.8,
                           markeredgewidth=0.5,
                           markeredgecolor=colors[algo],
                           markevery=1)

            legend_handles.append(line)
            legend_labels.append(display_name)

    # Customize plot - no title to match paper style
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Compute Time (s)')

    # Set axis limits and ticks with consistent intervals
    ax.set_xlim(0, 320)
    ax.set_ylim(0, max(df['compute-time']) * 1.1)

    # Set x-axis ticks at regular intervals
    ax.set_xticks(range(0, 320, 50))

    # Format y-axis to show more precision
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    # Grid and styling to match paper
    ax.grid(True, alpha=0.4, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend at the top matching paper style
    fig.legend(legend_handles, legend_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.05),
               ncol=3,
               frameon=True,
               fancybox=False,
               shadow=False,
               borderpad=0.3,
               columnspacing=2.0,
               handlelength=2.0,
               handletextpad=0.5,
               edgecolor='black',
               facecolor='white')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Ensure figures directory exists
    os.makedirs(os.path.dirname(get_output_path('scalability.pdf')), exist_ok=True)

    # Save figure
    pdf_filename = get_output_path('scalability.pdf')
    plt.savefig(pdf_filename, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {pdf_filename}")

    png_filename = get_output_path('scalability.png')
    plt.savefig(png_filename, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {png_filename}")

    plt.show()
    plt.close()

def main():
    print("Loading scalability data from extracted_time_network_scaling.csv...")
    df = load_scalability_data()

    print(f"Loaded {len(df)} timing measurements")
    print(f"Algorithms: {', '.join(df['algorithm'].unique())}")
    print(f"Node degrees: {', '.join(map(str, sorted(df['node-degree'].unique())))}")

    print("\nGenerating scalability comparison plot...")
    create_scalability_plot(df)

    print("\n✅ Scalability plot generated!")
    print("Features:")
    print("  - Compute time vs node degree comparison")
    print("  - SKETCHGUARD vs BALANCE vs UBAR")
    print("  - Publication-quality styling")
    print("  - Saved as both PDF and PNG in figures/ directory")

if __name__ == "__main__":
    main()