#!/usr/bin/env python3
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
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
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

    df['algorithm'] = df['aggregation-algorithm']

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
               frameon=False,
               fancybox=False,
               shadow=False,
               borderpad=0.3,
               columnspacing=2.0,
               handlelength=2.0,
               handletextpad=0.5)

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

def load_model_scalability_data(csv_file='extracted_time_model_scaling.csv'):
    """Load model dimension timing data and prepare for scalability visualization."""
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, '..', '..', csv_file)
    df = pd.read_csv(csv_path)

    df['algorithm'] = df['aggregation-algorithm']

    return df

def create_model_scalability_plot(df):
    """Create model dimension scalability comparison plot."""

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
            # Sort by model dimension for proper line plotting
            algo_data = algo_data.sort_values('model_dimension')

            display_name = algo.upper()

            line, = ax.plot(algo_data['model_dimension'],
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
    ax.set_xlabel('Model Dimension')
    ax.set_ylabel('Compute Time (s)')

    # Use log scale for x-axis due to large range of model dimensions
    ax.set_xscale('log')

    # Set axis limits
    ax.set_xlim(min(df['model_dimension']) * 0.8, max(df['model_dimension']) * 1.2)
    ax.set_ylim(0, max(df['compute-time']) * 1.1)

    # Format y-axis to show more precision
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))

    # Grid and styling to match paper
    ax.grid(True, alpha=0.4, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend at the top matching paper style
    fig.legend(legend_handles, legend_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.05),
               ncol=3,
               frameon=False,
               fancybox=False,
               shadow=False,
               borderpad=0.3,
               columnspacing=2.0,
               handlelength=2.0,
               handletextpad=0.5)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Ensure figures directory exists
    os.makedirs(os.path.dirname(get_output_path('model_scalability.pdf')), exist_ok=True)

    # Save figure
    pdf_filename = get_output_path('model_scalability.pdf')
    plt.savefig(pdf_filename, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {pdf_filename}")

    png_filename = get_output_path('model_scalability.png')
    plt.savefig(png_filename, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {png_filename}")

    plt.show()
    plt.close()

def create_combined_scalability_plot(df_network, df_model):
    """Create combined scalability plot with both network and model scalability as subplots."""

    # Create figure with 2x1 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

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

    # Create legend handles (will be shared)
    legend_handles = []
    legend_labels = []

    # ============ SUBPLOT 1: Network Scalability ============
    for algo in algorithms:
        algo_data = df_network[df_network['algorithm'] == algo]

        if len(algo_data) > 0:
            # Sort by node degree for proper line plotting
            algo_data = algo_data.sort_values('node-degree')

            display_name = algo.upper()

            line, = ax1.plot(algo_data['node-degree'],
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

            if algo == algorithms[0]:  # Only collect legend handles once
                legend_handles.append(line)
                legend_labels.append(display_name)

    # Customize subplot 1
    ax1.set_xlabel('Node Degree')
    ax1.set_ylabel('Compute Time (s)')
    ax1.set_xlim(0, 320)
    ax1.set_ylim(0, max(df_network['compute-time']) * 1.1)
    ax1.set_xticks(range(0, 320, 50))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax1.grid(True, alpha=0.4, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(0.02, 0.98, '(a) Network Scalability', transform=ax1.transAxes,
             fontsize=13, va='top', weight='bold')

    # ============ SUBPLOT 2: Model Scalability ============
    for algo in algorithms:
        algo_data = df_model[df_model['algorithm'] == algo]

        if len(algo_data) > 0:
            # Sort by model dimension for proper line plotting
            algo_data = algo_data.sort_values('model_dimension')

            display_name = algo.upper()

            line, = ax2.plot(algo_data['model_dimension'],
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

            if algo != algorithms[0]:  # Collect remaining legend handles
                legend_handles.append(line)
                legend_labels.append(display_name)

    # Customize subplot 2
    ax2.set_xlabel('Model Dimension')
    ax2.set_ylabel('Compute Time (s)')
    ax2.set_xscale('log')
    ax2.set_xlim(min(df_model['model_dimension']) * 0.8, max(df_model['model_dimension']) * 1.2)
    ax2.set_ylim(0, max(df_model['compute-time']) * 1.1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}' if max(df_model['compute-time']) < 5 else f'{y:.0f}'))
    ax2.grid(True, alpha=0.4, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(0.02, 0.98, '(b) Model Scalability', transform=ax2.transAxes,
             fontsize=13, va='top', weight='bold')

    # Add shared legend at the top
    fig.legend(legend_handles, legend_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.02),
               ncol=3,
               frameon=False,
               fancybox=False,
               shadow=False,
               borderpad=0.3,
               columnspacing=2.0,
               handlelength=2.0,
               handletextpad=0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.35)

    # Ensure figures directory exists
    os.makedirs(os.path.dirname(get_output_path('combined_scalability.pdf')), exist_ok=True)

    # Save figure
    pdf_filename = get_output_path('combined_scalability.pdf')
    plt.savefig(pdf_filename, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {pdf_filename}")

    png_filename = get_output_path('combined_scalability.png')
    plt.savefig(png_filename, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {png_filename}")

    plt.show()
    plt.close()

def main():
    # Load network scalability data
    print("Loading network scalability data from extracted_time_network_scaling.csv...")
    df_network = load_scalability_data()

    print(f"Loaded {len(df_network)} network timing measurements")
    print(f"Algorithms: {', '.join(df_network['algorithm'].unique())}")
    print(f"Node degrees: {', '.join(map(str, sorted(df_network['node-degree'].unique())))}")

    # Load model scalability data
    print("\nLoading model scalability data from extracted_time_model_scaling.csv...")
    df_model = load_model_scalability_data()

    print(f"Loaded {len(df_model)} model timing measurements")
    print(f"Algorithms: {', '.join(df_model['algorithm'].unique())}")
    print(f"Model dimensions: {', '.join(map(str, sorted(df_model['model_dimension'].unique())))}")

    # Create combined plot
    print("\nGenerating combined scalability plot (2x1 layout)...")
    create_combined_scalability_plot(df_network, df_model)

    print("\n✅ Combined scalability plot generated!")
    print("Features:")
    print("  - 2x1 subplot layout")
    print("  - (a) Network scalability: Compute time vs node degree")
    print("  - (b) Model scalability: Compute time vs model dimension")
    print("  - SKETCHGUARD vs BALANCE vs UBAR comparison")
    print("  - Publication-quality styling")
    print("  - Saved as both PDF and PNG in figures/ directory")

if __name__ == "__main__":
    main()