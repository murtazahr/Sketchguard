#!/usr/bin/env python3
"""
Script to check which experiments have been completed and generate commands for missing ones.
"""

import os
import itertools
from typing import List, Set
import argparse

def get_graph_configs():
    """Get all graph configurations."""
    return [
        {"name": "erdos", "p": 0.2},
        {"name": "erdos", "p": 0.45},
        {"name": "erdos", "p": 0.6},
        {"name": "fully", "p": None},
        {"name": "ring", "p": None}
    ]

def get_aggregation_methods():
    """Get all aggregation methods."""
    return ["coarse", "balance", "krum", "d-fedadj", "ubar"]

def get_attack_percentages():
    """Get all attack percentages."""
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def get_attack_types():
    """Get all attack types."""
    return ["directed_deviation", "gaussian"]

def get_datasets():
    """Get all datasets."""
    return ["femnist", "celeba"]

def build_log_filename(dataset, graph_config, agg_method, attack_pct, attack_type="directed_deviation"):
    """Build the log filename based on parameters."""
    # Format attack percentage for filename
    attack_str = f"{int(attack_pct*100)}attack" if attack_pct > 0 else "0attack"
    
    # Format graph name for filename
    if graph_config["name"] == "erdos":
        graph_str = f"erdos_{str(graph_config['p']).replace('.', '')}"
    else:
        graph_str = graph_config["name"]
    
    # Add attack type suffix for gaussian attacks
    attack_suffix = "_gaussian" if attack_type == "gaussian" and attack_pct > 0 else ""
    
    # Build filename
    filename = f"{dataset}_20_10_3_{graph_str}_64_10000_{agg_method}_{attack_str}{attack_suffix}_1lambda.log"
    return filename

def build_command(dataset, graph_config, agg_method, attack_pct, attack_type="directed_deviation"):
    """Build the command to run."""
    cmd = [
        "python", "decentralized_fl_sim.py",
        "--dataset", dataset,
        "--num-nodes", "20",
        "--rounds", "10",
        "--local-epochs", "3",
        "--seed", "987654321",
        "--graph", graph_config["name"],
        "--batch-size", "64",
        "--lr", "0.01",
        "--max-samples", "10000",
        "--agg", agg_method,
        "--attack-percentage", str(attack_pct),
        "--attack-type", attack_type,
        "--verbose"
    ]
    
    # Add p parameter for erdos graphs
    if graph_config["name"] == "erdos" and graph_config["p"] is not None:
        cmd.extend(["--p", str(graph_config["p"])])
    
    # Add pct-compromised for krum
    if agg_method == "krum":
        cmd.extend(["--pct-compromised", str(attack_pct)])
    
    # Add ubar-rho for ubar (1 - attack_percentage)
    if agg_method == "ubar":
        ubar_rho = 1.0 - attack_pct
        cmd.extend(["--ubar-rho", str(ubar_rho)])
    
    return cmd

def get_existing_logs(directory="."):
    """Get all existing log files in the directory."""
    existing = set()
    for file in os.listdir(directory):
        if file.endswith(".log"):
            existing.add(file)
    return existing

def main():
    parser = argparse.ArgumentParser(description='Check missing experiments and generate resume script')
    parser.add_argument('--datasets', nargs='+', choices=['femnist', 'celeba'],
                        help='Specific datasets to check (default: all)')
    parser.add_argument('--agg-methods', nargs='+', 
                        choices=['coarse', 'balance', 'krum', 'd-fedadj', 'ubar'],
                        help='Specific aggregation methods to check (default: all)')
    parser.add_argument('--generate-script', action='store_true',
                        help='Generate a bash script to run missing experiments')
    parser.add_argument('--remote-path', type=str, default='.',
                        help='Path on remote machine where experiments are run')
    args = parser.parse_args()
    
    # Get parameter combinations
    datasets = args.datasets if args.datasets else get_datasets()
    agg_methods = args.agg_methods if args.agg_methods else get_aggregation_methods()
    attack_percentages = get_attack_percentages()
    attack_types = get_attack_types()
    graph_configs = get_graph_configs()
    
    # Get existing log files
    existing_logs = get_existing_logs()
    
    # Track completed and missing experiments
    completed = []
    missing = []
    
    # Check all experiment combinations
    for dataset, graph_config, agg_method, attack_pct in itertools.product(
        datasets, graph_configs, agg_methods, attack_percentages
    ):
        # For 0% attack, only check once (attack type doesn't matter)
        if attack_pct == 0:
            log_file = build_log_filename(dataset, graph_config, agg_method, attack_pct)
            if log_file in existing_logs:
                completed.append((dataset, graph_config, agg_method, attack_pct, "directed_deviation"))
            else:
                missing.append((dataset, graph_config, agg_method, attack_pct, "directed_deviation"))
        else:
            # For non-zero attacks, check for each attack type
            for attack_type in attack_types:
                log_file = build_log_filename(dataset, graph_config, agg_method, attack_pct, attack_type)
                if log_file in existing_logs:
                    completed.append((dataset, graph_config, agg_method, attack_pct, attack_type))
                else:
                    missing.append((dataset, graph_config, agg_method, attack_pct, attack_type))
    
    # Print summary
    total = len(completed) + len(missing)
    print(f"Experiment Status Summary")
    print(f"{'='*50}")
    print(f"Total experiments: {total}")
    print(f"Completed: {len(completed)} ({100*len(completed)/total:.1f}%)")
    print(f"Missing: {len(missing)} ({100*len(missing)/total:.1f}%)")
    
    if missing:
        print(f"\nMissing Experiments:")
        print(f"{'-'*50}")
        
        # Group missing experiments by dataset and aggregation method for readability
        grouped = {}
        for dataset, graph_config, agg_method, attack_pct, attack_type in missing:
            key = (dataset, agg_method)
            if key not in grouped:
                grouped[key] = []
            graph_str = f"{graph_config['name']}" + (f"(p={graph_config['p']})" if graph_config['p'] else "")
            grouped[key].append((graph_str, attack_pct, attack_type))
        
        for (dataset, agg_method), experiments in grouped.items():
            print(f"\n{dataset} - {agg_method}:")
            for graph_str, attack_pct, attack_type in experiments[:5]:  # Show first 5
                print(f"  - {graph_str}, attack={attack_pct:.0%} ({attack_type})")
            if len(experiments) > 5:
                print(f"  ... and {len(experiments)-5} more")
    
    # Generate resume script if requested
    if args.generate_script and missing:
        script_name = "resume_experiments.sh"
        with open(script_name, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Script to resume missing experiments\n")
            f.write(f"# Generated script will run {len(missing)} missing experiments\n\n")
            
            if args.remote_path != '.':
                f.write(f"cd {args.remote_path}\n\n")
            
            f.write("# Counter for tracking progress\n")
            f.write("COMPLETED=0\n")
            f.write(f"TOTAL={len(missing)}\n\n")
            
            for i, (dataset, graph_config, agg_method, attack_pct, attack_type) in enumerate(missing, 1):
                cmd = build_command(dataset, graph_config, agg_method, attack_pct, attack_type)
                log_file = build_log_filename(dataset, graph_config, agg_method, attack_pct, attack_type)
                
                f.write(f"# Experiment {i}/{len(missing)}\n")
                f.write(f'echo "Running experiment $((COMPLETED+1))/$TOTAL: {dataset} {agg_method} attack={attack_pct:.0%} ({attack_type})"\n')
                f.write(f"{' '.join(cmd)} > {log_file} 2>&1\n")
                f.write("if [ $? -eq 0 ]; then\n")
                f.write('    echo "  ✓ Completed successfully"\n')
                f.write("else\n")
                f.write('    echo "  ✗ Failed"\n')
                f.write("fi\n")
                f.write("COMPLETED=$((COMPLETED+1))\n")
                f.write(f'echo "Progress: $COMPLETED/$TOTAL completed"\n')
                f.write("\n")
            
            f.write('echo "All experiments completed!"\n')
        
        os.chmod(script_name, 0o755)
        print(f"\nGenerated resume script: {script_name}")
        print(f"To run on remote machine: scp {script_name} remote:path/ && ssh remote 'cd path && ./{script_name}'")

if __name__ == "__main__":
    main()