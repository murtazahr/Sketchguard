#!/usr/bin/env python3
"""
Script to run decentralized federated learning experiments with various parameter combinations.
"""

import subprocess
import itertools
from typing import List, Dict, Any
import argparse
import time

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

def get_datasets():
    """Get all datasets."""
    return ["femnist", "celeba"]

def build_log_filename(dataset, graph_config, agg_method, attack_pct):
    """Build the log filename based on parameters."""
    # Format attack percentage for filename
    attack_str = f"{int(attack_pct*100)}attack" if attack_pct > 0 else "0attack"
    
    # Format graph name for filename
    if graph_config["name"] == "erdos":
        graph_str = f"erdos_{str(graph_config['p']).replace('.', '')}"
    else:
        graph_str = graph_config["name"]
    
    # Build filename
    filename = f"{dataset}_20_10_3_{graph_str}_64_10000_{agg_method}_{attack_str}_1lambda.log"
    return filename

def build_command(dataset, graph_config, agg_method, attack_pct):
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

def run_experiment(dataset, graph_config, agg_method, attack_pct, dry_run=False):
    """Run a single experiment."""
    cmd = build_command(dataset, graph_config, agg_method, attack_pct)
    log_filename = build_log_filename(dataset, graph_config, agg_method, attack_pct)
    
    print(f"\n{'='*80}")
    print(f"Running: {dataset} | {graph_config['name']}" + 
          (f" (p={graph_config['p']})" if graph_config['p'] else "") +
          f" | {agg_method} | attack={attack_pct}")
    print(f"Output: {log_filename}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("DRY RUN - Command not executed")
        return True
    
    try:
        with open(log_filename, 'w') as log_file:
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            process.wait()
            
        if process.returncode == 0:
            print(f"✓ Completed successfully")
            return True
        else:
            print(f"✗ Failed with return code {process.returncode}")
            return False
    except Exception as e:
        print(f"✗ Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run FL simulation experiments')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Print commands without executing them')
    parser.add_argument('--datasets', nargs='+', choices=['femnist', 'celeba'],
                        help='Specific datasets to run (default: all)')
    parser.add_argument('--agg-methods', nargs='+', 
                        choices=['coarse', 'balance', 'krum', 'd-fedadj', 'ubar'],
                        help='Specific aggregation methods to run (default: all)')
    parser.add_argument('--attack-percentages', nargs='+', type=float,
                        help='Specific attack percentages to run (default: all)')
    args = parser.parse_args()
    
    # Get parameter combinations
    datasets = args.datasets if args.datasets else get_datasets()
    graph_configs = get_graph_configs()
    agg_methods = args.agg_methods if args.agg_methods else get_aggregation_methods()
    attack_percentages = (args.attack_percentages if args.attack_percentages 
                         else get_attack_percentages())
    
    # Validate attack percentages
    for pct in attack_percentages:
        if not 0 <= pct <= 1:
            print(f"Error: Attack percentage {pct} must be between 0 and 1")
            return
    
    # Calculate total experiments
    total_experiments = (len(datasets) * len(graph_configs) * 
                        len(agg_methods) * len(attack_percentages))
    
    print(f"Total experiments to run: {total_experiments}")
    print(f"Datasets: {datasets}")
    print(f"Graph configs: {len(graph_configs)} configurations")
    print(f"Aggregation methods: {agg_methods}")
    print(f"Attack percentages: {attack_percentages}")
    
    # Run all experiments
    successful = 0
    failed = 0
    experiment_count = 0
    start_time = time.time()
    
    for dataset, graph_config, agg_method, attack_pct in itertools.product(
        datasets, graph_configs, agg_methods, attack_percentages
    ):
        experiment_count += 1
        print(f"\nExperiment {experiment_count}/{total_experiments}")
        
        success = run_experiment(dataset, graph_config, agg_method, attack_pct, 
                               dry_run=args.dry_run)
        if success:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    if not args.dry_run:
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average time per experiment: {elapsed_time/total_experiments:.2f} seconds")
    else:
        print("DRY RUN - No experiments were actually executed")

if __name__ == "__main__":
    main()