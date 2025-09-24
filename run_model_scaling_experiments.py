#!/usr/bin/env python3
"""
Script to run model scaling experiments for decentralized learning algorithms.
Runs commands serially and saves logs with naming convention: agg_modelvariant_dataset.log
Tests how different aggregation methods perform with varying model sizes.
All experiments run on CPU to ensure consistent performance.
"""

import subprocess
import os
from datetime import datetime

# List of experiments to run - testing model scaling with 50% malicious nodes
# Reordered to run all aggregation algorithms with tiny model, then small, then large, then xlarge
experiments = [
    # TINY model with all three aggregation algorithms
    {
        "agg": "balance",
        "model_variant": "tiny",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant tiny --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg balance --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },
    {
        "agg": "coarse",
        "model_variant": "tiny",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant tiny --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg coarse --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },
    {
        "agg": "ubar",
        "model_variant": "tiny",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant tiny --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg ubar --ubar-rho 0.4 --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },

    # SMALL model with all three aggregation algorithms
    {
        "agg": "balance",
        "model_variant": "small",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant small --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg balance --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },
    {
        "agg": "coarse",
        "model_variant": "small",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant small --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg coarse --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },
    {
        "agg": "ubar",
        "model_variant": "small",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant small --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg ubar --ubar-rho 0.4 --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },

    # LARGE model with all three aggregation algorithms
    {
        "agg": "balance",
        "model_variant": "large",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant large --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg balance --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },
    {
        "agg": "coarse",
        "model_variant": "large",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant large --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg coarse --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },
    {
        "agg": "ubar",
        "model_variant": "large",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant large --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg ubar --ubar-rho 0.4 --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },

    # XLARGE model with all three aggregation algorithms
    {
        "agg": "balance",
        "model_variant": "xlarge",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant xlarge --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg balance --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },
    {
        "agg": "coarse",
        "model_variant": "xlarge",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant xlarge --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg coarse --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },
    {
        "agg": "ubar",
        "model_variant": "xlarge",
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --model-variant xlarge --rounds 3 --local-epochs 1 --seed 42 --batch-size 32 --lr 0.01 --agg ubar --ubar-rho 0.4 --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    }
]

def run_experiment(exp_dict):
    """Run a single experiment and save the log."""
    # Create log filename
    log_filename = f"{exp_dict['agg']}_{exp_dict['model_variant']}_{exp_dict['dataset']}.log"
    log_path = os.path.join("results", "model_scaling_results", log_filename)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Force CPU usage for all experiments
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices to force CPU

    print(f"\n{'='*60}")
    print(f"Running: {exp_dict['agg'].upper()} with {exp_dict['model_variant']} model (CPU Mode)")
    print(f"Log file: {log_path}")
    print(f"Command: {exp_dict['cmd']}")
    print(f"{'='*60}")

    # Run the command and capture output
    try:
        with open(log_path, 'w') as log_file:
            # Write header with experiment details
            log_file.write(f"# Model Scaling Experiment\n")
            log_file.write(f"# Algorithm: {exp_dict['agg']}\n")
            log_file.write(f"# Model variant: {exp_dict['model_variant']}\n")
            log_file.write(f"# Dataset: {exp_dict['dataset']}\n")
            log_file.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            log_file.write(f"# Device: CPU (forced for consistent performance)\n")
            log_file.write(f"# Command: {exp_dict['cmd']}\n")
            log_file.write(f"{'='*80}\n\n")
            log_file.flush()

            # Run the command with CPU-only environment
            process = subprocess.Popen(
                exp_dict['cmd'].split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env  # Use modified environment to force CPU
            )

            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')  # Print to console
                log_file.write(line)  # Write to log file
                log_file.flush()

            # Wait for process to complete
            process.wait()

            if process.returncode == 0:
                print(f"✓ Successfully completed: {log_filename}")
                log_file.write(f"\n\n# Experiment completed successfully\n")
            else:
                print(f"✗ Failed with return code {process.returncode}: {log_filename}")
                log_file.write(f"\n\n# Experiment failed with return code {process.returncode}\n")

    except Exception as e:
        print(f"✗ Error running experiment: {e}")
        with open(log_path, 'a') as log_file:
            log_file.write(f"\n\n# Error: {e}\n")
        return False

    return True

def main():
    """Main function to run all experiments."""
    print("Starting model scaling experiments (CPU Mode)...")
    print(f"Total experiments to run: {len(experiments)}")
    print("\n⚠️  All experiments will run on CPU for consistent performance")
    print("🎯 Testing model scaling: all algorithms with tiny, then small, then large, then xlarge")
    print("🔄 New order: tiny (balance, coarse, ubar) → small (balance, coarse, ubar) → large (balance, coarse, ubar) → xlarge (balance, coarse, ubar)")
    print("🚨 50% malicious nodes, k-regular network")

    successful = 0
    failed = 0

    start_time = datetime.now()

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting experiment {i}...")
        if run_experiment(exp):
            successful += 1
        else:
            failed += 1

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {duration}")
    print(f"Logs saved in: results/model_scaling_results/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()