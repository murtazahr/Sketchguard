#!/usr/bin/env python3
"""
Script to run LARGE scalability experiments (k=154 and k=299) for decentralized learning algorithms.
These run on CPU to avoid GPU VRAM issues.
Saves logs with naming convention: agg_kvalue_dataset.log
"""

import subprocess
import os
import sys
from datetime import datetime

# Only the large experiments (k=154 and k=299)
experiments = [
    {
        "agg": "coarse",
        "k": 154,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg coarse --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },
    {
        "agg": "balance",
        "k": 154,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg balance --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },
    {
        "agg": "ubar",
        "k": 154,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg ubar --ubar-rho 0.5 --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
    },
    {
        "agg": "coarse",
        "k": 299,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg coarse --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 299 --num-nodes 300"
    },
    {
        "agg": "balance",
        "k": 299,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg balance --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 299 --num-nodes 300"
    },
    {
        "agg": "ubar",
        "k": 299,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg ubar --ubar-rho 0.5 --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 299 --num-nodes 300"
    }
]

def run_experiment(exp_dict):
    """Run a single experiment and save the log."""
    # Create log filename
    log_filename = f"{exp_dict['agg']}_{exp_dict['k']}_{exp_dict['dataset']}.log"
    log_path = os.path.join("results", "scalability_results", log_filename)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Force CPU usage for these large experiments to avoid GPU VRAM issues
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices to force CPU

    print(f"\n{'='*60}")
    print(f"Running: {exp_dict['agg'].upper()} with k={exp_dict['k']} (CPU Mode)")
    print(f"Nodes: {exp_dict['k']+1}")
    print(f"Log file: {log_path}")
    print(f"Command: {exp_dict['cmd']}")
    print(f"{'='*60}")

    # Run the command and capture output
    try:
        with open(log_path, 'w') as log_file:
            # Write header with experiment details
            log_file.write(f"# Experiment: {exp_dict['agg']} k={exp_dict['k']} dataset={exp_dict['dataset']}\n")
            log_file.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            log_file.write(f"# Device: CPU (forced for large experiment)\n")
            log_file.write(f"# Command: {exp_dict['cmd']}\n")
            log_file.write(f"{'='*80}\n\n")
            log_file.flush()

            # Run the command with CPU environment
            process = subprocess.Popen(
                exp_dict['cmd'].split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
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
    print("Starting LARGE scalability experiments (CPU mode)...")
    print(f"Total experiments to run: {len(experiments)}")
    print("\n⚠️  These experiments will run on CPU to avoid GPU VRAM issues")
    print("Configurations:")
    print("  - k=154 (155 nodes): COARSE, BALANCE, UBAR")
    print("  - k=299 (300 nodes): COARSE, BALANCE, UBAR")

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
    print(f"Average time per experiment: {duration / len(experiments)}")
    print(f"Logs saved in: results/scalability_results/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()