#!/usr/bin/env python3
"""
Script to run scalability experiments for decentralized learning algorithms.
Runs commands serially and saves logs with naming convention: agg_kvalue_dataset.log
All experiments run on CPU to ensure consistent performance and avoid GPU VRAM issues.
"""

import subprocess
import os
import sys
from datetime import datetime

# List of experiments to run
experiments = [
    {
        "agg": "balance",
        "k": 16,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg balance --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 16 --num-nodes 20"
    },
    {
        "agg": "sketchguard",
        "k": 16,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg sketchguard --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 16 --num-nodes 20"
    },
    {
        "agg": "ubar",
        "k": 16,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg ubar --ubar-rho 0.5 --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 16 --num-nodes 20"
    },
    {
        "agg": "balance",
        "k": 32,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg balance --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 32 --num-nodes 35"
    },
    {
        "agg": "sketchguard",
        "k": 32,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg sketchguard --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 32 --num-nodes 35"
    },
    {
        "agg": "ubar",
        "k": 32,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg ubar --ubar-rho 0.5 --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 32 --num-nodes 35"
    },
    {
        "agg": "balance",
        "k": 96,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg balance --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 96 --num-nodes 100"
    },
    {
        "agg": "sketchguard",
        "k": 96,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg sketchguard --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 96 --num-nodes 100"
    },
    {
        "agg": "ubar",
        "k": 96,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg ubar --ubar-rho 0.5 --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 96 --num-nodes 100"
    },
    {
        "agg": "sketchguard",
        "k": 154,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg sketchguard --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 154 --num-nodes 155"
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
        "agg": "sketchguard",
        "k": 299,
        "dataset": "femnist",
        "cmd": "python decentralized_fl_sim.py --dataset femnist --rounds 3 --local-epochs 1 --seed 987654321 --batch-size 64 --lr 0.01 --max-samples 10000 --agg sketchguard --attack-percentage 0.5 --attack-type directed_deviation --verbose --graph k-regular --k 299 --num-nodes 300"
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

    # Force CPU usage for all experiments
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices to force CPU

    print(f"\n{'='*60}")
    print(f"Running: {exp_dict['agg'].upper()} with k={exp_dict['k']} (CPU Mode)")
    print(f"Log file: {log_path}")
    print(f"Command: {exp_dict['cmd']}")
    print(f"{'='*60}")

    # Run the command and capture output
    try:
        with open(log_path, 'w') as log_file:
            # Write header with experiment details
            log_file.write(f"# Experiment: {exp_dict['agg']} k={exp_dict['k']} dataset={exp_dict['dataset']}\n")
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
    print("Starting scalability experiments (CPU Mode)...")
    print(f"Total experiments to run: {len(experiments)}")
    print("\n⚠️  All experiments will run on CPU for consistent performance")

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
    print(f"Logs saved in: results/scalability_results/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()