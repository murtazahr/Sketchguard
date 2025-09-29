#!/usr/bin/env python3
import os
import re
import csv
import glob
from pathlib import Path

def extract_parameters_from_filename(filename):
    """Extract experiment parameters from the log filename."""
    # Example: femnist_20_10_3_erdos_02_64_10000_balance_40attack_1lambda.log
    # or: femnist_20_10_3_erdos_02_64_10000_balance_40attack_gaussian_1lambda.log
    # or: celeba_20_10_3_ring_64_10000_ubar_0attack_1lambda.log
    
    basename = os.path.basename(filename).replace('.log', '')
    parts = basename.split('_')
    
    params = {}
    params['dataset'] = parts[0]  # femnist or celeba
    params['num_nodes'] = parts[1]
    params['num_epochs'] = parts[2]
    params['local_epochs'] = parts[3]
    params['graph_type'] = parts[4]  # erdos, ring, etc.
    
    # Handle graph parameter (erdos has p-value, fully/ring don't)
    idx = 5
    if params['graph_type'] == 'erdos':
        # erdos has p-parameter: femnist_20_10_3_erdos_02_64_10000_...
        params['graph_param'] = parts[idx]
        idx += 1
    else:
        # fully/ring don't have p-parameter: femnist_20_10_3_fully_64_10000_...
        params['graph_param'] = 'NA'
    
    params['batch_size'] = parts[idx]
    params['samples_per_client'] = parts[idx + 1]
    params['algorithm'] = parts[idx + 2]  # balance, ubar, sketchguard, etc.
    
    # Parse attack configuration
    attack_part = parts[idx + 3]
    if 'attack' in attack_part:
        params['attack_percentage'] = attack_part.replace('attack', '')
    else:
        params['attack_percentage'] = '0'
    
    # Check if it's gaussian attack
    if idx + 4 < len(parts) and 'gaussian' in parts[idx + 4]:
        params['attack_type'] = 'gaussian'
        params['lambda'] = parts[idx + 5].replace('lambda', '') if idx + 5 < len(parts) else '1'
    else:
        params['attack_type'] = 'directed_deviation'
        params['lambda'] = parts[idx + 4].replace('lambda', '') if idx + 4 < len(parts) else '1'
    
    return params

def extract_final_accuracies(filepath):
    """Extract final honest and malicious accuracies from log file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # First, check attack percentage from filename to determine what to look for
    filename = os.path.basename(filepath)
    attack_match = re.search(r'(\d+)attack', filename)
    attack_percentage = int(attack_match.group(1)) if attack_match else 0
    
    # Pattern 1: "Final accuracy - Compromised: 0.8383, Honest: 0.8386"
    # This appears when there ARE compromised nodes
    pattern = r'Final accuracy - Compromised: ([\d.]+), Honest: ([\d.]+)'
    match = re.search(pattern, content)
    
    if match:
        compromised_acc = float(match.group(1))
        honest_acc = float(match.group(2))
        return compromised_acc, honest_acc
    
    # Pattern 2: For intermediate rounds, look for last occurrence
    # "compromised: 0.8383, honest: 0.8386"
    pattern2 = r'compromised: ([\d.]+), honest: ([\d.]+)'
    matches = re.findall(pattern2, content)
    if matches:
        last_match = matches[-1]
        return float(last_match[0]), float(last_match[1])
    
    # Pattern 3: When there's NO attack (0%), there might only be overall accuracy
    if attack_percentage == 0:
        # Look for "Overall test accuracy: mean=0.8342"
        pattern3 = r'Overall test accuracy: mean=([\d.]+)'
        match3 = re.search(pattern3, content)
        if match3:
            overall_acc = float(match3.group(1))
            # For 0% attack, both honest and compromised are the same (no compromised nodes)
            return overall_acc, overall_acc
    
    # Pattern 4: Some algorithms might just have test accuracy at the end
    # Try to find the last test accuracy mention
    pattern4 = r'test acc[uracy]*[\s:=]+([\d.]+)'
    matches4 = re.findall(pattern4, content, re.IGNORECASE)
    if matches4:
        last_acc = float(matches4[-1])
        # If no attack, both are the same
        if attack_percentage == 0:
            return last_acc, last_acc
        else:
            # This is not ideal but better than nothing
            return None, last_acc
    
    return None, None

def process_log_files(directory='results'):
    """Process all log files and extract accuracies."""
    log_files = glob.glob(os.path.join(directory, '*.log'))
    
    results = []
    failed_files = []
    
    print(f"Found {len(log_files)} log files to process\n")
    
    for i, log_file in enumerate(sorted(log_files), 1):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(log_files)} files processed...")
        
        # Extract parameters from filename
        try:
            params = extract_parameters_from_filename(log_file)
        except Exception as e:
            print(f"  ⚠️  Error parsing filename {os.path.basename(log_file)}: {e}")
            failed_files.append(log_file)
            continue
        
        # Extract accuracies from file content
        try:
            compromised_acc, honest_acc = extract_final_accuracies(log_file)
        except Exception as e:
            print(f"  ⚠️  Error extracting accuracies from {os.path.basename(log_file)}: {e}")
            failed_files.append(log_file)
            continue
        
        if compromised_acc is not None and honest_acc is not None:
            result = params.copy()
            result['final_compromised_accuracy'] = compromised_acc
            result['final_honest_accuracy'] = honest_acc
            result['filename'] = os.path.basename(log_file)
            results.append(result)
        else:
            failed_files.append(log_file)
            if len(failed_files) <= 10:  # Only show first 10 failures
                print(f"  ⚠️  Could not extract accuracies from: {os.path.basename(log_file)}")
    
    return results, failed_files

def save_to_csv(results, output_file=None):
    """Save results to CSV file."""
    if output_file is None:
        script_dir = os.path.dirname(__file__)
        output_file = os.path.join(script_dir, '..', '..', 'extracted_accuracies.csv')
    if not results:
        print("No results to save.")
        return
    
    # Define column order
    columns = [
        'dataset', 'num_nodes', 'num_epochs', 'local_epochs', 
        'graph_type', 'graph_param', 'batch_size', 'samples_per_client',
        'algorithm', 'attack_percentage', 'attack_type', 'lambda',
        'final_compromised_accuracy', 'final_honest_accuracy', 'filename'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Results saved to {output_file}")

def main():
    print("=" * 60)
    print("Extracting Final Accuracies from Experiment Log Files")
    print("=" * 60)
    
    # Get the path to results directory relative to script location
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, '..', '..', 'results')
    
    results, failed_files = process_log_files(results_dir)
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully processed: {len(results)} files")
    print(f"❌ Failed to process: {len(failed_files)} files")
    
    if failed_files:
        print(f"\nShowing first {min(10, len(failed_files))} failed files:")
        for f in failed_files[:10]:
            print(f"  - {os.path.basename(f)}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    # Save to CSV
    save_to_csv(results)
    
    # Print detailed summary statistics
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        # Group by dataset
        datasets = set(r['dataset'] for r in results)
        for dataset in sorted(datasets):
            dataset_results = [r for r in results if r['dataset'] == dataset]
            print(f"\n📊 {dataset.upper()}: {len(dataset_results)} experiments")
            
            # Group by algorithm
            algorithms = set(r['algorithm'] for r in dataset_results)
            for algo in sorted(algorithms):
                algo_results = [r for r in dataset_results if r['algorithm'] == algo]
                print(f"  └─ {algo}: {len(algo_results)} runs")
                
                # Group by attack type
                attack_types = set(r['attack_type'] for r in algo_results)
                for attack_type in sorted(attack_types):
                    attack_results = [r for r in algo_results if r['attack_type'] == attack_type]
                    attack_percentages = sorted(set(int(r['attack_percentage']) for r in attack_results))
                    print(f"      └─ {attack_type}: {len(attack_results)} runs (attack %: {attack_percentages})")
        
        print("\n" + "=" * 60)
        print("You can now use 'extracted_accuracies.csv' for visualization!")
        print("=" * 60)

if __name__ == "__main__":
    main()