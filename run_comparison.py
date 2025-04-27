#!/usr/bin/env python
"""
Main script to run the comparison between original NeuOpt and memory-augmented NeuOpt.
This script orchestrates the entire evaluation process.
"""
import os
import sys
import subprocess
import argparse

def run_command(cmd):
    """Run a command and return its exit code"""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)

def main():
    parser = argparse.ArgumentParser(description='Compare Original NeuOpt with Memory-Augmented NeuOpt')
    
    # Add arguments for benchmark configuration
    parser.add_argument('--problem', default='tsp', choices=['tsp', 'cvrp'], help='Problem to solve')
    parser.add_argument('--sizes', type=int, nargs='+', default=[20], help='Problem sizes to benchmark')
    parser.add_argument('--instances', type=int, default=10, help='Number of problem instances per size')
    parser.add_argument('--eval_steps', type=int, default=50, help='Number of optimization steps')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--output_dir', default='comparison_results', help='Directory to save results')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pre-trained models if available')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    
    args = parser.parse_args()
    
    # Step 1: Backup code versions
    print("Step 1: Backing up code versions...")
    backup_result = run_command(["python", "backup_code_versions.py"])
    if backup_result != 0:
        print("Backup process failed. Please fix the issues and try again.")
        return 1
    
    # Step 2: Run comparison
    print("\nStep 2: Running comparison between original and memory-augmented NeuOpt...")
    
    comparison_cmd = [
        "python", "compare_neuopt_versions.py",
        "--problem", args.problem
    ]
    
    # Add sizes
    comparison_cmd.extend(["--sizes"] + [str(s) for s in args.sizes])
    
    # Add other args
    comparison_cmd.extend([
        "--instances", str(args.instances),
        "--eval_steps", str(args.eval_steps),
        "--seed", str(args.seed),
        "--output_dir", args.output_dir
    ])
    
    if args.use_pretrained:
        comparison_cmd.append("--use_pretrained")
    
    if args.cuda:
        comparison_cmd.append("--cuda")
    
    comparison_result = run_command(comparison_cmd)
    if comparison_result != 0:
        print("Comparison process failed. Check the logs for errors.")
        return 1
    
    # Step 3: Restore to memory-augmented version
    print("\nStep 3: Restoring to memory-augmented version...")
    restore_result = run_command(["python", "toggle_memory.py", "--mode", "memory"])
    if restore_result != 0:
        print("Failed to restore memory-augmented version. Check your backup files.")
        return 1
    
    print("\nComparison completed successfully!")
    print(f"Results saved in: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 