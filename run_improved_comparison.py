#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch
import random
from abc_comparison import run_comparison

def parse_args():
    parser = argparse.ArgumentParser(description='Run Improved ABC vs NeuOpt-enhanced ABC comparison')
    
    parser.add_argument('--dataset', type=str, default='datasets/tsp_20.pkl',
                        help='Path to the TSP dataset')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of TSP instances to solve')
    parser.add_argument('--colony_size', type=int, default=100,
                        help='Size of the colony (number of employed/onlooker bees)')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='Maximum number of iterations')
    parser.add_argument('--limit', type=int, default=50,
                        help='Maximum number of trials before abandonment')
    parser.add_argument('--output_dir', type=str, default='improved_comparison_results',
                        help='Directory to save results')
    parser.add_argument('--model_path', type=str, default='pre-trained/tsp20.pt',
                        help='Path to the pretrained NeuOpt model')
    parser.add_argument('--seed', type=int, default=6666,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Run comparison with improved parameters
    print("Running improved comparison with:")
    print(f"- Colony size: {args.colony_size}")
    print(f"- Max iterations: {args.max_iterations}")
    print(f"- Limit: {args.limit}")
    print(f"- Number of samples: {args.num_samples}")
    
    run_comparison(
        dataset=args.dataset,
        num_samples=args.num_samples,
        colony_size=args.colony_size,
        max_iterations=args.max_iterations,
        limit=args.limit,
        output_dir=args.output_dir,
        model_path=args.model_path
    )
    
    print(f"\nResults saved to {args.output_dir}")
    print("\nBoth algorithms now use improved 2-opt and 3-opt local search.")
    print("Check the output directory for visualizations.")

if __name__ == "__main__":
    main() 