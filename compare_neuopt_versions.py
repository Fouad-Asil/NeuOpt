import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
from run import load_problem
from options import get_options
from agent.ppo import PPO
from utils import torch_load_cpu, get_inner_model, move_to

def toggle_memory_features(enable=True):
    """
    Enables or disables the memory features in the model architecture.
    This function modifies the code in place.
    """
    if enable:
        print("Enabling memory-augmented version...")
    else:
        print("Disabling memory features (reverting to original NeuOpt)...")
    
    # Use the toggle_memory.py utility to switch between versions
    from toggle_memory import toggle_memory_mode
    toggle_memory_mode(enable=enable)

def run_benchmark(args, problem_size, num_instances, memory_enabled=True, dataset=None):
    """
    Run benchmark evaluation of NeuOpt with or without memory.
    
    Args:
        args: Command line arguments
        problem_size: Size of problem instances
        num_instances: Number of instances to evaluate
        memory_enabled: Whether to use memory-augmented version
        dataset: Optional dataset to use (if None, will generate random instances)
        
    Returns:
        Dictionary of benchmark results
    """
    # Toggle memory features
    toggle_memory_features(memory_enabled)
    
    # Set problem size in args
    args.graph_size = problem_size
    
    # Set up problem
    ProblemClass = load_problem(args.problem)
    problem = ProblemClass(
                p_size = args.graph_size,
                init_val_met = args.init_val_met,
                with_assert = args.use_assert,
                DUMMY_RATE = args.dummy_rate,
                k = args.k,
                with_bonus = not args.wo_bonus,
                with_regular = not args.wo_regular
            )
    
    # Create/load dataset
    if dataset is None:
        dataset = problem.make_dataset(size=problem_size, num_samples=num_instances)
    
    # Initialize agent
    agent = PPO(problem, args)
    
    # Load pre-trained model if specified
    if args.load_path:
        agent.load(args.load_path)
        
    agent.eval()
    
    # Run evaluation
    start_time = time.time()
    
    # Prepare results container
    results = {
        'costs': [],
        'runtimes': [],
        'problem_size': problem_size,
        'num_instances': num_instances,
        'memory_enabled': memory_enabled,
        'model_path': args.load_path,
        'improvement_steps': []
    }
    
    # Evaluate each instance
    for i, instance in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating instances"):
        instance_start = time.time()
        batch = move_to({"coordinates": instance[None, :]}, args.device)  # Add batch dimension
        
        # Record solution quality at each step to see improvement
        step_costs = []
        
        # Initial solution
        solutions = move_to(problem.get_initial_solutions(batch), args.device)
        init_cost, _ = problem.get_costs(batch, solutions, get_context=True)
        step_costs.append(init_cost.item())
        
        # Run optimization for T steps
        T = args.T_eval
        val_m = args.val_m  # Number of augmentations
        stall_limit = args.stall_limit
        
        obj_val, obj_history, reward, _ = agent.rollout(problem, T, val_m, stall_limit, batch)
        
        # Record the solution quality at each step
        for t in range(T):
            if t < obj_history.size(1):
                step_costs.append(obj_history[0, t, 0].item())
        
        instance_runtime = time.time() - instance_start
        results['costs'].append(obj_val.item())
        results['runtimes'].append(instance_runtime)
        results['improvement_steps'].append(step_costs)
    
    # Calculate aggregate statistics
    results['total_runtime'] = time.time() - start_time
    results['mean_cost'] = np.mean(results['costs'])
    results['std_cost'] = np.std(results['costs'])
    results['mean_runtime'] = np.mean(results['runtimes'])
    
    return results

def plot_comparison(baseline_results, memory_results, problem_name, problem_size, output_dir):
    """
    Generate plots comparing baseline and memory-augmented results.
    
    Args:
        baseline_results: Results from baseline NeuOpt
        memory_results: Results from memory-augmented NeuOpt
        problem_name: Name of the problem (e.g., 'tsp', 'cvrp')
        problem_size: Size of problem instances
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Final solution quality comparison
    plt.figure(figsize=(10, 6))
    plt.boxplot([baseline_results['costs'], memory_results['costs']], 
                labels=['Baseline NeuOpt', 'Memory-Augmented NeuOpt'])
    plt.ylabel('Solution Cost')
    plt.title(f'{problem_name.upper()}{problem_size} Solution Quality Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f'{problem_name}{problem_size}_solution_quality.png'), dpi=300)
    
    # 2. Learning curve comparison (average improvement over time)
    plt.figure(figsize=(12, 6))
    
    # Get the maximum number of steps across all instances
    max_steps = max(
        max(len(steps) for steps in baseline_results['improvement_steps']),
        max(len(steps) for steps in memory_results['improvement_steps'])
    )
    
    # Pad shorter sequences with their final values
    baseline_curves = []
    for steps in baseline_results['improvement_steps']:
        padded = steps + [steps[-1]] * (max_steps - len(steps))
        baseline_curves.append(padded)
    
    memory_curves = []
    for steps in memory_results['improvement_steps']:
        padded = steps + [steps[-1]] * (max_steps - len(steps))
        memory_curves.append(padded)
    
    # Calculate averages
    baseline_avg = np.mean(baseline_curves, axis=0)
    memory_avg = np.mean(memory_curves, axis=0)
    
    # Plot
    plt.plot(baseline_avg, label='Baseline NeuOpt', linestyle='-', marker='o', markersize=4)
    plt.plot(memory_avg, label='Memory-Augmented NeuOpt', linestyle='-', marker='x', markersize=4)
    plt.xlabel('Optimization Step')
    plt.ylabel('Average Solution Cost')
    plt.title(f'{problem_name.upper()}{problem_size} Improvement Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f'{problem_name}{problem_size}_improvement_curve.png'), dpi=300)
    
    # 3. Relative improvement percentage
    relative_improvement = (np.array(baseline_results['costs']) - np.array(memory_results['costs'])) / np.array(baseline_results['costs']) * 100
    
    plt.figure(figsize=(10, 6))
    plt.hist(relative_improvement, bins=20, alpha=0.7, color='blue')
    plt.axvline(np.mean(relative_improvement), color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {np.mean(relative_improvement):.2f}%')
    plt.xlabel('Improvement Percentage (%)')
    plt.ylabel('Number of Instances')
    plt.title(f'{problem_name.upper()}{problem_size} Relative Improvement with Memory')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f'{problem_name}{problem_size}_relative_improvement.png'), dpi=300)

def main():
    parser = argparse.ArgumentParser(description='Compare Original NeuOpt with Memory-Augmented Version')
    
    # Add arguments for benchmark configuration
    parser.add_argument('--problem', default='tsp', choices=['tsp', 'cvrp'], help='Problem to solve')
    parser.add_argument('--sizes', type=int, nargs='+', default=[20, 50, 100], help='Problem sizes to benchmark')
    parser.add_argument('--instances', type=int, default=100, help='Number of problem instances per size')
    parser.add_argument('--eval_steps', type=int, default=100, help='Number of optimization steps')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--output_dir', default='comparison_results', help='Directory to save results')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pre-trained models if available')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    
    # Parse benchmark args first
    benchmark_args, remaining = parser.parse_known_args()
    
    # Get NeuOpt args
    neuopt_args = get_options(remaining)
    
    # Override some NeuOpt args with benchmark args
    neuopt_args.problem = benchmark_args.problem
    neuopt_args.T_eval = benchmark_args.eval_steps
    neuopt_args.seed = benchmark_args.seed
    neuopt_args.no_progress_bar = True  # Cleaner output
    
    # Set device
    if benchmark_args.cuda and torch.cuda.is_available():
        neuopt_args.device = torch.device('cuda')
    else:
        neuopt_args.device = torch.device('cpu')
    
    # Set result directory
    os.makedirs(benchmark_args.output_dir, exist_ok=True)
    
    # Fix random seeds
    torch.manual_seed(benchmark_args.seed)
    np.random.seed(benchmark_args.seed)
    
    for size in benchmark_args.sizes:
        print(f"\n{'='*80}")
        print(f"Benchmarking {benchmark_args.problem.upper()} with size {size}")
        print(f"{'='*80}")
        
        # Set pretrained model path if needed
        if benchmark_args.use_pretrained:
            neuopt_args.load_path = f"pre-trained/{benchmark_args.problem}{size}.pt"
            if not os.path.exists(neuopt_args.load_path):
                print(f"Warning: No pre-trained model found at {neuopt_args.load_path}. Using random initialization.")
                neuopt_args.load_path = None
        else:
            neuopt_args.load_path = None
        
        # Generate dataset for consistent comparison
        ProblemClass = load_problem(neuopt_args.problem)
        problem = ProblemClass(
                    p_size = size,
                    init_val_met = neuopt_args.init_val_met,
                    with_assert = neuopt_args.use_assert,
                    DUMMY_RATE = neuopt_args.dummy_rate,
                    k = neuopt_args.k,
                    with_bonus = not neuopt_args.wo_bonus,
                    with_regular = not neuopt_args.wo_regular
                )
        dataset = problem.make_dataset(size=size, num_samples=benchmark_args.instances, 
                                       DUMMY_RATE=0.5 if benchmark_args.problem == 'cvrp' else 0.0)
        
        # Run baseline (without memory)
        baseline_results = run_benchmark(
            neuopt_args, size, benchmark_args.instances, 
            memory_enabled=False, dataset=dataset
        )
        
        # Run memory-augmented version
        memory_results = run_benchmark(
            neuopt_args, size, benchmark_args.instances, 
            memory_enabled=True, dataset=dataset
        )
        
        # Generate plots
        plot_comparison(
            baseline_results, memory_results, 
            benchmark_args.problem, size, 
            benchmark_args.output_dir
        )
        
        # Save result data
        result_file = os.path.join(benchmark_args.output_dir, 
                                  f"{benchmark_args.problem}{size}_comparison.json")
        
        with open(result_file, 'w') as f:
            json.dump({
                'baseline': baseline_results,
                'memory_augmented': memory_results,
                'config': vars(benchmark_args)
            }, f, indent=2)
        
        print(f"\nResults summary for {benchmark_args.problem.upper()}{size}:")
        print(f"Baseline mean cost: {baseline_results['mean_cost']:.4f} ± {baseline_results['std_cost']:.4f}")
        print(f"Memory-aug mean cost: {memory_results['mean_cost']:.4f} ± {memory_results['std_cost']:.4f}")
        
        improvement = (baseline_results['mean_cost'] - memory_results['mean_cost']) / baseline_results['mean_cost'] * 100
        print(f"Average improvement: {improvement:.2f}%")
        
        print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main() 