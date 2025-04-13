import os
import numpy as np
import torch
import random
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from problems.problem_tsp import TSP, TSPDataset
from abc_neuopt import ABC_NeuOpt, load_tsp_dataset


class ABC_Original:
    """
    Original Artificial Bee Colony algorithm for TSP
    """
    def __init__(self, colony_size=50, max_iterations=1000, limit=20):
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.limit = limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Original ABC using device: {self.device}")
        
        # Initialize TSP problem
        self.problem = TSP(p_size=20, init_val_met='random')
    
    def _ensure_device(self, tensor_or_dict):
        """Helper method to ensure tensors are on the correct device"""
        if isinstance(tensor_or_dict, dict):
            return {k: self._ensure_device(v) for k, v in tensor_or_dict.items()}
        elif isinstance(tensor_or_dict, torch.Tensor):
            return tensor_or_dict.to(self.device)
        else:
            return tensor_or_dict
    
    def random_k_opt(self, solution, k=2):
        """
        Apply a random k-opt move (simplified version) for the original ABC
        """
        batch_size, graph_size = solution.size()
        
        # Ensure solution is on the correct device
        solution = self._ensure_device(solution)
        
        # Create a random action (k indices to swap)
        selected_indices = torch.randint(0, graph_size, (batch_size, k), device=self.device)
        
        # For simplicity, we'll just swap pairs of these indices
        for i in range(0, k, 2):
            if i+1 < k:
                # Get nodes at selected indices
                idx1 = selected_indices[:, i]
                idx2 = selected_indices[:, i+1]
                
                # Find successors
                next1 = solution.gather(1, idx1.unsqueeze(1)).squeeze(1)
                next2 = solution.gather(1, idx2.unsqueeze(1)).squeeze(1)
                
                # Swap connections (2-opt style)
                solution_new = solution.clone()
                solution_new.scatter_(1, idx1.unsqueeze(1), next2.unsqueeze(1))
                solution_new.scatter_(1, idx2.unsqueeze(1), next1.unsqueeze(1))
                
                # Update solution if it's valid (sometimes these swaps can break tours)
                try:
                    self.problem.check_feasibility(solution_new)
                    solution = solution_new
                except:
                    pass  # Keep original solution if swap creates invalid tour
        
        return solution
    
    def improved_two_opt(self, solution, batch):
        """
        Improved 2-opt implementation that systematically tries all possible edge swaps
        and keeps the best improvement.
        """
        batch_size, graph_size = solution.size()
        solution = self._ensure_device(solution)
        
        # Get current solution order as a route (easier to manipulate)
        route = self.problem.get_order(solution, return_solution=True)
        
        # Get current cost
        current_cost = self.problem.get_costs(batch, solution)
        
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # Try all possible 2-opt swaps
            for i in range(graph_size):
                for j in range(i+2, graph_size):
                    if i == 0 and j == graph_size-1:
                        continue  # Skip invalid swap
                    
                    # Create new route after 2-opt swap
                    new_route = route.clone()
                    # Extract the segment and reverse it using flip
                    segment = route[0, i+1:j+1].clone()
                    new_route[0, i+1:j+1] = segment.flip(0)
                    
                    # Convert route to solution format
                    new_solution = torch.zeros_like(solution)
                    for k in range(graph_size):
                        idx = (k + 1) % graph_size
                        new_solution[0, new_route[0, k]] = new_route[0, idx]
                    
                    # Compute new cost
                    new_cost = self.problem.get_costs(batch, new_solution)
                    
                    # If better, update
                    delta = current_cost - new_cost
                    if delta > best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            # Apply best swap if improvement found
            if best_delta > 0:
                improved = True
                # Apply the best swap by extracting and flipping the segment
                segment = route[0, best_i+1:best_j+1].clone()
                route[0, best_i+1:best_j+1] = segment.flip(0)
                
                # Convert route to solution format
                for k in range(graph_size):
                    idx = (k + 1) % graph_size
                    solution[0, route[0, k]] = route[0, idx]
                
                current_cost -= best_delta
        
        return solution
    
    def solve(self, tsp_instance):
        """
        Solve a TSP instance using the original ABC algorithm
        """
        # Convert to batch format and move to device
        batch = {'coordinates': tsp_instance['coordinates'].unsqueeze(0)}
        batch = self._ensure_device(batch)
        
        # Initialize employed bees (food sources)
        employed_bees = []
        trial_counters = []
        
        # Generate initial solutions for all employed bees
        for _ in range(self.colony_size):
            solution = self.problem.get_initial_solutions(batch)
            # Make sure solution is on the correct device
            solution = self._ensure_device(solution)
            cost = self.problem.get_costs(batch, solution)
            
            employed_bees.append({'solution': solution.clone(), 'cost': cost.item()})
            trial_counters.append(0)
        
        # Best solution overall
        best_solution = employed_bees[0]['solution'].clone()
        best_cost = employed_bees[0]['cost']
        
        for i in range(self.colony_size):
            if employed_bees[i]['cost'] < best_cost:
                best_cost = employed_bees[i]['cost']
                best_solution = employed_bees[i]['solution'].clone()
        
        # Use 2-opt to improve the best initial solution
        best_solution = self.improved_two_opt(best_solution, batch)
        best_cost = self.problem.get_costs(batch, best_solution).item()
        
        # For tracking convergence
        iteration_costs = [best_cost]
        
        # Keep track of stagnation
        last_improvement = 0
        stagnation_count = 0
        improvement_iterations = []
        
        # Main ABC loop
        for iteration in tqdm(range(self.max_iterations), desc="Original ABC"):
            improved_this_iteration = False
            
            # EMPLOYED BEE PHASE
            for i in range(self.colony_size):
                # Apply improved two-opt instead of random k-opt
                current_solution = self._ensure_device(employed_bees[i]['solution'])
                new_solution = self.improved_two_opt(current_solution.clone(), batch)
                new_cost = self.problem.get_costs(batch, new_solution)
                
                # Greedy selection
                if new_cost < employed_bees[i]['cost']:
                    employed_bees[i]['solution'] = new_solution.clone()
                    employed_bees[i]['cost'] = new_cost.item()
                    trial_counters[i] = 0
                    
                    # Update global best if needed
                    if new_cost < best_cost:
                        best_cost = new_cost.item()
                        best_solution = new_solution.clone()
                        improved_this_iteration = True
                else:
                    trial_counters[i] += 1
            
            # ONLOOKER BEE PHASE
            # Calculate selection probabilities based on fitness
            total_fitness = sum(1.0 / (1.0 + bee['cost']) for bee in employed_bees)
            probabilities = [(1.0 / (1.0 + bee['cost'])) / total_fitness for bee in employed_bees]
            
            # Onlooker bees select food sources
            for _ in range(self.colony_size):
                # Select a food source using roulette wheel selection
                selected_idx = np.random.choice(range(self.colony_size), p=probabilities)
                
                # Apply improved two-opt
                current_solution = self._ensure_device(employed_bees[selected_idx]['solution'])
                new_solution = self.improved_two_opt(current_solution.clone(), batch)
                new_cost = self.problem.get_costs(batch, new_solution)
                
                # Greedy selection
                if new_cost < employed_bees[selected_idx]['cost']:
                    employed_bees[selected_idx]['solution'] = new_solution.clone()
                    employed_bees[selected_idx]['cost'] = new_cost.item()
                    trial_counters[selected_idx] = 0
                    
                    # Update global best if needed
                    if new_cost < best_cost:
                        best_cost = new_cost.item()
                        best_solution = new_solution.clone()
                        improved_this_iteration = True
                else:
                    trial_counters[selected_idx] += 1
            
            # SCOUT BEE PHASE
            for i in range(self.colony_size):
                if trial_counters[i] >= self.limit:
                    # Generate a new random solution and apply 2-opt to it
                    solution = self.problem.get_initial_solutions(batch)
                    solution = self._ensure_device(solution)  # Ensure device consistency
                    solution = self.improved_two_opt(solution, batch)  # Apply 2-opt to improve it
                    cost = self.problem.get_costs(batch, solution)
                    
                    employed_bees[i]['solution'] = solution.clone()
                    employed_bees[i]['cost'] = cost.item()
                    trial_counters[i] = 0
                    
                    # Update global best if needed
                    if cost < best_cost:
                        best_cost = cost.item()
                        best_solution = solution.clone()
                        improved_this_iteration = True
            
            # Track best cost for this iteration
            iteration_costs.append(best_cost)
            
            # Track improvement iterations
            if improved_this_iteration:
                improvement_iterations.append(iteration)
                last_improvement = iteration
                stagnation_count = 0
            else:
                stagnation_count += 1
        
        # Calculate metrics
        metrics = {
            'stagnation_count': stagnation_count,
            'last_improvement': last_improvement,
            'improvement_iterations': improvement_iterations,
            'solution_quality': best_cost,
            'convergence_curve': iteration_costs
        }
        
        # Return best solution and metrics
        return best_solution.squeeze(0), best_cost, iteration_costs, metrics


def run_comparison(dataset, num_samples, colony_size, max_iterations, limit, output_dir, model_path='pre-trained/tsp20.pt'):
    """
    Run comparison between original ABC and NeuOpt-enhanced ABC
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set random seeds for reproducibility
    torch.manual_seed(6666)
    np.random.seed(6666)
    random.seed(6666)
    
    # Load TSP dataset
    dataset = load_tsp_dataset(dataset, num_samples=num_samples)
    
    # Initialize both algorithms
    abc_original = ABC_Original(
        colony_size=colony_size,
        max_iterations=max_iterations,
        limit=limit
    )
    
    abc_neuopt = ABC_NeuOpt(
        colony_size=colony_size,
        max_iterations=max_iterations,
        limit=limit,
        model_path=model_path
    )
    
    # Results storage
    results = {
        'original': {
            'costs': [], 
            'times': [], 
            'convergence': [],
            'convergence_rate': [],
            'stagnation_counts': [],
            'last_improvements': [],
            'improvement_iterations': [],
            'time_to_best': []
        },
        'neuopt': {
            'costs': [], 
            'times': [], 
            'convergence': [],
            'convergence_rate': [],
            'stagnation_counts': [],
            'last_improvements': [],
            'improvement_iterations': [],
            'time_to_best': []
        }
    }
    
    # Store instance details for analysis
    instance_data = []
    
    # Solve each instance with both algorithms
    for i in range(len(dataset)):
        print(f"\n{'='*50}")
        print(f"Solving TSP instance {i+1}/{len(dataset)}")
        print(f"{'='*50}")
        
        instance = dataset[i]
        instance_device = {'coordinates': instance['coordinates'].to(abc_original.device)}
        
        # Extract instance features for analysis
        coords = instance['coordinates'].cpu().numpy()
        
        # Calculate instance features
        x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
        y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
        area = x_range * y_range
        avg_distance = np.mean([
            np.sqrt(np.sum((coords[i] - coords[j])**2)) 
            for i in range(len(coords)) for j in range(i+1, len(coords))
        ])
        
        # Solve with Original ABC
        print("\nOriginal ABC:")
        start_time = time.time()
        solution_orig, cost_orig, convergence_orig, metrics_orig = abc_original.solve(instance_device)
        time_orig = time.time() - start_time
        
        # Calculate time to best solution
        if metrics_orig['last_improvement'] > 0:
            time_to_best_orig = (time_orig / max_iterations) * metrics_orig['last_improvement']
        else:
            time_to_best_orig = time_orig
        
        # Calculate convergence rate (avg improvement per iteration)
        if len(metrics_orig['improvement_iterations']) > 1:
            improvements = [convergence_orig[0] - convergence_orig[it+1] for it in metrics_orig['improvement_iterations'][:-1]]
            convergence_rate_orig = np.mean(improvements) if improvements else 0
        else:
            convergence_rate_orig = 0
        
        # Make sure solution is on CPU before passing to get_order (to avoid device issues)
        solution_cpu = solution_orig.cpu().unsqueeze(0)
        route_orig = abc_original.problem.get_order(solution_cpu, return_solution=True).squeeze(0)
        print(f"Best tour cost: {cost_orig:.6f}")
        print(f"Time taken: {time_orig:.2f} seconds")
        print(f"Stagnation count: {metrics_orig['stagnation_count']}")
        print(f"Last improvement at iteration: {metrics_orig['last_improvement']}")
        
        # Solve with NeuOpt ABC
        print("\nNeuOpt-enhanced ABC:")
        start_time = time.time()
        solution_neuopt, cost_neuopt, convergence_neuopt, metrics_neuopt = abc_neuopt.solve(instance_device)
        time_neuopt = time.time() - start_time
        
        # Calculate time to best solution
        if metrics_neuopt['last_improvement'] > 0:
            time_to_best_neuopt = (time_neuopt / max_iterations) * metrics_neuopt['last_improvement']
        else:
            time_to_best_neuopt = time_neuopt
        
        # Calculate convergence rate (avg improvement per iteration)
        if len(metrics_neuopt['improvement_iterations']) > 1:
            improvements = [convergence_neuopt[0] - convergence_neuopt[it+1] for it in metrics_neuopt['improvement_iterations'][:-1]]
            convergence_rate_neuopt = np.mean(improvements) if improvements else 0
        else:
            convergence_rate_neuopt = 0
        
        # Make sure solution is on CPU before passing to get_order
        solution_cpu = solution_neuopt.cpu().unsqueeze(0)
        route_neuopt = abc_neuopt.problem.get_order(solution_cpu, return_solution=True).squeeze(0)
        print(f"Best tour cost: {cost_neuopt:.6f}")
        print(f"Time taken: {time_neuopt:.2f} seconds")
        print(f"Stagnation count: {metrics_neuopt['stagnation_count']}")
        print(f"Last improvement at iteration: {metrics_neuopt['last_improvement']}")
        
        # Store results
        results['original']['costs'].append(cost_orig)
        results['original']['times'].append(time_orig)
        results['original']['convergence'].append(convergence_orig)
        results['original']['stagnation_counts'].append(metrics_orig['stagnation_count'])
        results['original']['last_improvements'].append(metrics_orig['last_improvement'])
        results['original']['improvement_iterations'].append(metrics_orig['improvement_iterations'])
        results['original']['convergence_rate'].append(convergence_rate_orig)
        results['original']['time_to_best'].append(time_to_best_orig)
        
        results['neuopt']['costs'].append(cost_neuopt)
        results['neuopt']['times'].append(time_neuopt)
        results['neuopt']['convergence'].append(convergence_neuopt)
        results['neuopt']['stagnation_counts'].append(metrics_neuopt['stagnation_count'])
        results['neuopt']['last_improvements'].append(metrics_neuopt['last_improvement'])
        results['neuopt']['improvement_iterations'].append(metrics_neuopt['improvement_iterations'])
        results['neuopt']['convergence_rate'].append(convergence_rate_neuopt)
        results['neuopt']['time_to_best'].append(time_to_best_neuopt)
        
        # Store instance data for correlation analysis
        instance_data.append({
            'instance_id': i+1,
            'area': area,
            'avg_distance': avg_distance,
            'x_range': x_range,
            'y_range': y_range,
            'improvement': ((cost_orig - cost_neuopt) / cost_orig) * 100
        })
        
        # Calculate improvement
        improvement = ((cost_orig - cost_neuopt) / cost_orig) * 100
        print(f"\nImprovement: {improvement:.2f}%")
        
        # Plot convergence for this instance
        plt.figure(figsize=(12, 8))
        plt.plot(convergence_orig, label='Original ABC', color='blue', linestyle='-')
        plt.plot(convergence_neuopt, label='NeuOpt-enhanced ABC', color='red', linestyle='-')
        
        # Mark improvement iterations
        for it in metrics_orig['improvement_iterations']:
            plt.axvline(x=it, color='blue', linestyle='--', alpha=0.3)
        for it in metrics_neuopt['improvement_iterations']:
            plt.axvline(x=it, color='red', linestyle='--', alpha=0.3)
            
        plt.xlabel('Iteration')
        plt.ylabel('Tour Cost')
        plt.title(f'Convergence Comparison - Instance {i+1}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/instance_{i+1}_convergence.png")
        plt.close()
        
        # Create animated convergence plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get max iterations across both algorithms
        max_iter = max(len(convergence_orig), len(convergence_neuopt))
        
        line_orig, = ax.plot([], [], 'b-', label='Original ABC')
        line_neuopt, = ax.plot([], [], 'r-', label='NeuOpt ABC')
        
        # Add text annotations for current best
        text_orig = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='blue')
        text_neuopt = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='red')
        iter_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        
        # Set up the animation
        def init():
            ax.set_xlim(0, max_iter)
            ax.set_ylim(min(min(convergence_orig), min(convergence_neuopt)) * 0.95, 
                        max(convergence_orig[0], convergence_neuopt[0]) * 1.05)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Tour Cost')
            ax.set_title(f'Convergence Animation - Instance {i+1}')
            ax.grid(True)
            ax.legend(loc='upper right')
            return line_orig, line_neuopt, text_orig, text_neuopt, iter_text
        
        def update(frame):
            # Update original ABC line
            if frame < len(convergence_orig):
                x_orig = list(range(frame + 1))
                y_orig = convergence_orig[:frame + 1]
                line_orig.set_data(x_orig, y_orig)
                text_orig.set_text(f'Original ABC: {convergence_orig[frame]:.4f}')
            
            # Update NeuOpt ABC line
            if frame < len(convergence_neuopt):
                x_neuopt = list(range(frame + 1))
                y_neuopt = convergence_neuopt[:frame + 1]
                line_neuopt.set_data(x_neuopt, y_neuopt)
                text_neuopt.set_text(f'NeuOpt ABC: {convergence_neuopt[frame]:.4f}')
            
            iter_text.set_text(f'Iteration: {frame}')
            return line_orig, line_neuopt, text_orig, text_neuopt, iter_text
        
        # Create animation (skip frames to make file smaller)
        skip = max(1, max_iter // 100)  # Limit to about 100 frames
        ani = FuncAnimation(fig, update, frames=range(0, max_iter, skip), 
                            init_func=init, blit=True, repeat=False)
        
        # Save animation
        ani.save(f"{output_dir}/instance_{i+1}_convergence_animation.gif", writer='pillow', fps=10)
        plt.close()
        
        # Visualize the tours with more detail
        plt.figure(figsize=(15, 7))
        
        # Plot original ABC tour
        plt.subplot(1, 2, 1)
        coords = instance['coordinates'].cpu().numpy()
        route_idx = route_orig.cpu().numpy()
        plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=50)
        
        # Draw the tour with direction arrows
        for j in range(len(route_idx)):
            start = route_idx[j]
            end = route_idx[(j+1) % len(route_idx)]
            plt.arrow(coords[start, 0], coords[start, 1], 
                     coords[end, 0] - coords[start, 0], 
                     coords[end, 1] - coords[start, 1],
                     head_width=0.02, head_length=0.03, 
                     fc='black', ec='black', alpha=0.7)
        
        # Add node numbers
        for j in range(len(coords)):
            plt.text(coords[j, 0], coords[j, 1], str(j), fontsize=8)
            
        plt.title(f'Original ABC\nCost: {cost_orig:.4f}')
        
        # Plot NeuOpt-enhanced ABC tour
        plt.subplot(1, 2, 2)
        route_idx = route_neuopt.cpu().numpy()
        plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=50)
        
        # Draw the tour with direction arrows
        for j in range(len(route_idx)):
            start = route_idx[j]
            end = route_idx[(j+1) % len(route_idx)]
            plt.arrow(coords[start, 0], coords[start, 1], 
                     coords[end, 0] - coords[start, 0], 
                     coords[end, 1] - coords[start, 1],
                     head_width=0.02, head_length=0.03, 
                     fc='black', ec='black', alpha=0.7)
            
        # Add node numbers
        for j in range(len(coords)):
            plt.text(coords[j, 0], coords[j, 1], str(j), fontsize=8)
            
        plt.title(f'NeuOpt-enhanced ABC\nCost: {cost_neuopt:.4f}')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/instance_{i+1}_tours.png")
        plt.close()
    
    # Create a dataframe for analysis
    df_instances = pd.DataFrame(instance_data)
    
    # Statistical analysis
    t_stat, p_value = stats.ttest_rel(results['original']['costs'], results['neuopt']['costs'])
    effect_size = (np.mean(results['original']['costs']) - np.mean(results['neuopt']['costs'])) / np.std(results['original']['costs'])
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    print("\nOriginal ABC:")
    print(f"Average tour cost: {np.mean(results['original']['costs']):.6f} ± {np.std(results['original']['costs']):.6f}")
    print(f"Average time: {np.mean(results['original']['times']):.2f} seconds")
    print(f"Best tour cost: {np.min(results['original']['costs']):.6f}")
    print(f"Average time to best solution: {np.mean(results['original']['time_to_best']):.2f} seconds")
    print(f"Average convergence rate: {np.mean(results['original']['convergence_rate']):.6f}")
    print(f"Average iterations to last improvement: {np.mean(results['original']['last_improvements']):.2f}")
    
    print("\nNeuOpt-enhanced ABC:")
    print(f"Average tour cost: {np.mean(results['neuopt']['costs']):.6f} ± {np.std(results['neuopt']['costs']):.6f}")
    print(f"Average time: {np.mean(results['neuopt']['times']):.2f} seconds")
    print(f"Best tour cost: {np.min(results['neuopt']['costs']):.6f}")
    print(f"Average time to best solution: {np.mean(results['neuopt']['time_to_best']):.2f} seconds")
    print(f"Average convergence rate: {np.mean(results['neuopt']['convergence_rate']):.6f}")
    print(f"Average iterations to last improvement: {np.mean(results['neuopt']['last_improvements']):.2f}")
    
    # Calculate overall improvement
    avg_improvement = ((np.mean(results['original']['costs']) - np.mean(results['neuopt']['costs'])) / 
                       np.mean(results['original']['costs'])) * 100
    time_improvement = ((np.mean(results['original']['time_to_best']) - np.mean(results['neuopt']['time_to_best'])) / 
                       np.mean(results['original']['time_to_best'])) * 100
    print(f"\nAverage cost improvement: {avg_improvement:.2f}%")
    print(f"Average time to best solution improvement: {time_improvement:.2f}%")
    print(f"Statistical significance: p-value = {p_value:.6f}, effect size = {effect_size:.4f}")
    
    # Create more detailed visualizations
    
    # 1. Box plot comparison
    plt.figure(figsize=(12, 8))
    data = [results['original']['costs'], results['neuopt']['costs']]
    sns.boxplot(data=data)
    plt.xticks([0, 1], ['Original ABC', 'NeuOpt ABC'])
    plt.ylabel('Tour Cost')
    plt.title('Distribution of Tour Costs')
    plt.grid(axis='y')
    plt.savefig(f"{output_dir}/boxplot_comparison.png")
    plt.close()
    
    # 2. Violin plot comparison
    plt.figure(figsize=(12, 8))
    data = {
        'Algorithm': ['Original ABC'] * len(results['original']['costs']) + ['NeuOpt ABC'] * len(results['neuopt']['costs']),
        'Cost': results['original']['costs'] + results['neuopt']['costs']
    }
    df = pd.DataFrame(data)
    sns.violinplot(x='Algorithm', y='Cost', data=df)
    plt.title('Violin Plot of Tour Costs')
    plt.grid(axis='y')
    plt.savefig(f"{output_dir}/violinplot_comparison.png")
    plt.close()
    
    # 3. Convergence comparison across all instances
    plt.figure(figsize=(15, 10))
    
    # Normalize convergence curves
    normalized_orig = []
    normalized_neuopt = []
    
    # Find minimum length of convergence arrays to ensure consistent shapes
    min_len_orig = min(len(conv) for conv in results['original']['convergence'])
    min_len_neuopt = min(len(conv) for conv in results['neuopt']['convergence'])
    
    # Use a common minimum length for fair comparison
    common_min_len = min(min_len_orig, min_len_neuopt)
    
    for i in range(len(results['original']['convergence'])):
        # Truncate to common minimum length
        orig = np.array(results['original']['convergence'][i][:common_min_len])
        neuopt = np.array(results['neuopt']['convergence'][i][:common_min_len])
        
        # Normalize to starting cost
        normalized_orig.append(orig / orig[0])
        normalized_neuopt.append(neuopt / neuopt[0])
    
    # Plot average normalized convergence
    avg_orig = np.mean(normalized_orig, axis=0)
    avg_neuopt = np.mean(normalized_neuopt, axis=0)
    std_orig = np.std(normalized_orig, axis=0)
    std_neuopt = np.std(normalized_neuopt, axis=0)
    
    iterations = range(len(avg_orig))
    plt.plot(iterations, avg_orig, 'b-', label='Original ABC')
    plt.fill_between(iterations, avg_orig - std_orig, avg_orig + std_orig, color='blue', alpha=0.2)
    plt.plot(iterations, avg_neuopt, 'r-', label='NeuOpt ABC')
    plt.fill_between(iterations, avg_neuopt - std_neuopt, avg_neuopt + std_neuopt, color='red', alpha=0.2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Tour Cost')
    plt.title('Average Convergence Across All Instances')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/average_convergence.png")
    plt.close()
    
    # 4. Correlation between instance features and improvement
    plt.figure(figsize=(14, 10))
    
    # Correlation matrix
    corr = df_instances.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Between Instance Features and Improvement')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
    
    # 5. Scatter plots for feature relationships
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.scatterplot(x='area', y='improvement', data=df_instances, ax=axes[0, 0])
    axes[0, 0].set_title('Area vs Improvement')
    axes[0, 0].grid(True)
    
    sns.scatterplot(x='avg_distance', y='improvement', data=df_instances, ax=axes[0, 1])
    axes[0, 1].set_title('Average Distance vs Improvement')
    axes[0, 1].grid(True)
    
    sns.scatterplot(x='x_range', y='improvement', data=df_instances, ax=axes[1, 0])
    axes[1, 0].set_title('X Range vs Improvement')
    axes[1, 0].grid(True)
    
    sns.scatterplot(x='y_range', y='improvement', data=df_instances, ax=axes[1, 1])
    axes[1, 1].set_title('Y Range vs Improvement')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_vs_improvement.png")
    plt.close()
    
    # 6. Bar chart of metrics comparison
    plt.figure(figsize=(15, 10))
    
    # Calculate normalized metrics for fair comparison
    metrics = {
        'Cost': [np.mean(results['original']['costs']), np.mean(results['neuopt']['costs'])],
        'Time': [np.mean(results['original']['times']), np.mean(results['neuopt']['times'])],
        'Time to Best': [np.mean(results['original']['time_to_best']), np.mean(results['neuopt']['time_to_best'])],
        'Last Improvement': [np.mean(results['original']['last_improvements']), np.mean(results['neuopt']['last_improvements'])],
        'Convergence Rate': [np.mean(results['original']['convergence_rate']), np.mean(results['neuopt']['convergence_rate'])]
    }
    
    # Normalize metrics (lower is better for all except convergence rate)
    normalized_metrics = {}
    for metric, values in metrics.items():
        if metric == 'Convergence Rate':
            # Higher is better
            max_val = max(values)
            normalized_metrics[metric] = [v / max_val for v in values]
        else:
            # Lower is better
            max_val = max(values)
            normalized_metrics[metric] = [1 - (v / max_val) for v in values]
    
    # Create dataframe for plotting
    df_metrics = pd.DataFrame({
        'Metric': list(normalized_metrics.keys()) * 2,
        'Algorithm': ['Original ABC'] * len(normalized_metrics) + ['NeuOpt ABC'] * len(normalized_metrics),
        'Value': sum([[v[0] for v in normalized_metrics.values()], [v[1] for v in normalized_metrics.values()]], [])
    })
    
    sns.barplot(x='Metric', y='Value', hue='Algorithm', data=df_metrics)
    plt.title('Normalized Performance Metrics (Higher is Better)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/normalized_metrics.png")
    plt.close()
    
    # 7. Radar chart for comprehensive comparison
    plt.figure(figsize=(10, 10))
    
    # Prepare data
    categories = list(normalized_metrics.keys())
    N = len(categories)
    
    # Set angle
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Values for original ABC
    values_orig = [normalized_metrics[metric][0] for metric in categories]
    values_orig += values_orig[:1]
    
    # Values for NeuOpt ABC
    values_neuopt = [normalized_metrics[metric][1] for metric in categories]
    values_neuopt += values_neuopt[:1]
    
    # Set up radar chart
    ax = plt.subplot(111, polar=True)
    
    # Plot values
    ax.plot(angles, values_orig, 'b-', linewidth=2, label='Original ABC')
    ax.fill(angles, values_orig, 'blue', alpha=0.1)
    
    ax.plot(angles, values_neuopt, 'r-', linewidth=2, label='NeuOpt ABC')
    ax.fill(angles, values_neuopt, 'red', alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories)
    
    # Draw axis lines for each angle and label
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=8)
    plt.ylim(0, 1)
    
    plt.title('Performance Radar Chart (Higher is Better)')
    plt.legend(loc='upper right')
    
    plt.savefig(f"{output_dir}/radar_chart.png")
    plt.close()
    
    # Save detailed results to file
    with open(f"{output_dir}/results.txt", 'w') as f:
        f.write("="*50 + "\n")
        f.write("COMPREHENSIVE COMPARISON RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write("Original ABC:\n")
        f.write(f"Average tour cost: {np.mean(results['original']['costs']):.6f} ± {np.std(results['original']['costs']):.6f}\n")
        f.write(f"Average time: {np.mean(results['original']['times']):.2f} seconds\n")
        f.write(f"Best tour cost: {np.min(results['original']['costs']):.6f}\n")
        f.write(f"Average time to best solution: {np.mean(results['original']['time_to_best']):.2f} seconds\n")
        f.write(f"Average convergence rate: {np.mean(results['original']['convergence_rate']):.6f}\n")
        f.write(f"Average iterations to last improvement: {np.mean(results['original']['last_improvements']):.2f}\n\n")
        
        f.write("NeuOpt-enhanced ABC:\n")
        f.write(f"Average tour cost: {np.mean(results['neuopt']['costs']):.6f} ± {np.std(results['neuopt']['costs']):.6f}\n")
        f.write(f"Average time: {np.mean(results['neuopt']['times']):.2f} seconds\n")
        f.write(f"Best tour cost: {np.min(results['neuopt']['costs']):.6f}\n")
        f.write(f"Average time to best solution: {np.mean(results['neuopt']['time_to_best']):.2f} seconds\n")
        f.write(f"Average convergence rate: {np.mean(results['neuopt']['convergence_rate']):.6f}\n")
        f.write(f"Average iterations to last improvement: {np.mean(results['neuopt']['last_improvements']):.2f}\n\n")
        
        f.write(f"Average cost improvement: {avg_improvement:.2f}%\n")
        f.write(f"Average time to best solution improvement: {time_improvement:.2f}%\n")
        f.write(f"Statistical significance: p-value = {p_value:.6f}, effect size = {effect_size:.4f}\n\n")
        
        f.write("Instance details:\n")
        for i in range(len(dataset)):
            f.write(f"Instance {i+1}:\n")
            f.write(f"  Original ABC: cost={results['original']['costs'][i]:.6f}, time={results['original']['times'][i]:.2f}s, " + 
                    f"time to best={results['original']['time_to_best'][i]:.2f}s, " +
                    f"last improvement={results['original']['last_improvements'][i]} iter\n")
                    
            f.write(f"  NeuOpt ABC: cost={results['neuopt']['costs'][i]:.6f}, time={results['neuopt']['times'][i]:.2f}s, " +
                    f"time to best={results['neuopt']['time_to_best'][i]:.2f}s, " +
                    f"last improvement={results['neuopt']['last_improvements'][i]} iter\n")
                    
            improvement = ((results['original']['costs'][i] - results['neuopt']['costs'][i]) / 
                           results['original']['costs'][i]) * 100
            f.write(f"  Improvement: {improvement:.2f}%\n\n")
            
        # Write correlation information
        f.write("\nCorrelation between instance features and improvement:\n")
        f.write(corr.loc['improvement'].to_string() + "\n\n")
    
    # Generate LaTeX table for publication
    with open(f"{output_dir}/latex_table.tex", 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison between Original ABC and NeuOpt-enhanced ABC}\n")
        f.write("\\begin{tabular}{lrr}\n")
        f.write("\\hline\n")
        f.write("Metric & Original ABC & NeuOpt ABC \\\\\n")
        f.write("\\hline\n")
        f.write(f"Average Tour Cost & {np.mean(results['original']['costs']):.4f} $\\pm$ {np.std(results['original']['costs']):.4f} & {np.mean(results['neuopt']['costs']):.4f} $\\pm$ {np.std(results['neuopt']['costs']):.4f} \\\\\n")
        f.write(f"Best Tour Cost & {np.min(results['original']['costs']):.4f} & {np.min(results['neuopt']['costs']):.4f} \\\\\n")
        f.write(f"Average Time (s) & {np.mean(results['original']['times']):.2f} & {np.mean(results['neuopt']['times']):.2f} \\\\\n")
        f.write(f"Time to Best Solution (s) & {np.mean(results['original']['time_to_best']):.2f} & {np.mean(results['neuopt']['time_to_best']):.2f} \\\\\n")
        f.write(f"Convergence Rate & {np.mean(results['original']['convergence_rate']):.6f} & {np.mean(results['neuopt']['convergence_rate']):.6f} \\\\\n")
        f.write(f"Iterations to Last Improvement & {np.mean(results['original']['last_improvements']):.2f} & {np.mean(results['neuopt']['last_improvements']):.2f} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Improvement & \\multicolumn{{2}}{{c}}{{\\textbf{{{avg_improvement:.2f}\\%}}}} \\\\\n")
        f.write(f"p-value & \\multicolumn{{2}}{{c}}{{\\textbf{{{p_value:.6f}}}}} \\\\\n")
        f.write(f"Effect Size & \\multicolumn{{2}}{{c}}{{\\textbf{{{effect_size:.4f}}}}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        
    # Generate CSV file for further analysis
    df_results = pd.DataFrame({
        'Instance': range(1, len(dataset) + 1),
        'Original_Cost': results['original']['costs'],
        'NeuOpt_Cost': results['neuopt']['costs'],
        'Original_Time': results['original']['times'],
        'NeuOpt_Time': results['neuopt']['times'],
        'Original_TimeToBest': results['original']['time_to_best'],
        'NeuOpt_TimeToBest': results['neuopt']['time_to_best'],
        'Original_LastImprovement': results['original']['last_improvements'],
        'NeuOpt_LastImprovement': results['neuopt']['last_improvements'],
        'Original_ConvergenceRate': results['original']['convergence_rate'],
        'NeuOpt_ConvergenceRate': results['neuopt']['convergence_rate'],
        'Improvement': [(results['original']['costs'][i] - results['neuopt']['costs'][i]) / results['original']['costs'][i] * 100 
                        for i in range(len(results['original']['costs']))],
        'Area': df_instances['area'],
        'Avg_Distance': df_instances['avg_distance'],
        'X_Range': df_instances['x_range'],
        'Y_Range': df_instances['y_range']
    })
    
    df_results.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    
    print(f"\nAll results saved to {output_dir}/")
    print(f"View {output_dir}/results.txt for detailed analysis.")


def main():
    parser = argparse.ArgumentParser(description="Compare Original ABC vs NeuOpt-enhanced ABC for TSP")
    parser.add_argument('--dataset', type=str, default='datasets/tsp_20.pkl', help='TSP dataset file')
    parser.add_argument('--model', type=str, default='pre-trained/tsp20.pt', help='Pretrained NeuOpt model')
    parser.add_argument('--colony_size', type=int, default=30, help='Colony size')
    parser.add_argument('--max_iterations', type=int, default=50, help='Maximum iterations')
    parser.add_argument('--limit', type=int, default=15, help='Trial limit for scout bees')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of TSP instances to solve')
    parser.add_argument('--output_dir', type=str, default='comparison_results', help='Output directory for results')
    parser.add_argument('--seed', type=int, default=6666, help='Random seed for reproducibility')
    parser.add_argument('--save_animations', action='store_true', help='Save convergence animations (can be slow)')
    
    args = parser.parse_args()
    
    # Set seed from args
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    run_comparison(
        dataset=args.dataset,
        num_samples=args.num_samples,
        colony_size=args.colony_size,
        max_iterations=args.max_iterations,
        limit=args.limit,
        output_dir=args.output_dir,
        model_path=args.model
    )


if __name__ == "__main__":
    main() 