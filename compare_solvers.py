import torch
import numpy as np
import time
import math
import os
import argparse
import sys
import json
from datetime import datetime
from typing import List, Dict, Tuple, Union
import copy

# Optional matplotlib import for visualizations
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Visualizations will be disabled.")

# Import NeuOpt components
# Assuming NeuOpt code is in the python path or current directory structure
try:
    from problems.problem_tsp import TSP
    # from problems.problem_cvrp import CVRP # Uncomment if needed
    from agent.ppo import PPO
    from options import get_options
    # Import the Hybrid ACO class from the existing file
    from hybrid_aco_neuopt import ACO as HybridACO
except ImportError as e:
    print(f"Error importing NeuOpt components: {e}")
    print("Please ensure the NeuOpt project directory is in your PYTHONPATH or accessible.")
    sys.exit(1)


def parse_arguments():
    """
    Parse command line arguments for the comparison script.
    """
    parser = argparse.ArgumentParser(description='Compare Hybrid ACO-NeuOpt, Standard ACO, and NeuOpt solvers')

    # --- Problem Parameters ---
    parser.add_argument('--problem', type=str, default='tsp', choices=['tsp'], help='Problem type (currently only tsp supported)')
    parser.add_argument('--graph_size', type=int, default=20, help='Problem size (number of cities)')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of independent runs (instances) to average over')
    parser.add_argument('--seed', type=int, default=1234, help='Base random seed')

    # --- ACO Parameters (Shared by Hybrid and Standard) ---
    parser.add_argument('--num_ants', type=int, default=20, help='Number of ants') # Reduced default for faster comparison
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Pheromone evaporation rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='Pheromone influence factor')
    parser.add_argument('--beta', type=float, default=2.0, help='Heuristic influence factor')
    parser.add_argument('--q0', type=float, default=0.9, help='Exploration/exploitation balance parameter')
    parser.add_argument('--aco_iterations', type=int, default=50, help='Maximum number of ACO iterations') # Reduced default

    # --- NeuOpt Specific Parameters ---
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained NeuOpt model')
    parser.add_argument('--local_search_steps', type=int, default=20, help='Number of NeuOpt steps per solution in Hybrid ACO')
    parser.add_argument('--neuopt_steps', type=int, default=1000, help='Number of steps for standalone NeuOpt run (approx equiv. to Hybrid)') # aco_iterations * local_search_steps

    # --- Optimization Parameters ---
    parser.add_argument('--elite_percentage', type=float, default=0.25, help='Percentage of solutions to apply local search to')
    parser.add_argument('--early_stop_threshold', type=int, default=5, help='Stop local search after N non-improving steps')
    parser.add_argument('--adaptive_search', action='store_true', default=True, help='Increase search depth as iterations progress')
    parser.add_argument('--parallel_processing', action='store_true', default=True, help='Use parallel processing for local search')
    parser.add_argument('--progressive_elite', action='store_true', default=True, help='Gradually increase elite percentage')

    # --- Solver Selection ---
    parser.add_argument('--run_hybrid', action='store_true', default=True, help='Run Hybrid ACO-NeuOpt')
    parser.add_argument('--run_standard_aco', action='store_true', default=True, help='Run Standard ACO')
    parser.add_argument('--run_neuopt', action='store_true', default=True, help='Run Standalone NeuOpt')
    
    # --- Output Options ---
    parser.add_argument('--save_results', action='store_true', default=False, help='Save results to a JSON file')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', default=False, help='Generate performance visualizations')

    # Need to handle NeuOpt's internal args differently
    # Parse known args first, then pass remaining to get_options
    args, unknown = parser.parse_known_args()

    # Load NeuOpt options, providing an empty list to avoid re-parsing standard args
    # We'll override necessary settings later
    original_argv = sys.argv.copy()
    sys.argv = [original_argv[0]] # Keep only the script name
    opts = get_options([])
    sys.argv = original_argv # Restore original argv

    # Apply relevant settings from our args to NeuOpt opts
    opts.load_path = args.model_path
    opts.graph_size = args.graph_size
    opts.problem = args.problem
    opts.seed = args.seed # NeuOpt uses its own seed handling internally, but set base here
    opts.eval_only = True

    return args, opts


# --- Placeholder for StandardACO ---
class StandardACO:
    """
    Standard Ant Colony Optimization framework (without NeuOpt local search).
    Based on the HybridACO class.
    """
    def __init__(self, problem, instance_batch, num_ants=50, decay_rate=0.1, alpha=1, beta=2, q0=0.9,
                 initial_pheromone=1.0, min_pheromone=0.1, max_pheromone=10.0, device='cpu'):
        """
        Initialize Standard ACO.
        Args:
            problem: The VRP instance (e.g., TSP or CVRP object from NeuOpt).
            instance_batch: The specific problem instance data (e.g., coordinates).
            num_ants (int): Number of ants (solutions generated per iteration).
            decay_rate (float): Pheromone evaporation rate (rho).
            alpha (float): Pheromone influence factor.
            beta (float): Heuristic influence factor (e.g., inverse distance).
            q0 (float): Exploration/exploitation trade-off parameter.
            initial_pheromone (float): Initial pheromone value.
            min_pheromone (float): Minimum pheromone level.
            max_pheromone (float): Maximum pheromone level.
            device: PyTorch device.
        """
        self.problem = problem
        self.instance_batch = instance_batch # Use the provided instance
        self.num_ants = num_ants
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone
        self.device = device

        self.is_cvrp = False # Currently only supporting TSP
        if self.is_cvrp:
             raise NotImplementedError("CVRP is not supported in this StandardACO version")

        self.num_nodes = problem.size
        self.locations = self.instance_batch['coordinates'][0] # Shape [num_nodes, 2]

        # Calculate distances matrix for heuristic information
        self.distances = torch.cdist(self.locations, self.locations)

        # Initialize heuristic information (1/distance)
        self.heuristic_info = 1.0 / (self.distances + 1e-10)
        self.heuristic_info.fill_diagonal_(0)

        # Initialize pheromone trails
        self.pheromone = torch.ones((self.num_nodes, self.num_nodes), device=device) * initial_pheromone
        self.pheromone.fill_diagonal_(0)

        self.best_solution = None
        self.best_cost = float('inf')

    def construct_solutions(self):
        """ Construct solutions using multiple ants. """
        all_solutions = []
        all_costs = []
        for _ in range(self.num_ants):
            if self.is_cvrp:
                 # solution, cost = self._construct_cvrp_solution()
                 raise NotImplementedError("CVRP not supported")
            else:
                solution, cost = self._construct_tsp_solution()
            all_solutions.append(solution)
            all_costs.append(cost)
        return all_solutions, all_costs

    def _construct_tsp_solution(self):
        """ Construct a single TSP solution tour. """
        solution = []
        visited = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        # Start from a random city (ensure reproducibility within a run if needed later)
        current_node_idx = torch.randint(0, self.num_nodes, (1,), device=self.device).item()

        solution.append(current_node_idx)
        visited[current_node_idx] = True
        current_node = current_node_idx

        for _ in range(self.num_nodes - 1):
            next_node = self._select_next_node(current_node, visited)
            solution.append(next_node)
            visited[next_node] = True
            current_node = next_node

        # Calculate cost using the problem's method
        solution_tensor = torch.tensor([solution], device=self.device, dtype=torch.long)
        cost = self.problem.get_costs(self.instance_batch, solution_tensor).item()
        return solution, cost

    def _select_next_node(self, current_node, visited):
        """ Select the next node using ACO's probabilistic rule. """
        candidate_indices = torch.where(~visited)[0]

        if candidate_indices.numel() == 0:
             # Should not happen in standard TSP construction unless num_nodes=1
             # If it does, maybe return to start? For now, error or handle gracefully.
             # This case needs care if problem size can be 1.
             if self.num_nodes > 0:
                 # Use first node as fallback (instead of solution[0])
                 return 0 # Return to depot/first city if no candidates (unlikely)
             else:
                 return -1 # Or some indicator of no nodes left

        # Get pheromone and heuristic values for candidates
        pheromone_vals = self.pheromone[current_node, candidate_indices]
        heuristic_vals = self.heuristic_info[current_node, candidate_indices]

        # Calculate scores
        scores = (pheromone_vals ** self.alpha) * (heuristic_vals ** self.beta)

        # Check for zero scores or numerical issues
        if torch.isinf(scores).any() or torch.isnan(scores).any() or scores.sum() == 0:
             # Fallback to random choice among candidates if scores are invalid
             rand_idx = torch.randint(0, candidate_indices.numel(), (1,)).item()
             return candidate_indices[rand_idx].item()

        # Pseudo-random proportional rule
        if torch.rand(1).item() < self.q0: # Exploitation
            best_candidate_idx = torch.argmax(scores)
            return candidate_indices[best_candidate_idx].item()
        else: # Exploration (probabilistic choice)
            probabilities = scores / scores.sum()
            # Use multinomial sampling to choose based on probabilities
            try:
                choice_idx = torch.multinomial(probabilities, 1).item()
                return candidate_indices[choice_idx].item()
            except RuntimeError: # Handle potential numerical issues in multinomial
                 rand_idx = torch.randint(0, candidate_indices.numel(), (1,)).item()
                 return candidate_indices[rand_idx].item()


    def update_pheromones(self, solutions, costs):
        """ Update pheromone trails based on solution quality. """
        # 1. Evaporation
        self.pheromone *= (1.0 - self.decay_rate)

        # 2. Deposition (using the best ant of the iteration)
        best_idx = np.argmin(costs)
        best_solution = solutions[best_idx]
        best_cost = costs[best_idx]

        # Deposit pheromone based on the quality of the best solution
        if self.is_cvrp:
             raise NotImplementedError("CVRP not supported")
        else:
            self._deposit_pheromone_tsp(best_solution, best_cost)

        # Clamp pheromone levels
        self.pheromone = torch.clamp(self.pheromone, self.min_pheromone, self.max_pheromone)

        # Update global best if necessary
        if best_cost < self.best_cost:
            self.best_cost = best_cost
            self.best_solution = best_solution
            # print(f"  [StdACO] New best: {self.best_cost:.4f}") # Optional: per-iteration print

    def _deposit_pheromone_tsp(self, solution, cost):
        """ Deposit pheromone for TSP. """
        delta_tau = 1.0 / (cost + 1e-9) # Add epsilon for safety

        solution_tensor = torch.tensor(solution, device=self.device)
        start_nodes = solution_tensor
        end_nodes = torch.roll(solution_tensor, shifts=-1) # Get next node in tour

        # Update pheromone on the edges of the tour
        self.pheromone[start_nodes, end_nodes] += delta_tau
        self.pheromone[end_nodes, start_nodes] += delta_tau # Assume symmetric TSP


    def run(self, max_iterations):
        """
        Run the main Standard ACO loop.
        """
        print(f"Running Standard ACO for {max_iterations} iterations...")
        start_time = time.time()
        iter_costs = []

        for i in range(max_iterations):
            # 1. Construct solutions using ants
            constructed_solutions, constructed_costs = self.construct_solutions()

            # Store best cost of this iteration
            iter_best_cost = min(constructed_costs) if constructed_costs else float('inf')
            iter_costs.append(iter_best_cost)

            # 2. Update pheromones based ONLY on constructed solutions
            self.update_pheromones(constructed_solutions, constructed_costs)

            # Optional: Print progress
            if (i + 1) % 10 == 0 or i == 0:
                 print(f"  [StdACO] Iter {i+1}/{max_iterations}, Iter Best: {iter_best_cost:.4f}, Global Best: {self.best_cost:.4f}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Standard ACO finished in {total_time:.2f} seconds. Best cost: {self.best_cost:.4f}")

        return self.best_solution, self.best_cost, total_time


# --- Placeholder for NeuOpt Runner ---
def run_neuopt_solver(problem, neuopt_agent, instance_batch, max_steps, device):
    """
    Runs the standalone NeuOpt solver on a given instance.

    Args:
        problem: The problem object (e.g., TSP).
        neuopt_agent: The loaded PPO agent.
        instance_batch: The specific problem instance data.
        max_steps (int): The number of improvement steps to perform.
        device: The PyTorch device.

    Returns:
        Tuple: (best_solution_list, best_cost, total_time)
    """
    print(f"Running Standalone NeuOpt for {max_steps} steps...")
    start_time = time.time()

    try:
        # Generate an initial random solution (permutation for TSP)
        batch_size = instance_batch['coordinates'].shape[0]
        graph_size = problem.size
        # Create a random permutation for each item in the batch (usually batch_size=1)
        initial_solution = torch.stack([torch.randperm(graph_size, device=device) for _ in range(batch_size)])

        current_solution = initial_solution
        best_solution = current_solution.clone()
        
        # Get initial cost
        current_cost = problem.get_costs(instance_batch, current_solution).item()
        best_cost = current_cost

        # Cost tensor format should be [batch_size, 3] with the same value repeated
        # This matches the hybrid implementation exactly
        cost_tensor = torch.tensor([[current_cost, current_cost, current_cost]], device=device)

        # For tracking improvement
        previous_cost = current_cost
        
        # Initial feature encoding
        batch_feature = problem.input_feature_encoding(instance_batch)
        
        # Initialize required variables for actor forward
        last_action = None
        context = None  # For TSP, context is None
        context2 = torch.zeros(batch_size, 9, device=device).float()

        # Set agent to evaluation mode
        neuopt_agent.eval()

        print(f"  [NeuOpt] Initial cost: {current_cost:.4f}")

        for step in range(max_steps):
            try:
                # Get next action from policy
                with torch.no_grad():
                    action, _, _ = neuopt_agent.actor.forward(
                        problem=problem,
                        batch=instance_batch,
                        x_in=batch_feature,
                        solution=current_solution,
                        context=context,
                        context2=context2,
                        last_action=last_action,
                        require_entropy=False
                    )

                # Apply the action to get next state
                # Always use t=0 to match hybrid implementation
                next_solution, _, next_cost_tensor, _, _, _, _ = problem.step(
                    instance_batch,  # batch
                    current_solution,  # rec (current solution)
                    action,  # action
                    cost_tensor,  # obj (cost tensor)
                    None,  # feasible_history (not needed for TSP)
                    0,  # t (always 0 in hybrid implementation)
                    0  # weights
                )

                # Get actual cost of new solution
                next_cost = problem.get_costs(instance_batch, next_solution).item()

                # Track improvement
                if next_cost < previous_cost:
                    previous_cost = next_cost

                # Apply greedy acceptance (if better)
                if next_cost < current_cost:
                    current_solution = next_solution
                    current_cost = next_cost
                    cost_tensor = torch.tensor([[current_cost, current_cost, current_cost]], device=device)
                    
                    # Update best solution found
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_solution = current_solution.clone()
                        print(f"  [NeuOpt] Step {step+1}: New best cost = {best_cost:.4f}")

                # Update last action for next iteration
                last_action = action

            except Exception as e:
                print(f"  [NeuOpt] Error in step {step+1}: {e}")
                # Continue with next step or break if severe
                if step == 0:  # If we can't even complete one step, give up
                    raise
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Standalone NeuOpt finished in {total_time:.2f} seconds. Best cost: {best_cost:.4f}")

        # Convert best solution to list format for consistent reporting
        best_solution_list = best_solution.squeeze(0).cpu().numpy().tolist()

        return best_solution_list, best_cost, total_time
    
    except Exception as e:
        print(f"Fatal error in NeuOpt solver: {e}")
        import traceback
        traceback.print_exc()
        # Return failure
        return None, float('inf'), 0.0


# --- Placeholder for Instance Generation ---
def generate_tsp_instances(num_instances, graph_size, device, base_seed=1234):
    """
    Generates a list of TSP instances.
    Each instance is a batch dictionary containing coordinates.
    Args:
        num_instances (int): How many instances to generate.
        graph_size (int): The number of nodes (cities) in each instance.
        device: The PyTorch device to store instance data on.
        base_seed (int): Seed for the random number generator.
    Returns:
        List[Dict]: A list of instance batch dictionaries.
    """
    instances = []
    print(f"Generating {num_instances} TSP instances of size {graph_size}...")
    # Use a specific generator for reproducibility
    rng = np.random.default_rng(base_seed)
    for i in range(num_instances):
        # Generate random coordinates in [0, 1] x [0, 1]
        # Shape: (batch_size=1, num_nodes, 2)
        coords = rng.random((1, graph_size, 2))
        instance_batch = {
            # Ensure data is float32 as often expected by models
            'coordinates': torch.tensor(coords, dtype=torch.float32, device=device)
        }
        instances.append(instance_batch)
    print(f"{len(instances)} instances generated successfully.")
    return instances


def visualize_results(summary, args):
    """
    Create visualization charts for benchmark results.
    
    Args:
        summary: Dictionary containing benchmark summary
        args: Command-line arguments
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping visualizations.")
        return
    
    print("\nGenerating visualizations...")
    
    # Extract solver data
    solver_names = []
    avg_costs = []
    best_costs = []
    avg_times = []
    
    for solver_name, solver_data in summary['solvers'].items():
        if solver_data['status'] == 'success':
            solver_names.append(solver_name.upper())
            avg_costs.append(solver_data['avg_cost'])
            best_costs.append(solver_data['best_cost'])
            avg_times.append(solver_data['avg_time'])
    
    # Set up a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Benchmark Results - {args.problem.upper()} {args.graph_size}', fontsize=16)
    
    # Plot 1: Average Solution Costs
    x = range(len(solver_names))
    barwidth = 0.35
    
    bars1 = ax1.bar(x, avg_costs, barwidth, label='Avg Cost', color='steelblue')
    bars2 = ax1.bar([i + barwidth for i in x], best_costs, barwidth, label='Best Cost', color='forestgreen')
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax1.set_ylabel('Cost')
    ax1.set_title('Solution Quality')
    ax1.set_xticks([i + barwidth/2 for i in x])
    ax1.set_xticklabels(solver_names)
    ax1.legend()
    
    # Plot 2: Average Running Times
    bars3 = ax2.bar(x, avg_times, barwidth, color='indianred')
    
    # Add value labels on top of bars
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}s',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Computational Efficiency')
    ax2.set_xticks(x)
    ax2.set_xticklabels(solver_names)
    
    plt.tight_layout()
    
    # Save figure if save_results is enabled
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{args.problem}_{args.graph_size}_{timestamp}_plot.png"
        plot_filepath = os.path.join(args.output_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_filepath}")
    
    plt.show()


def main(args, opts):
    """
    Main function to run the solver comparison benchmark.
    
    Args:
        args: Command-line arguments for our comparison script
        opts: NeuOpt-specific options
    """
    print("\n" + "="*80)
    print(f"COMPARING SOLVERS FOR {args.problem.upper()} {args.graph_size}")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create problem instance based on problem type
    if args.problem == 'tsp':
        problem = TSP(p_size=args.graph_size)
    else:
        raise NotImplementedError(f"Problem type {args.problem} not implemented")
    
    # Load NeuOpt agent if needed
    if args.run_hybrid or args.run_neuopt:
        try:
            print(f"Loading NeuOpt model from {opts.load_path}...")
            neuopt_agent = PPO(problem, opts)
            neuopt_agent.load(opts.load_path)
            neuopt_agent.eval()  # Set to evaluation mode
            print("NeuOpt model loaded successfully.")
        except Exception as e:
            print(f"Error loading NeuOpt model: {e}")
            if args.run_hybrid:
                print("Disabling Hybrid ACO-NeuOpt due to model loading failure.")
                args.run_hybrid = False
            if args.run_neuopt:
                print("Disabling standalone NeuOpt due to model loading failure.")
                args.run_neuopt = False
    
    # Generate problem instances for testing
    instances = generate_tsp_instances(args.num_runs, args.graph_size, device, args.seed)
    
    # Prepare results storage
    results = {
        'hybrid': [] if args.run_hybrid else None,
        'standard_aco': [] if args.run_standard_aco else None,
        'neuopt': [] if args.run_neuopt else None
    }
    
    # Run solvers on each instance
    for i, instance in enumerate(instances):
        print(f"\nSOLVING INSTANCE {i+1}/{len(instances)}")
        print("-" * 50)
        
        # Run each solver if enabled
        if args.run_standard_aco:
            print("\n[STANDARD ACO]")
            try:
                # Create Standard ACO solver
                std_aco = StandardACO(
                    problem=problem,
                    instance_batch=instance,
                    num_ants=args.num_ants,
                    decay_rate=args.decay_rate,
                    alpha=args.alpha,
                    beta=args.beta,
                    q0=args.q0,
                    device=device
                )
                # Run Standard ACO
                sol_std, cost_std, time_std = std_aco.run(args.aco_iterations)
                results['standard_aco'].append({
                    'solution': sol_std,
                    'cost': cost_std,
                    'time': time_std
                })
            except Exception as e:
                print(f"Error running Standard ACO: {e}")
                results['standard_aco'].append({
                    'solution': None,
                    'cost': float('inf'),
                    'time': 0.0
                })
        
        if args.run_neuopt:
            print("\n[STANDALONE NEUOPT]")
            try:
                # Run standalone NeuOpt
                sol_neuopt, cost_neuopt, time_neuopt = run_neuopt_solver(
                    problem=problem,
                    neuopt_agent=neuopt_agent,
                    instance_batch=instance,
                    max_steps=args.neuopt_steps,
                    device=device
                )
                results['neuopt'].append({
                    'solution': sol_neuopt,
                    'cost': cost_neuopt,
                    'time': time_neuopt
                })
            except Exception as e:
                print(f"Error running standalone NeuOpt: {e}")
                results['neuopt'].append({
                    'solution': None,
                    'cost': float('inf'),
                    'time': 0.0
                })
        
        if args.run_hybrid:
            print("\n[HYBRID ACO-NEUOPT]")
            try:
                # Create Hybrid ACO-NeuOpt solver
                hybrid_aco = HybridACO(
                    problem=problem,
                    neuopt_agent=neuopt_agent,
                    num_ants=args.num_ants,
                    decay_rate=args.decay_rate,
                    alpha=args.alpha,
                    beta=args.beta,
                    q0=args.q0,
                    local_search_steps=args.local_search_steps,
                    elite_percentage=args.elite_percentage,
                    early_stop_threshold=args.early_stop_threshold,
                    adaptive_search=args.adaptive_search,
                    parallel_processing=args.parallel_processing,
                    progressive_elite=args.progressive_elite,
                    device=device
                )
                # Run Hybrid ACO-NeuOpt (HybridACO doesn't return time currently, so track it manually)
                hybrid_start = time.time()
                sol_hybrid, cost_hybrid = hybrid_aco.run(args.aco_iterations)
                hybrid_time = time.time() - hybrid_start
                results['hybrid'].append({
                    'solution': sol_hybrid,
                    'cost': cost_hybrid,
                    'time': hybrid_time
                })
            except Exception as e:
                print(f"Error running Hybrid ACO-NeuOpt: {e}")
                results['hybrid'].append({
                    'solution': None,
                    'cost': float('inf'),
                    'time': 0.0
                })
    
    # Calculate and print summary statistics
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # Prepare summary table headers
    print(f"{'Solver':<20} {'Avg Cost':<12} {'Best Cost':<12} {'Worst Cost':<12} {'Avg Time(s)':<12}")
    print("-" * 76)
    
    # Calculate statistics for each solver and prepare summary for JSON export
    summary = {
        'problem': args.problem,
        'graph_size': args.graph_size,
        'num_runs': args.num_runs,
        'seed': args.seed,
        'aco_params': {
            'num_ants': args.num_ants,
            'decay_rate': args.decay_rate,
            'alpha': args.alpha,
            'beta': args.beta,
            'q0': args.q0,
            'iterations': args.aco_iterations
        },
        'neuopt_params': {
            'model_path': args.model_path,
            'steps': args.neuopt_steps,
            'local_search_steps': args.local_search_steps
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'solvers': {}
    }
    
    for solver_name, solver_results in results.items():
        if solver_results is None:
            continue  # Skip disabled solvers
            
        # Extract costs and times, handling potential failures
        costs = [result['cost'] for result in solver_results 
                if result['solution'] is not None and not math.isinf(result['cost'])]
        times = [result['time'] for result in solver_results 
                if result['solution'] is not None and not math.isinf(result['cost'])]
        
        if not costs:  # Handle the case where all runs failed
            print(f"{solver_name.upper():<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            summary['solvers'][solver_name] = {
                'status': 'failed',
                'message': 'All runs failed'
            }
            continue
            
        avg_cost = sum(costs) / len(costs)
        best_cost = min(costs)
        worst_cost = max(costs)
        avg_time = sum(times) / len(times)
        
        # Store statistics in summary
        summary['solvers'][solver_name] = {
            'status': 'success',
            'avg_cost': avg_cost,
            'best_cost': best_cost,
            'worst_cost': worst_cost,
            'avg_time': avg_time,
            'instances': []
        }
        
        # Add per-instance results
        for i, result in enumerate(solver_results):
            if result['solution'] is not None and not math.isinf(result['cost']):
                # Convert numpy arrays or tensors to Python lists for JSON serialization
                solution = result['solution']
                if isinstance(solution, np.ndarray):
                    solution = solution.tolist()
                elif isinstance(solution, torch.Tensor):
                    solution = solution.cpu().numpy().tolist()
                
                summary['solvers'][solver_name]['instances'].append({
                    'instance_id': i,
                    'cost': result['cost'],
                    'time': result['time'],
                    # Optionally include solutions (might be large)
                    'solution': solution if isinstance(solution, list) else None
                })
        
        # Print statistics row
        print(f"{solver_name.upper():<20} {avg_cost:<12.4f} {best_cost:<12.4f} {worst_cost:<12.4f} {avg_time:<12.2f}")
    
    print("\n" + "="*80)
    print("DETAILED INSTANCE RESULTS")
    print("="*80)
    
    # Print per-instance results
    for i in range(len(instances)):
        print(f"\nInstance {i+1}:")
        print(f"{'Solver':<20} {'Cost':<12} {'Time(s)':<12}")
        print("-" * 44)
        
        for solver_name, solver_results in results.items():
            if solver_results is None:
                continue  # Skip disabled solvers
                
            if i < len(solver_results):
                result = solver_results[i]
                if result['solution'] is None or math.isinf(result['cost']):
                    print(f"{solver_name.upper():<20} {'FAILED':<12} {'N/A':<12}")
                else:
                    print(f"{solver_name.upper():<20} {result['cost']:<12.4f} {result['time']:<12.2f}")
    
    # Save results to file if requested
    if args.save_results:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.problem}_{args.graph_size}_{timestamp}.json"
        filepath = os.path.join(args.output_dir, filename)
        
        # Save JSON results
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    # Generate visualizations if requested
    if args.visualize:
        visualize_results(summary, args)
    
    print("\nBenchmark completed successfully.")
    
    # Return results for potential further analysis
    return results, summary


if __name__ == "__main__":
    args, opts = parse_arguments()
    print("Arguments parsed:")
    print(f"  Problem: {args.problem}, Size: {args.graph_size}")
    print(f"  Num Runs: {args.num_runs}, Base Seed: {args.seed}")
    print(f"  ACO Ants: {args.num_ants}, Iterations: {args.aco_iterations}")
    print(f"  NeuOpt Model: {args.model_path}")
    print(f"  Hybrid Local Steps: {args.local_search_steps}")
    print(f"  Standalone NeuOpt Steps: {args.neuopt_steps}")
    print("NeuOpt Options (subset):")
    print(f"  opts.load_path = {opts.load_path}")
    print(f"  opts.graph_size = {opts.graph_size}")
    print(f"  opts.problem = {opts.problem}")
    print(f"  opts.eval_only = {opts.eval_only}")

    # Call main function and get results
    results, summary = main(args, opts) 