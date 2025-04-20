import torch
import numpy as np
import time
import math
import os
import argparse
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Union

# Import NeuOpt components
from problems.problem_tsp import TSP
from problems.problem_cvrp import CVRP
from agent.ppo import PPO
from options import get_options


class ACO:
    """
    Ant Colony Optimization framework designed to integrate with NeuOpt.
    """
    def __init__(self, problem, neuopt_agent, num_ants=50, decay_rate=0.1, alpha=1, beta=2, 
                 q0=0.9, initial_pheromone=1.0, min_pheromone=0.1, max_pheromone=10.0, 
                 local_search_steps=20, elite_percentage=0.25, early_stop_threshold=5,
                 adaptive_search=True, parallel_processing=True, num_workers=None,
                 progressive_elite=True, device='cpu'):
        """
        Initialize ACO.

        Args:
            problem: The VRP instance (e.g., TSP or CVRP object from NeuOpt).
            neuopt_agent: A loaded NeuOpt agent/policy for local search.
            num_ants (int): Number of ants (solutions generated per iteration).
            decay_rate (float): Pheromone evaporation rate (rho).
            alpha (float): Pheromone influence factor.
            beta (float): Heuristic influence factor (e.g., inverse distance).
            q0 (float): Exploration/exploitation trade-off parameter (pseudo-random proportional).
            initial_pheromone (float): Initial pheromone value.
            min_pheromone (float): Minimum pheromone level (prevents stagnation).
            max_pheromone (float): Maximum pheromone level (prevents dominance).
            local_search_steps (int): Maximum number of steps to run NeuOpt local search.
            elite_percentage (float): Percentage of solutions to apply local search to.
            early_stop_threshold (int): Number of non-improving steps before stopping local search.
            adaptive_search (bool): Whether to increase search depth as iterations progress.
            parallel_processing (bool): Whether to use parallel processing for local search.
            num_workers (int): Number of worker processes for parallel processing (None=auto).
            progressive_elite (bool): Whether to gradually increase elite percentage.
            device: PyTorch device.
        """
        self.problem = problem
        self.neuopt_agent = neuopt_agent
        self.num_ants = num_ants
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone
        self.local_search_steps = local_search_steps
        self.elite_percentage = elite_percentage
        self.early_stop_threshold = early_stop_threshold
        self.adaptive_search = adaptive_search
        self.parallel_processing = parallel_processing
        self.progressive_elite = progressive_elite
        self.device = device
        
        # Setup parallel processing
        if self.parallel_processing:
            self.num_workers = num_workers if num_workers is not None else max(1, multiprocessing.cpu_count() - 1)
            print(f"Using {self.num_workers} worker processes for parallel local search")
        
        # Determine if the problem is TSP or CVRP
        self.is_cvrp = isinstance(problem, CVRP)

        # Initialize problem-specific attributes
        self.num_nodes = problem.size
        
        # Generate a random instance to work with for this ACO run
        # This will be our consistent problem instance throughout the ACO run
        if self.is_cvrp:
            # TODO: Handle CVRP case 
            # For CVRP, locations include depot at index 0
            raise NotImplementedError("CVRP is not fully implemented in this version")
        else:
            # For TSP, generate a random instance
            self.instance_batch = {'coordinates': torch.rand(1, self.num_nodes, 2, device=device)}
            self.locations = self.instance_batch['coordinates'][0]  # Shape [num_nodes, 2]
            
        # Calculate distances matrix for heuristic information
        self.distances = torch.cdist(self.locations, self.locations)
        
        # Initialize heuristic information (1/distance) with small epsilon to avoid division by zero
        self.heuristic_info = 1.0 / (self.distances + 1e-10)
        # No heuristic value for self-loops
        self.heuristic_info.fill_diagonal_(0)
        
        # Initialize pheromone trails uniformly
        self.pheromone = torch.ones((self.num_nodes, self.num_nodes), device=device) * initial_pheromone
        # No pheromone on self-loops
        self.pheromone.fill_diagonal_(0)
        
        self.best_solution = None
        self.best_cost = float('inf')
        
        # Track the current iteration for adaptive search
        self.current_iteration = 0
        self.max_iterations = 100  # Default value, will be updated in run()

    def construct_solutions(self):
        """
        Construct solutions using multiple ants based on pheromone and heuristic info.
        Returns list of solutions (routes for CVRP, permutations for TSP) and their costs.
        """
        all_solutions = []
        all_costs = []

        for _ in range(self.num_ants):
            if self.is_cvrp:
                solution, cost = self._construct_cvrp_solution()
            else:
                solution, cost = self._construct_tsp_solution()
                
            all_solutions.append(solution)
            all_costs.append(cost)

        return all_solutions, all_costs

    def _construct_tsp_solution(self):
        """
        Construct a TSP solution using the ant colony approach.
        Returns a permutation of cities (list of indices) and its cost.
        """
        solution = []
        visited = [False] * self.num_nodes
        current_node = np.random.randint(0, self.num_nodes)  # Start from a random city
        solution.append(current_node)
        visited[current_node] = True
        
        # Construct the tour
        for _ in range(self.num_nodes - 1):
            next_node = self._select_next_node(current_node, visited)
            solution.append(next_node)
            visited[next_node] = True
            current_node = next_node
            
        # Calculate solution cost
        solution_tensor = torch.tensor([solution], device=self.device)
        cost = self.problem.get_costs(self.instance_batch, solution_tensor).item()
        return solution, cost

    def _construct_cvrp_solution(self):
        """
        Construct a CVRP solution using the ant colony approach.
        Returns a list of routes (each route starts and ends at depot 0) and its cost.
        """
        # Currently not fully implemented
        raise NotImplementedError("CVRP is not fully implemented in this version")
        
    def _select_next_node(self, current_node, visited=None, candidates=None):
        """
        Select the next node to visit based on pheromone and heuristic information.
        Uses the pseudo-random proportional rule from ACO.
        
        Args:
            current_node: Current position
            visited: Boolean array tracking visited nodes (for TSP)
            candidates: List of candidate next nodes (for CVRP)
            
        Returns:
            Selected next node index
        """
        if candidates is None:
            # For TSP: candidates are unvisited nodes
            candidates = [i for i in range(self.num_nodes) if not visited[i]]
        
        # With probability q0, choose the best option (exploitation)
        if np.random.random() < self.q0:
            # Calculate the scores for each candidate
            scores = torch.zeros(len(candidates), device=self.device)
            for i, candidate in enumerate(candidates):
                scores[i] = (self.pheromone[current_node, candidate] ** self.alpha) * \
                           (self.heuristic_info[current_node, candidate] ** self.beta)
            
            # Select the candidate with the highest score
            best_idx = torch.argmax(scores).item()
            return candidates[best_idx]
        else:
            # Otherwise, use the proportional rule (exploration)
            scores = torch.zeros(len(candidates), device=self.device)
            for i, candidate in enumerate(candidates):
                scores[i] = (self.pheromone[current_node, candidate] ** self.alpha) * \
                           (self.heuristic_info[current_node, candidate] ** self.beta)
            
            # Convert to probabilities
            probabilities = scores / scores.sum()
            probabilities = probabilities.cpu().numpy()
            
            # Handle numerical instability
            if np.isnan(probabilities).any() or np.sum(probabilities) == 0:
                return np.random.choice(candidates)
            
            # Select based on probability distribution
            selected_idx = np.random.choice(len(candidates), p=probabilities)
            return candidates[selected_idx]

    def apply_local_search(self, solutions, costs):
        """
        Apply NeuOpt local search to the constructed solutions.
        Only applies local search to the top elite_percentage of solutions.
        Can use parallel processing if enabled.
        Returns list of improved solutions and their costs.
        """
        improved_solutions = [None] * len(solutions)
        improved_costs = [None] * len(solutions)
        
        # Sort solutions by cost (ascending)
        sorted_indices = np.argsort(costs)
        
        # For progressive elite selection, adjust elite percentage based on iteration
        if self.progressive_elite:
            # Start with 10% of elite percentage, gradually increase to full percentage
            progress_ratio = min(1.0, self.current_iteration / (0.7 * self.max_iterations))
            current_elite_percentage = 0.1 * self.elite_percentage + progress_ratio * 0.9 * self.elite_percentage
        else:
            current_elite_percentage = self.elite_percentage
            
        # Select only the top percentage of solutions for local search
        num_elite = max(1, int(current_elite_percentage * len(solutions)))
        elite_indices = sorted_indices[:num_elite]
        
        print(f"Applying local search to {num_elite}/{len(solutions)} solutions (elite rate: {current_elite_percentage:.2f})")
        
        # Pre-fill the results arrays with existing solutions and costs first
        # This ensures we have valid values for non-elite solutions
        for idx in range(len(solutions)):
            improved_solutions[idx] = solutions[idx]
            improved_costs[idx] = costs[idx]
        
        if self.parallel_processing and num_elite > 1:
            # Process elite solutions one by one (parallel processing will be implemented in a future version)
            # Current sequential implementation that's safe from the NoneType error
            for idx in elite_indices:
                solution = solutions[idx]
                # Convert ACO solution to NeuOpt state format
                neuopt_state = self._convert_to_neuopt_state(solution)
                
                # Calculate adaptive search depth if enabled
                if self.adaptive_search:
                    progress_ratio = min(1.0, self.current_iteration / 50)
                    search_steps = int(self.local_search_steps * (1.0 + progress_ratio))
                else:
                    search_steps = self.local_search_steps
                
                # Process solution
                improved_state, improved_cost = self._run_neuopt_local_search(neuopt_state, search_steps)
                
                # Convert improved state back to ACO solution format
                final_solution = self._convert_from_neuopt_state(improved_state)
                
                # Store improved solution
                improved_solutions[idx] = final_solution
                improved_costs[idx] = improved_cost
        else:
            # Sequential processing
            for idx in elite_indices:
                solution = solutions[idx]
                # For elite solutions, apply NeuOpt local search
                neuopt_state = self._convert_to_neuopt_state(solution)
                
                # Use adaptive search depth if enabled
                if self.adaptive_search:
                    progress_ratio = min(1.0, self.current_iteration / 50)
                    search_steps = int(self.local_search_steps * (1.0 + progress_ratio))
                else:
                    search_steps = self.local_search_steps
                
                improved_state, improved_cost = self._run_neuopt_local_search(neuopt_state, search_steps)
                
                # Convert improved state back to ACO solution format
                final_solution = self._convert_from_neuopt_state(improved_state)
                
                improved_solutions[idx] = final_solution
                improved_costs[idx] = improved_cost
        
        return improved_solutions, improved_costs
    
    def _convert_to_neuopt_state(self, solution):
        """
        Convert an ACO solution to a state format expected by NeuOpt.
        This depends on NeuOpt's internal state representation.
        """
        # For TSP, the solution is typically a permutation of nodes
        # For CVRP, the solution is a sequence with depot visits
        
        if self.is_cvrp:
            # For CVRP, we need to ensure the solution has proper depot visits
            # NeuOpt expects depot (0) to be used as a separator between routes
            
            # First, ensure there's no leading depot
            if solution and solution[0] == 0:
                solution = solution[1:]
                
            # Make sure we have proper depot visits to separate routes
            formatted_solution = []
            prev_node = 0  # Start from depot
            for node in solution:
                if node == 0:  # If this is a depot visit
                    # Only add if previous node wasn't depot
                    if prev_node != 0:
                        formatted_solution.append(node)
                else:
                    formatted_solution.append(node)
                prev_node = node
                
            # Ensure the solution ends with a depot visit
            if not formatted_solution or formatted_solution[-1] != 0:
                formatted_solution.append(0)
                
            solution_tensor = torch.tensor(formatted_solution, device=self.device)
        else:
            # For TSP, just use the solution as is
            solution_tensor = torch.tensor(solution, device=self.device)
        
        # Create a state dictionary as expected by NeuOpt
        state = {
            'solution': solution_tensor.unsqueeze(0),  # Add batch dimension
            'problem': self.problem,
            # Add the instance coordinates - NeuOpt expects this
            'coordinates': self.instance_batch['coordinates']
        }
        
        return state
    
    def _run_neuopt_local_search(self, state, search_steps):
        """
        Run NeuOpt local search starting from the given state.
        Implements early stopping when no improvement is found.
        Returns improved state and cost.
        """
        current_solution = state['solution']
        # Make sure to use the instance batch instead of just the state for TSP
        current_cost = self.problem.get_costs(self.instance_batch, current_solution).item()
        
        # Get the current cost tensor format expected by step method
        cost_tensor = torch.tensor([[current_cost, current_cost, current_cost]], device=self.device)
        
        # Initialize last_action
        last_action = None
        
        # Keep track of best solution so far
        best_solution = current_solution.clone()
        best_cost = current_cost
        
        # For early stopping
        no_improvement_steps = 0

        # Run NeuOpt for a fixed number of steps or until early stopping
        for step in range(search_steps):
            # Call the actor forward method with all required arguments
            # For TSP, context is None
            batch_feature = self.problem.input_feature_encoding(self.instance_batch)
            
            with torch.no_grad():
                action, _, _ = self.neuopt_agent.actor.forward(
                    problem=self.problem,
                    batch=self.instance_batch,
                    x_in=batch_feature,
                    solution=current_solution,
                    context=None,  # For TSP, context is None
                    context2=torch.zeros(1, 9, device=self.device).float(),  # Default context2
                    last_action=last_action,  # Pass the previous action
                    require_entropy=False
                )
            
            # Apply the action to get the next state
            next_solution, _, next_cost_tensor, _, _, _, _ = self.problem.step(
                self.instance_batch, 
                current_solution, 
                action, 
                cost_tensor,
                None,  # feasibility_history not needed for TSP
                0,     # t = 0
                weights=0
            )
            
            # Get actual cost
            next_cost = self.problem.get_costs(self.instance_batch, next_solution).item()
            
            # Track early stopping
            if next_cost < best_cost:
                best_solution = next_solution.clone()
                best_cost = next_cost
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1
                
            # Check early stopping condition
            if no_improvement_steps >= self.early_stop_threshold:
                # Early stopping - no improvement for several steps
                break
            
            # Accept the new state if it's better than current (greedy acceptance)
            if next_cost < current_cost:
                current_solution = next_solution
                current_cost = next_cost
                cost_tensor = torch.tensor([[current_cost, current_cost, current_cost]], device=self.device)
            
            # Update last_action for the next iteration
            last_action = action
                
        # Use the best solution found during the search
        # This could be different from the current solution if we didn't accept worse solutions
        final_state = {
            'solution': best_solution,
            'problem': self.problem,
            'coordinates': self.instance_batch['coordinates']
        }
        
        return final_state, best_cost
    
    def _convert_from_neuopt_state(self, state):
        """
        Convert from NeuOpt state back to ACO solution format.
        """
        # Extract the solution from the state
        solution = state['solution'].squeeze(0).cpu().numpy().tolist()
        return solution

    def update_pheromones(self, solutions, costs):
        """
        Update pheromone trails based on the quality of the solutions found.
        """
        # 1. Evaporation
        self.pheromone *= (1.0 - self.decay_rate)
        
        # 2. Deposition
        # Get the best solution in this iteration
        best_idx = np.argmin(costs)
        best_solution = solutions[best_idx]
        best_cost = costs[best_idx]
        
        # Deposit pheromone based on the quality of solutions
        if self.is_cvrp:
            self._deposit_pheromone_cvrp(best_solution, best_cost)
        else:
            self._deposit_pheromone_tsp(best_solution, best_cost)
        
        # Clamp pheromone levels
        self.pheromone = torch.clamp(self.pheromone, self.min_pheromone, self.max_pheromone)
        
        # Update global best if necessary
        if best_cost < self.best_cost:
            self.best_cost = best_cost
            self.best_solution = best_solution
            print(f"New best solution found: Cost = {self.best_cost:.4f}")
    
    def _deposit_pheromone_tsp(self, solution, cost):
        """
        Deposit pheromone for TSP based on solution quality.
        """
        # Calculate amount of pheromone to deposit
        delta_tau = 1.0 / cost
        
        # Deposit on edges in the solution
        for i in range(len(solution) - 1):
            node1, node2 = solution[i], solution[i + 1]
            self.pheromone[node1, node2] += delta_tau
            self.pheromone[node2, node1] += delta_tau  # For symmetric TSP
            
        # Deposit on the edge connecting the last and first nodes (complete the tour)
        node1, node2 = solution[-1], solution[0]
        self.pheromone[node1, node2] += delta_tau
        self.pheromone[node2, node1] += delta_tau  # For symmetric TSP
    
    def _deposit_pheromone_cvrp(self, solution, cost):
        """
        Deposit pheromone for CVRP based on solution quality.
        """
        # Currently not fully implemented
        raise NotImplementedError("CVRP is not fully implemented in this version")

    def run(self, max_iterations):
        """
        Run the main ACO loop integrated with NeuOpt local search.
        """
        start_time = time.time()
        self.current_iteration = 0
        self.max_iterations = max_iterations
        
        for i in range(max_iterations):
            self.current_iteration = i
            # 1. Construct solutions using ants
            constructed_solutions, constructed_costs = self.construct_solutions()
            
            # 2. Apply NeuOpt local search
            improved_solutions, improved_costs = self.apply_local_search(constructed_solutions, constructed_costs)
            
            # 3. Update pheromones based on improved solutions
            self.update_pheromones(improved_solutions, improved_costs)
            
            # Print progress
            if (i + 1) % 10 == 0 or i == 0:
                current_best_iter_cost = min(improved_costs) if improved_costs else float('inf')
                print(f"Iteration {i+1}/{max_iterations}, Iter Best Cost: {current_best_iter_cost:.4f}, Global Best Cost: {self.best_cost:.4f}")
                
        end_time = time.time()
        print(f"ACO finished after {max_iterations} iterations in {end_time - start_time:.2f} seconds.")
        print(f"Best solution cost found: {self.best_cost:.4f}")
        
        return self.best_solution, self.best_cost


def parse_arguments():
    """
    Parse command line arguments specific to the ACO-NeuOpt hybrid.
    """
    parser = argparse.ArgumentParser(description='ACO-NeuOpt hybrid solver for routing problems')
    
    # ACO parameters
    parser.add_argument('--num_ants', type=int, default=50, help='Number of ants')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Pheromone evaporation rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='Pheromone influence factor')
    parser.add_argument('--beta', type=float, default=2.0, help='Heuristic influence factor')
    parser.add_argument('--q0', type=float, default=0.9, help='Exploration/exploitation balance parameter')
    parser.add_argument('--max_iterations', type=int, default=100, help='Maximum number of ACO iterations')
    parser.add_argument('--local_search_steps', type=int, default=20, help='Number of NeuOpt steps per solution')
    
    # Performance optimization parameters
    parser.add_argument('--elite_percentage', type=float, default=0.25, help='Percentage of solutions to apply local search to (0.0-1.0)')
    parser.add_argument('--early_stop_threshold', type=int, default=5, help='Stop local search after N non-improving steps')
    parser.add_argument('--adaptive_search', action='store_true', default=True, help='Increase search depth as iterations progress')
    parser.add_argument('--parallel_processing', action='store_true', default=True, help='Use parallel processing for local search')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker processes (default: auto)')
    parser.add_argument('--progressive_elite', action='store_true', default=True, help='Gradually increase elite percentage')
    
    # Problem parameters
    parser.add_argument('--problem', type=str, default='tsp', choices=['tsp', 'cvrp'], help='Problem type')
    parser.add_argument('--graph_size', type=int, default=20, help='Problem size (number of cities/customers)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    # NeuOpt model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained NeuOpt model')
    
    return parser.parse_args()


def main():
    """
    Main function to run the ACO-NeuOpt hybrid.
    """
    # Save a copy of the original sys.argv
    original_argv = sys.argv.copy()
    
    # Parse our own arguments first
    args = parse_arguments()
    
    # Restore original sys.argv for get_options()
    sys.argv = original_argv
    
    # Load NeuOpt options but with empty list to prevent command line parsing
    opts = get_options([])
    
    # Set NeuOpt options from our parsed arguments
    opts.load_path = args.model_path
    opts.graph_size = args.graph_size
    opts.problem = args.problem
    opts.seed = args.seed
    opts.eval_only = True  # We only need evaluation mode
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create problem instance
    if args.problem == 'tsp':
        problem = TSP(p_size=args.graph_size)
    elif args.problem == 'cvrp':
        print("Note: CVRP is not fully implemented in this version. Please use TSP for now.")
        print("Future work includes completing the CVRP implementation.")
        return
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")
    
    # Load pretrained NeuOpt agent
    neuopt_agent = PPO(problem, opts)
    neuopt_agent.load(opts.load_path)
    neuopt_agent.eval()  # Set to evaluation mode
    
    # Create ACO solver
    aco_solver = ACO(
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
        num_workers=args.num_workers,
        progressive_elite=args.progressive_elite,
        device=device
    )
    
    # Print optimization settings
    print("\nOptimization settings:")
    print(f"- Selective local search: {int(args.elite_percentage * 100)}% of solutions")
    print(f"- Early stopping threshold: {args.early_stop_threshold} steps")
    print(f"- Adaptive search depth: {'Enabled' if args.adaptive_search else 'Disabled'}")
    print(f"- Progressive elite selection: {'Enabled' if args.progressive_elite else 'Disabled'}")
    print(f"- Parallel processing: {'Enabled' if args.parallel_processing else 'Disabled'}")
    if args.parallel_processing:
        print(f"- Number of workers: {aco_solver.num_workers}")
    print(f"- Base local search steps: {args.local_search_steps}")
    print()
    
    # Run ACO
    best_solution, best_cost = aco_solver.run(args.max_iterations)
    
    print(f"Final solution: Cost = {best_cost:.4f}")
    
    # Optionally save solution
    # output_dir = "results"
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, f"{args.problem}_{args.graph_size}_{time.strftime('%Y%m%d_%H%M%S')}.txt")


if __name__ == "__main__":
    main() 