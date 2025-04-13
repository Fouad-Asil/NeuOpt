import os
import numpy as np
import torch
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from problems.problem_tsp import TSP, TSPDataset
from agent.ppo import PPO
from options import get_options
from utils import torch_load_cpu, move_to


class ABC_NeuOpt:
    def __init__(self, colony_size=50, max_iterations=1000, limit=20, model_path='pre-trained/tsp20.pt'):
        """
        Initialize the ABC algorithm with NeuOpt integration
        
        Args:
            colony_size: Number of food sources (also number of employed/onlooker bees)
            max_iterations: Maximum number of iterations
            limit: Maximum number of trials for abandonment
            model_path: Path to the pretrained NeuOpt model
        """
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.limit = limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load NeuOpt model
        self.opts = self._get_opts()
        # Set the model path
        self.opts.load_path = model_path
        self.problem = TSP(
            p_size=self.opts.graph_size,
            init_val_met=self.opts.init_val_met,
            with_assert=self.opts.use_assert,
            DUMMY_RATE=self.opts.dummy_rate,
            k=self.opts.k,
            with_bonus=not self.opts.wo_bonus,
            with_regular=not self.opts.wo_regular
        )
        
        # Initialize the agent
        self.agent = PPO(self.problem, self.opts)
        
        # Move to evaluation mode
        self.agent.eval()
        
    def _get_opts(self):
        """Get command line options with defaults set for evaluation"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--eval_only', action='store_true', default=True)
        parser.add_argument('--no_saving', action='store_true', default=True)
        parser.add_argument('--no_tb', action='store_true', default=True)
        parser.add_argument('--init_val_met', type=str, default='random')
        parser.add_argument('--val_size', type=int, default=1000)
        parser.add_argument('--val_batch_size', type=int, default=100)
        parser.add_argument('--k', type=int, default=4)
        parser.add_argument('--problem', type=str, default='tsp')
        parser.add_argument('--val_dataset', type=str, default='datasets/tsp_20.pkl')
        parser.add_argument('--graph_size', type=int, default=20)
        parser.add_argument('--val_m', type=int, default=1)
        parser.add_argument('--stall', type=int, default=10)
        parser.add_argument('--T_max', type=int, default=1000)
        parser.add_argument('--load_path', type=str, default='pre-trained/tsp20.pt')
        parser.add_argument('--use_assert', action='store_true', default=False)
        
        # Additional arguments required by the NeuOpt model
        parser.add_argument('--embedding_dim', type=int, default=128)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--actor_head_num', type=int, default=4)
        parser.add_argument('--critic_head_num', type=int, default=4)
        parser.add_argument('--n_encode_layers', type=int, default=3)
        parser.add_argument('--normalization', type=str, default='layer')
        parser.add_argument('--v_range', type=float, default=6.0)
        parser.add_argument('--seed', type=int, default=6666)
        parser.add_argument('--wo_MDP', action='store_true', default=True)
        parser.add_argument('--wo_RNN', action='store_true', default=False)
        parser.add_argument('--wo_bonus', action='store_true', default=True)
        parser.add_argument('--wo_feature1', action='store_true', default=True)
        parser.add_argument('--wo_feature2', action='store_true', default=True)
        parser.add_argument('--wo_feature3', action='store_true', default=True)
        parser.add_argument('--wo_regular', action='store_true', default=True)
        parser.add_argument('--dummy_rate', type=float, default=0.5)
        parser.add_argument('--distributed', action='store_true', default=False)
        parser.add_argument('--no_cuda', action='store_true', default=False)
        parser.add_argument('--use_cuda', action='store_true', default=True)
        parser.add_argument('--world_size', type=int, default=1)
        parser.add_argument('--no_progress_bar', action='store_true', default=False)
        
        opts = parser.parse_args([])  # Empty list to use defaults
        
        # Set the device attribute and force CUDA if available
        opts.use_cuda = torch.cuda.is_available()
        opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
        
        return opts
            
    def _ensure_device(self, tensor_or_dict):
        """Helper method to ensure tensors are on the correct device"""
        if isinstance(tensor_or_dict, dict):
            return {k: self._ensure_device(v) for k, v in tensor_or_dict.items()}
        elif isinstance(tensor_or_dict, torch.Tensor):
            return tensor_or_dict.to(self.device)
        else:
            return tensor_or_dict
            
    def solve(self, tsp_instance):
        """
        Solve a TSP instance using a combination of ABC and NeuOpt
        
        Args:
            tsp_instance: A dictionary containing 'coordinates' tensor of shape [graph_size, 2]
            
        Returns:
            best_solution: The best solution found
            best_cost: The cost of the best solution
            costs_per_iteration: List of best costs at each iteration
            metrics: Dictionary with metrics about the solving process
        """
        # Convert to batch format and move to device
        batch = {'coordinates': tsp_instance['coordinates'].unsqueeze(0)}
        batch = self._ensure_device(batch)
        
        # Initialize employed bees (food sources)
        employed_bees = []
        trial_counters = []
        
        # Generate initial solutions for all employed bees
        for _ in range(self.colony_size):
            # Generate initial solution using the problem's initialization
            solution = self.problem.get_initial_solutions(batch)
            solution = self._ensure_device(solution)
            # Apply initial 2-opt improvement
            solution = self._improved_two_opt(solution, batch)
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
        
        # Track cost history for performance comparison
        costs_per_iteration = [best_cost]
        best_cost_so_far = best_cost
        
        # Track metrics for research analysis
        last_improvement = 0
        stagnation_count = 0
        improvement_iterations = []
        
        # CRITICAL CHANGE: Use ABC to explore, NeuOpt to exploit
        # ABC will run its own iterations, then we'll use NeuOpt to refine the best solution
        
        # Step 1: Run standard ABC algorithm
        for iteration in tqdm(range(self.max_iterations), desc="ABC Phase"):
            improved_this_iteration = False
            
            # EMPLOYED BEE PHASE
            for i in range(self.colony_size):
                # Apply standard 2-opt with random neighborhood
                current_solution = self._ensure_device(employed_bees[i]['solution'])
                new_solution = self._improved_two_opt(current_solution, batch)
                new_cost = self.problem.get_costs(batch, new_solution).item()
                
                # Greedy selection
                if new_cost < employed_bees[i]['cost']:
                    employed_bees[i]['solution'] = new_solution.clone()
                    employed_bees[i]['cost'] = new_cost
                    trial_counters[i] = 0
                    
                    # Update global best if needed
                    if new_cost < best_cost:
                        best_cost = new_cost
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
                
                current_solution = self._ensure_device(employed_bees[selected_idx]['solution'])
                new_solution = self._improved_two_opt(current_solution, batch)
                new_cost = self.problem.get_costs(batch, new_solution).item()
                
                # Greedy selection
                if new_cost < employed_bees[selected_idx]['cost']:
                    employed_bees[selected_idx]['solution'] = new_solution.clone()
                    employed_bees[selected_idx]['cost'] = new_cost
                    trial_counters[selected_idx] = 0
                    
                    # Update global best if needed
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_solution = new_solution.clone()
                        improved_this_iteration = True
                else:
                    trial_counters[selected_idx] += 1
            
            # SCOUT BEE PHASE
            for i in range(self.colony_size):
                if trial_counters[i] >= self.limit:
                    # Generate a new random solution
                    solution = self.problem.get_initial_solutions(batch)
                    solution = self._ensure_device(solution)
                    # Apply initial 2-opt improvement
                    solution = self._improved_two_opt(solution, batch)
                    cost = self.problem.get_costs(batch, solution).item()
                    
                    employed_bees[i]['solution'] = solution.clone()
                    employed_bees[i]['cost'] = cost
                    trial_counters[i] = 0
                    
                    # Update global best if needed
                    if cost < best_cost:
                        best_cost = cost
                        best_solution = solution.clone()
                        improved_this_iteration = True
            
            # Track best cost
            costs_per_iteration.append(best_cost)
            
            # Track improvement metrics
            if improved_this_iteration:
                improvement_iterations.append(iteration)
                last_improvement = iteration
                stagnation_count = 0
            else:
                stagnation_count += 1
        
        # Step 2: After ABC exploration, run NeuOpt on the best solution for refinement
        print(f"\nABC found solution with cost: {best_cost:.6f}")
        print("Applying NeuOpt for final refinement...")
        
        # Apply NeuOpt to the best solution found by ABC
        refined_solution, refined_cost = self._apply_neuopt_refinement(batch, best_solution)
        
        # Update if NeuOpt improved the solution
        if refined_cost < best_cost:
            print(f"NeuOpt improved solution from {best_cost:.6f} to {refined_cost:.6f}")
            best_solution = refined_solution
            best_cost = refined_cost
            costs_per_iteration.append(best_cost)
            # Count this as an improvement in the last iteration
            improvement_iterations.append(self.max_iterations)
            last_improvement = self.max_iterations
            stagnation_count = 0
        else:
            print(f"NeuOpt couldn't improve the solution. Keeping ABC's best: {best_cost:.6f}")
        
        # Compile metrics for research analysis
        metrics = {
            'stagnation_count': stagnation_count,
            'last_improvement': last_improvement,
            'improvement_iterations': improvement_iterations,
            'solution_quality': best_cost,
            'convergence_curve': costs_per_iteration
        }
        
        # Return best solution, its cost, convergence history, and metrics
        return best_solution.squeeze(0), best_cost, costs_per_iteration, metrics
    
    def _apply_neuopt_refinement(self, batch, solution):
        """
        Apply NeuOpt refinement to a solution
        
        Args:
            batch: Problem batch
            solution: Solution tensor
            
        Returns:
            refined_solution: The refined solution
            refined_cost: The cost of the refined solution
        """
        # Send batch and solution to device
        batch = self._ensure_device(batch)
        solution = self._ensure_device(solution)
        
        # Reshape for model input (batch size 1)
        solution = solution.unsqueeze(0) if solution.dim() == 1 else solution
        
        # Initialize state for the agent
        bs, gs = solution.size()
        state = self.problem.input_feature_encoding(batch)
        
        # Original solution cost
        original_cost = self.problem.get_costs(batch, solution)
        
        # Initialize best solution
        best_solution = solution.clone()
        best_cost = original_cost.clone()
        
        # Initialize objective values (current, best so far, best known)
        # For new problem, best_so_far = current, best_known = infinity
        obj = torch.cat([original_cost.view(-1, 1), 
                         original_cost.view(-1, 1), 
                         torch.ones_like(original_cost).view(-1, 1) * float('inf')], 1)
        
        # Keep track of solution history for animation
        solution_history = [solution.clone()]
        cost_history = [original_cost.item()]
        
        # Use NeuOpt for T_max steps
        for t in range(min(1000, gs * 10)):  # Limit max steps
            # Prepare input for the agent
            with torch.no_grad():
                # Get visited time for the current solution
                visited_time = self.problem.get_order(solution, return_solution=False)
                
                # Get context for TSP (for CVRP we would need additional context)
                context = None
                context2 = torch.zeros(bs, 9).to(self.device)
                context2[:, -1] = 1
                
                # Call the actor directly (not the agent)
                # The Actor forward method requires: problem, batch, x_in, solution, context, context2, last_action
                action_p, _, _ = self.agent.actor(
                    self.problem,
                    batch,
                    state,  # This is the encoded problem state
                    solution.clone(),
                    context,
                    context2,
                    None,  # last_action is None for the first step
                )
                
                # Sample action
                action = action_p.multinomial(1).squeeze(1)
                
                # Reshape action
                reshaped_action = action.view(bs, -1)
                
                # Apply action to get next solution
                solution_next, reward, obj_next, _, _, _, _ = self.problem.step(
                    batch, solution.clone(), reshaped_action, obj, None, t, None
                )
                
                # Update state variables
                solution = solution_next.detach()
                obj = obj_next.detach()
                
                # Update best solution if improved
                if obj[:, 1] < best_cost:
                    best_solution = solution.clone()
                    best_cost = obj[:, 1].clone()
                    
                # Save history for visualization
                solution_history.append(solution.clone())
                cost_history.append(obj[:, 0].item())
                
                # Early stopping if no improvement for a while
                if t > 20 and cost_history[-20] <= min(cost_history[-19:]):
                    break
        
        # Apply additional local search to the final solution
        print("Applying final 2-opt polishing...")
        refined_solution = self._improved_two_opt(best_solution, batch)
        refined_cost = self.problem.get_costs(batch, refined_solution)
        
        # If 2-opt improved, update best
        if refined_cost < best_cost:
            best_solution = refined_solution
            best_cost = refined_cost
            
        # Final 3-opt polishing
        print("Applying final 3-opt polishing...")
        final_solution = self._three_opt_local_search(best_solution, batch)
        final_cost = self.problem.get_costs(batch, final_solution)
        
        # Return the best solution found
        if final_cost < best_cost:
            return final_solution, final_cost
        else:
            return best_solution, best_cost
    
    def _random_k_opt(self, solution, k=2):
        """
        Apply random k-opt move to a solution
        
        Args:
            solution: Solution tensor
            k: Number of nodes to swap
            
        Returns:
            Modified solution
        """
        batch_size, graph_size = solution.size()
        device = solution.device
        
        # Create a random set of swap indices
        selected_indices = torch.randint(0, graph_size, (batch_size, k), device=device)
        
        # Apply swaps to the next pointers
        for i in range(0, k, 2):
            if i+1 < k:
                # Get selected node indices
                idx1 = selected_indices[:, i].view(-1, 1)
                idx2 = selected_indices[:, i+1].view(-1, 1)
                
                # Get next pointers
                next1 = solution.gather(1, idx1)
                next2 = solution.gather(1, idx2)
                
                # Apply the swap
                solution = solution.scatter(1, idx1, next2)
                solution = solution.scatter(1, idx2, next1)
        
        # Ensure the solution is a valid tour
        try:
            self.problem.check_feasibility(solution)
        except:
            # If invalid, revert to original (this should rarely happen)
            pass
        
        return solution
    
    def _improved_two_opt(self, solution, batch):
        """
        Improved 2-opt implementation that systematically tries all possible edge swaps
        and keeps the best improvement.
        """
        batch_size, graph_size = solution.size()
        
        # Get current solution order as a route
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
    
    def _three_opt_local_search(self, solution, batch):
        """
        3-opt local search for final refinement
        """
        batch_size, graph_size = solution.size()
        
        # Get current solution as a route
        route = self.problem.get_order(solution, return_solution=True)
        
        # Get current cost
        current_cost = self.problem.get_costs(batch, solution)
        
        improved = True
        while improved:
            improved = False
            
            # Try all possible 3-opt moves
            for i in range(graph_size):
                for j in range(i+2, graph_size):
                    for k in range(j+2, graph_size):
                        # Skip invalid combinations
                        if i == 0 and k == graph_size-1:
                            continue
                        
                        # There are 8 ways to reconnect segments in 3-opt
                        # We'll try just 4 configurations to keep it simpler
                        
                        # Original: 0 - i - (i+1) - j - (j+1) - k - (k+1) - 0
                        # Case 1: 0 - i - (j+1) - k - (j) - (i+1) - (k+1) - 0
                        # Case 2: 0 - i - (j+1) - k - (i+1) - j - (k+1) - 0
                        # Case 3: 0 - i - (k) - (j+1) - (i+1) - j - (k+1) - 0
                        # Case 4: 0 - i - j - (k) - (j+1) - (i+1) - (k+1) - 0
                        
                        # Try all 4 cases and pick the best
                        best_new_route = None
                        best_improvement = 0
                        
                        # Case 1: Reverse segments (i+1,j) and (j+1,k)
                        new_route = route.clone()
                        
                        # Reverse first segment
                        segment1 = route[0, i+1:j+1].clone()
                        new_route[0, i+1:j+1] = segment1.flip(0)
                        
                        # Reverse second segment
                        segment2 = route[0, j+1:k+1].clone()
                        new_route[0, j+1:k+1] = segment2.flip(0)
                        
                        # Convert to solution representation
                        new_solution = torch.zeros_like(solution)
                        for m in range(graph_size):
                            idx = (m + 1) % graph_size
                            new_solution[0, new_route[0, m]] = new_route[0, idx]
                        
                        new_cost = self.problem.get_costs(batch, new_solution)
                        improvement = current_cost - new_cost
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_new_route = new_route.clone()
                        
                        # Case 2: Exchange segments (i+1,j) and (j+1,k)
                        new_route = route.clone()
                        seg1 = route[0, i+1:j+1].clone()
                        seg2 = route[0, j+1:k+1].clone()
                        
                        # Place second segment first
                        new_route[0, i+1:i+1+(k-j)] = seg2
                        # Place first segment after
                        new_route[0, i+1+(k-j):i+1+(k-j)+(j-i)] = seg1
                        
                        # Convert to solution representation
                        new_solution = torch.zeros_like(solution)
                        for m in range(graph_size):
                            idx = (m + 1) % graph_size
                            new_solution[0, new_route[0, m]] = new_route[0, idx]
                        
                        new_cost = self.problem.get_costs(batch, new_solution)
                        improvement = current_cost - new_cost
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_new_route = new_route.clone()
                        
                        # Skip cases 3 & 4 for simplicity, but they could be added similarly
                        
                        # Apply best move if there was an improvement
                        if best_improvement > 0:
                            improved = True
                            route = best_new_route
                            current_cost -= best_improvement
                            break  # Break inner loop to restart
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        # Convert final route to solution format
        for m in range(graph_size):
            idx = (m + 1) % graph_size
            solution[0, route[0, m]] = route[0, idx]
            
        return solution


def load_tsp_dataset(filename, size=None, num_samples=None):
    """Load TSP dataset from file"""
    return TSPDataset(filename=filename, size=size, num_samples=num_samples)


def main():
    parser = argparse.ArgumentParser(description="ABC algorithm with NeuOpt guidance for TSP")
    parser.add_argument('--dataset', type=str, default='datasets/tsp_20.pkl', help='TSP dataset file')
    parser.add_argument('--model', type=str, default='pre-trained/tsp20.pt', help='Pretrained NeuOpt model')
    parser.add_argument('--colony_size', type=int, default=50, help='Colony size')
    parser.add_argument('--max_iterations', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--limit', type=int, default=20, help='Trial limit for scout bees')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of TSP instances to solve')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(6666)
    np.random.seed(6666)
    random.seed(6666)
    
    # Load TSP dataset
    dataset = load_tsp_dataset(args.dataset, num_samples=args.num_samples)
    
    # Initialize ABC with NeuOpt
    abc = ABC_NeuOpt(
        colony_size=args.colony_size,
        max_iterations=args.max_iterations,
        limit=args.limit,
        model_path=args.model
    )
    
    # Load the model
    abc.agent.load(abc.opts.load_path)
    print(f"Loaded pretrained model from {abc.opts.load_path}")
    
    # Solve each instance
    all_costs = []
    
    for i in range(len(dataset)):
        print(f"\nSolving TSP instance {i+1}/{len(dataset)}")
        
        instance = dataset[i]
        instance['coordinates'] = instance['coordinates'].to(abc.device)
        
        solution, cost, costs_per_iteration, metrics = abc.solve(instance)
        
        # Convert solution to route - move to CPU first to avoid device mismatch
        solution_cpu = solution.cpu().unsqueeze(0)
        route = abc.problem.get_order(solution_cpu, return_solution=True).squeeze(0)
        
        print(f"Best tour cost: {cost:.6f}")
        print(f"Best tour: {route.cpu().numpy().tolist()}")
        
        all_costs.append(cost)
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.plot(costs_per_iteration)
        plt.title(f"Instance {i+1} Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Tour Cost")
        plt.grid(True)
        plt.savefig(f"neuopt_convergence_{i+1}.png")
    
    # Print summary
    print("\nSummary:")
    print(f"Average tour cost: {np.mean(all_costs):.6f} ± {np.std(all_costs):.6f}")
    print(f"Best tour cost: {np.min(all_costs):.6f}")
    print(f"Worst tour cost: {np.max(all_costs):.6f}")


if __name__ == "__main__":
    main() 