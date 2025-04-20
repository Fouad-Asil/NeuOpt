import torch
import numpy as np
import time
import sys
from problems.problem_tsp import TSP
from agent.ppo import PPO
from options import get_options

def main():
    """
    Simplified hybrid that uses NeuOpt for TSP local search.
    """
    # Load options without parsing command line arguments
    opts = get_options([])
    opts.load_path = "pre-trained/tsp100.pt"
    opts.graph_size = 100
    opts.problem = "tsp"
    opts.eval_only = True
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create problem instance
    problem = TSP(p_size=opts.graph_size)
    
    # Load pretrained NeuOpt agent
    neuopt_agent = PPO(problem, opts)
    neuopt_agent.load(opts.load_path)
    neuopt_agent.eval()
    
    # Generate a random TSP instance
    instance_batch = {'coordinates': torch.rand(1, problem.size, 2, device=device)}
    print(f"Generated a random TSP instance with {problem.size} nodes")
    
    # Generate a random initial solution
    initial_solution = problem.get_initial_solutions(instance_batch)
    
    # Calculate initial cost
    initial_cost = problem.get_costs(instance_batch, initial_solution).item()
    print(f"Initial solution cost: {initial_cost:.4f}")
    
    # Define parameters for improvement
    max_iterations = 100
    
    # Track best solution
    best_solution = initial_solution.clone()
    best_cost = initial_cost
    
    print(f"Running {max_iterations} iterations of NeuOpt local search...")
    
    # Run local search iterations
    start_time = time.time()
    current_solution = initial_solution.clone()
    current_cost = initial_cost
    
    for i in range(max_iterations):
        # Prepare for NeuOpt step
        batch_feature = problem.input_feature_encoding(instance_batch)
        cost_tensor = torch.tensor([[current_cost, current_cost, current_cost]], device=device)
        
        # Get action from NeuOpt
        with torch.no_grad():
            action, _, _, _ = neuopt_agent.actor.forward(
                problem=problem,
                batch=instance_batch,
                x_in=batch_feature,
                solution=current_solution,
                context=None,
                context2=torch.zeros(1, 9, device=device).float(),
                last_action=None,
                require_entropy=False
            )
        
        # Apply action
        next_solution, rewards, _, _, _, _, _ = problem.step(
            instance_batch,
            current_solution,
            action,
            cost_tensor,
            None,
            i,
            weights=0
        )
        
        # Calculate new cost
        next_cost = problem.get_costs(instance_batch, next_solution).item()
        
        # Update if better
        if next_cost < current_cost:
            current_solution = next_solution
            current_cost = next_cost
            
            # Update best overall
            if current_cost < best_cost:
                best_solution = current_solution.clone()
                best_cost = current_cost
                print(f"Iteration {i+1}: New best cost = {best_cost:.4f}")
        
        # Progress update
        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{max_iterations}, Current cost: {current_cost:.4f}, Best cost: {best_cost:.4f}")
    
    end_time = time.time()
    print(f"Local search completed in {end_time - start_time:.2f} seconds")
    print(f"Initial cost: {initial_cost:.4f}")
    print(f"Final cost: {best_cost:.4f}")
    print(f"Improvement: {(initial_cost - best_cost) / initial_cost * 100:.2f}%")

if __name__ == "__main__":
    main() 