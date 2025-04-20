import torch
import numpy as np
import time
import sys
from problems.problem_tsp import TSP
from agent.ppo import PPO
from options import get_options

def main():
    """
    Simple script that uses NeuOpt for TSP.
    """
    # Load options without parsing command line arguments
    opts = get_options([])
    opts.load_path = "pre-trained/tsp100.pt"
    opts.graph_size = 100
    opts.problem = "tsp"
    opts.eval_only = True
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts.device = device
    
    # Create problem instance
    problem = TSP(p_size=opts.graph_size)
    
    # Generate a random TSP instance
    np.random.seed(1234)
    torch.manual_seed(1234)
    
    # Create dataset with a single instance
    dataset = [{'coordinates': torch.rand(opts.graph_size, 2)}]
    val_dataset = torch.utils.data.DataLoader(dataset, batch_size=1)
    
    # Load pretrained NeuOpt agent
    print("Loading pretrained NeuOpt agent...")
    neuopt_agent = PPO(problem, opts)
    neuopt_agent.load(opts.load_path)
    neuopt_agent.eval()
    
    # Use the rollout function directly
    print("Running NeuOpt local search...")
    start_time = time.time()
    
    batch = next(iter(val_dataset))
    
    # Run with the number of iterations (T) and number of augmentation rounds (val_m)
    T = 100  # Number of improvement steps
    val_m = 1  # Number of augmentation rounds
    stall_limit = 0  # No stall limit
    
    # Run the rollout
    batch_min_cost, traj_costs, traj_rewards, traj_solutions = neuopt_agent.rollout(
        problem, T, val_m, stall_limit, batch, record=True, show_bar=True
    )
    
    end_time = time.time()
    
    # Calculate best cost
    best_cost = batch_min_cost.item()
    initial_cost = traj_costs[0, 0, 0].item()
    
    print(f"Local search completed in {end_time - start_time:.2f} seconds")
    print(f"Initial cost: {initial_cost:.4f}")
    print(f"Final cost: {best_cost:.4f}")
    print(f"Improvement: {(initial_cost - best_cost) / initial_cost * 100:.2f}%")
    
    # Plot the cost trajectory if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Extract cost trajectory
        cost_trajectory = traj_costs[0, 0, :, 0].cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(cost_trajectory)
        plt.title("NeuOpt Cost Trajectory")
        plt.xlabel("Step")
        plt.ylabel("Cost")
        plt.grid(True)
        plt.savefig("neuopt_trajectory.png")
        print("Cost trajectory saved to 'neuopt_trajectory.png'")
    except ImportError:
        print("matplotlib not available for plotting")

if __name__ == "__main__":
    main() 