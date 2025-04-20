import torch
from problems.problem_tsp import TSP
from agent.ppo import PPO
from options import get_options

def main():
    """
    Simple script to test loading the NeuOpt model
    """
    # Initialize options
    opts = get_options([])
    opts.load_path = "pre-trained/tsp100.pt"
    opts.graph_size = 100
    opts.problem = "tsp"
    opts.eval_only = True
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create problem instance
    print("Creating TSP problem instance...")
    problem = TSP(p_size=opts.graph_size)
    
    # Load agent
    print("Loading NeuOpt agent...")
    agent = PPO(problem, opts)
    agent.load(opts.load_path)
    agent.eval()
    
    # Print agent info
    print("\nAgent information:")
    print(f"Actor type: {type(agent.actor)}")
    
    # Load a sample instance
    print("\nCreating sample instance...")
    instance = {'coordinates': torch.rand(1, opts.graph_size, 2, device=opts.device)}
    
    # Print instance info
    print(f"Instance batch shape: {instance['coordinates'].shape}")
    
    # Try to get an initial solution
    print("\nGenerating initial solution...")
    initial_solution = problem.get_initial_solutions(instance)
    print(f"Initial solution shape: {initial_solution.shape}")
    
    # Calculate the cost
    print("\nCalculating solution cost...")
    cost = problem.get_costs(instance, initial_solution).item()
    print(f"Solution cost: {cost:.4f}")
    
    print("\nModel loaded and tested successfully!")

if __name__ == "__main__":
    main() 