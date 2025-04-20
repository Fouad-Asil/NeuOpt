import torch
from agent.ppo import PPO
from problems.problem_tsp import TSP
from options import get_options

# Test a simple import
print("Testing imports...")

# Get options without parsing CLI args
opts = get_options([])
opts.load_path = "pre-trained/tsp100.pt"
opts.graph_size = 100
opts.problem = "tsp"
opts.eval_only = True

# Create a problem instance
print("Creating problem instance...")
problem = TSP(p_size=opts.graph_size)

# Print TSP attributes
print("TSP attributes:")
print(dir(problem))

# Generate a sample batch
sample_batch = {'coordinates': torch.rand(1, problem.size, 2)}
print(f"Sample batch shape: {sample_batch['coordinates'].shape}")

# Try to create the PPO agent
print("Creating PPO agent...")
try:
    neuopt_agent = PPO(problem, opts)
    print("PPO agent created successfully")
    
    # Try to load the model
    print("Loading model...")
    neuopt_agent.load(opts.load_path)
    print("Model loaded successfully")
    
    # Set to evaluation mode
    neuopt_agent.eval()
    print("Agent set to evaluation mode")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 