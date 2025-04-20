# PSO TSP Solver with NeuOpt

This implementation combines Particle Swarm Optimization (PSO) with NeuOpt's neural reinforcement learning approach to solve Traveling Salesman Problems (TSP). It leverages NeuOpt's dynamic local search capabilities to enhance the PSO algorithm's performance.

## Overview

The implementation includes:

1. A standard PSO algorithm for TSP with velocity updates based on swap sequences
2. Integration with NeuOpt to enhance local search optimization
3. Visualization tools to analyze the solutions and convergence

## Requirements

* Python 3.10+
* PyTorch 1.13+
* NumPy
* Matplotlib
* tqdm

## Usage

### Basic Usage

To run the PSO TSP solver with randomly generated data:

```bash
python pso_tsp_solver.py --tsp_size 20 --num_particles 50 --max_iterations 100
```

### Using NeuOpt for Local Search

To use NeuOpt for enhanced local search, specify the path to a pre-trained NeuOpt model:

```bash
python pso_tsp_solver.py --tsp_size 20 --neuopt_model pre-trained/tsp20.pt
```

### Using Existing TSP Datasets

To use a dataset from the NeuOpt repository:

```bash
python pso_tsp_solver.py --data_path datasets/tsp_20.pkl --neuopt_model pre-trained/tsp20.pt
```

### Full Parameter List

```
--tsp_size INT       Number of cities for TSP (default: 20)
--data_path STR      Path to TSP data file (default: None, generates random data)
--num_particles INT  Number of particles for PSO (default: 50)
--max_iterations INT Maximum number of iterations (default: 100)
--inertia FLOAT      Inertia weight for PSO (default: 0.5)
--c1 FLOAT           Personal best influence coefficient (default: 1.5)
--c2 FLOAT           Global best influence coefficient (default: 1.5)
--neuopt_model STR   Path to pre-trained NeuOpt model (default: None)
--device STR         Device to use ('cuda' or 'cpu', default: 'cuda' if available)
--seed INT           Random seed (default: 42)
```

## How It Works

### PSO for TSP

The Particle Swarm Optimization algorithm for TSP works as follows:

1. Each particle represents a valid TSP tour (a permutation of cities)
2. Particle movement is defined through swap operations (velocities)
3. Each particle is influenced by its own best solution and the global best solution
4. The fitness is measured as the total distance of the tour

### NeuOpt Integration

The implementation integrates NeuOpt in the following way:

1. NeuOpt acts as a local search optimizer after each PSO iteration
2. For each particle, the current solution is fed into NeuOpt's pretrained policy
3. NeuOpt applies a series of k-opt moves to improve the solution
4. If NeuOpt finds a better solution, the particle's position is updated

This hybrid approach combines the global exploration capabilities of PSO with the powerful local search abilities of NeuOpt.

## Example Output

The implementation produces two visualizations:

1. A plot showing the fitness convergence over iterations
2. A visualization of the best TSP tour found

## Acknowledgements

This implementation builds upon the NeuOpt framework. For more details on NeuOpt, see the original repository and paper:

"Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-Opt" by Yining Ma, Zhiguang Cao, and Yeow Meng Chee. 

def calculate_fitness(self, tour):
    """Calculate the total distance of the tour"""
    coords = self.coordinates[tour]
    shift_coords = torch.roll(coords, shifts=1, dims=0)
    dists = torch.sqrt(((coords - shift_coords) ** 2).sum(dim=1))
    return dists.sum().item() 