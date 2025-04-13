# ABC-NeuOpt: Enhancing Artificial Bee Colony with Neural Optimization

This project integrates a trained neural optimization agent (NeuOpt) with the Artificial Bee Colony (ABC) algorithm to solve the Traveling Salesman Problem (TSP). The neural agent guides the local search in the employee bee phase, leading to improved solution quality.

## Overview

The Artificial Bee Colony (ABC) algorithm is a swarm intelligence optimization algorithm inspired by the foraging behavior of honey bees. This implementation enhances the local search capability of ABC by using a pre-trained neural network model (NeuOpt) to guide the k-opt moves during the employee bee phase.

The neural model was trained using reinforcement learning to learn effective local search moves. By incorporating this learned knowledge into ABC, we can achieve better solutions than using traditional random moves.

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- tqdm

## Directory Structure

```
.
├── abc_neuopt.py             # NeuOpt-enhanced ABC implementation
├── abc_comparison.py         # Script to compare original ABC and NeuOpt-enhanced ABC
├── run.py                    # Original NeuOpt inference script
├── options.py                # Command-line options for NeuOpt
├── problems/                 # Problem definitions (TSP, CVRP)
├── agent/                    # Neural models implementation
├── utils/                    # Utility functions
├── pre-trained/              # Pre-trained neural models
│   └── tsp20.pt              # Pre-trained model for TSP with 20 nodes
└── datasets/                 # Problem instances
    └── tsp_20.pkl            # TSP instances with 20 nodes
```

## Usage

### Running the NeuOpt-enhanced ABC algorithm

To solve TSP instances using the NeuOpt-enhanced ABC algorithm:

```bash
python abc_neuopt.py --dataset datasets/tsp_20.pkl --model pre-trained/tsp20.pt --colony_size 50 --max_iterations 100 --limit 20 --num_samples 10
```

Command-line arguments:
- `--dataset`: Path to the TSP dataset file
- `--model`: Path to the pre-trained NeuOpt model
- `--colony_size`: Number of employed/onlooker bees (default: 50)
- `--max_iterations`: Maximum number of iterations (default: 100)
- `--limit`: Maximum number of trials before abandonment (default: 20)
- `--num_samples`: Number of TSP instances to solve (default: 10)

### Comparing with original ABC algorithm

To compare the performance of NeuOpt-enhanced ABC with the original ABC algorithm:

```bash
python abc_comparison.py --dataset datasets/tsp_20.pkl --colony_size 30 --max_iterations 50 --limit 15 --num_samples 5 --output_dir comparison_results
```

Command-line arguments:
- `--dataset`: Path to the TSP dataset file
- `--colony_size`: Number of employed/onlooker bees (default: 30)
- `--max_iterations`: Maximum number of iterations (default: 50)
- `--limit`: Maximum number of trials before abandonment (default: 15)
- `--num_samples`: Number of TSP instances to solve (default: 5)
- `--output_dir`: Output directory for results and plots (default: comparison_results)

## How It Works

The integration of NeuOpt with ABC works as follows:

1. The ABC algorithm initializes a population of solutions (food sources).
2. During the employed bee phase, instead of making random local moves, we use the pre-trained neural network to suggest promising k-opt moves.
3. The neural network takes the current solution and instance as input and outputs an action that represents a k-opt move.
4. The suggested move is applied, and the solution is evaluated.
5. The remaining parts of ABC (onlooker and scout bee phases) follow the standard implementation.

This approach leverages the pattern recognition capabilities of neural networks to guide the search process more effectively.

## Expected Improvements

When compared to the original ABC algorithm, the NeuOpt-enhanced version typically shows:

1. Better solution quality (lower tour costs)
2. Faster convergence to good solutions
3. More consistent performance across different problem instances

The comparison script generates visualizations that highlight these improvements, including convergence plots and overall performance comparisons.

## Extending to Other Problems

While this implementation focuses on the TSP, the approach can be extended to other combinatorial optimization problems. You would need:

1. A trained neural model for the specific problem
2. Problem-specific solution representation and neighborhood operators
3. Proper integration with the ABC framework

## Acknowledgments

This implementation is based on the NeuOpt framework for neural combinatorial optimization. The pre-trained models were trained using reinforcement learning to learn effective local search strategies. 