# Hybrid ACO-NeuOpt Implementation Status

This document explains the status of the hybrid implementation combining Ant Colony Optimization (ACO) with NeuOpt for solving routing problems.

## Overview

The goal of this hybrid approach is to combine traditional ACO meta-heuristics with machine learning-based local search from NeuOpt. The idea is to use ACO for global exploration of the solution space and NeuOpt for intensification/local search to improve solutions found by ACO.

## Current Status

The hybrid implementation is currently a work in progress. Here's the status of the development:

1. NeuOpt Model Loading:
   - ✅ Successfully loads pretrained NeuOpt models
   - ✅ Can generate initial solutions for TSP
   - ✅ Can evaluate solution costs
   
2. Basic ACO Implementation:
   - ✅ Pheromone trail initialization
   - ✅ Solution construction for TSP
   - ✅ Pheromone update mechanisms
   
3. Integration with NeuOpt:
   - ❌ Not fully functional yet
   - ⚠️ Encounters issues with the NeuOpt actor.forward method
   - ⚠️ Likely issues with tensor shapes or indexing in the Decoder

## Implementation Challenges

The main challenges encountered are:

1. **NeuOpt Internal Structure**: The NeuOpt model's internal forward pass has specific expectations about tensor shapes and integration that are difficult to manage in a hybrid setting.

2. **Decoder Implementation**: The neural k-opt decoder in NeuOpt expects specific tensor formats that make integration difficult.

3. **Parameter Management**: There are differences in how the TSP and CVRP problems are implemented, requiring special handling for each problem type.

## Next Steps

To continue development of the hybrid approach:

1. **Use Higher-Level API**: Instead of trying to directly call the actor.forward method, it might be better to use the agent's rollout method for improved solutions.

2. **Simple Prototype**: Focus first on a simplified hybrid where ACO generates initial solutions and NeuOpt attempts to improve them without deep integration.

3. **CVRP Implementation**: Once TSP is working, extend the implementation to handle CVRP problems.

## Testing

A simple testing script `test_neuopt_load.py` has been created to verify that NeuOpt models can be loaded and used to evaluate solutions. This provides a foundation for further development.

## Usage

To test the NeuOpt model loading and basic functionality:

```bash
python test_neuopt_load.py
```

For future development, the goal is to have a command like:

```bash
python hybrid_aco_neuopt.py --model_path pre-trained/tsp100.pt --problem tsp --graph_size 100 --num_ants 50 --max_iterations 100
```

## References

- NeuOpt paper: "Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-Opt" (NeurIPS 2023)
- ACO: Dorigo, M., & Stützle, T. (2004). Ant Colony Optimization. MIT Press. 