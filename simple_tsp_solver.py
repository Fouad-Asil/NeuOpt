#!/usr/bin/env python3
import os
import numpy as np
import torch
import random
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

# Import only the problem definition from the original code
from problems.problem_tsp import TSP

def parse_args():
    parser = argparse.ArgumentParser(description='Simple TSP solver with 2-opt')
    
    parser.add_argument('--graph_size', type=int, default=20,
                        help='Number of nodes in the TSP problem')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='Maximum number of 2-opt iterations')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_figs', action='store_true',
                        help='Save visualization figures to files')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                        help='Directory to save visualization outputs')
    
    return parser.parse_args()

def generate_random_tsp(graph_size):
    """Generate a random TSP instance with coordinates in [0,1]"""
    return {'coordinates': torch.FloatTensor(graph_size, 2).uniform_(0, 1)}

def visualize_tsp(coordinates, best_solution, title, save_fig=False, output_dir='visualization_output', filename='tsp_solution.png'):
    """Visualize a TSP solution"""
    # Get coordinates
    coords = coordinates.numpy()
    
    # Get solution order
    problem = TSP(p_size=len(coords))
    solution_order = problem.get_order(best_solution, return_solution=True)[0].cpu().numpy()
    
    # Create plot
    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], s=100, color='blue', alpha=0.7, edgecolor='black')
    
    # Plot the tour with arrows indicating direction
    for idx in range(len(solution_order)):
        current_node = solution_order[idx]
        next_node = solution_order[(idx + 1) % len(solution_order)]
        
        # Plot line
        x1, y1 = coords[current_node]
        x2, y2 = coords[next_node]
        
        # Calculate the arrow position (80% along the line)
        arrow_pos = 0.8
        arrow_x = x1 + arrow_pos * (x2 - x1)
        arrow_y = y1 + arrow_pos * (y2 - y1)
        
        # Plot with arrow
        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=1.5)
        plt.arrow(arrow_x, arrow_y, (x2 - arrow_x) * 0.1, (y2 - arrow_y) * 0.1, 
                  head_width=0.02, head_length=0.02, fc='red', ec='red')
    
    # Label the points
    for i, coord in enumerate(coords):
        plt.annotate(str(i), (coord[0] + 0.01, coord[1] + 0.01), fontsize=12)
    
    plt.title(title, fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    
    if save_fig:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_cost_history(costs_per_iteration, save_fig=False, output_dir='visualization_output', filename='cost_history.png'):
    """Plot the cost history over iterations"""
    plt.figure(figsize=(12, 6))
    iterations = np.arange(len(costs_per_iteration))
    
    # Plot cost history
    plt.plot(iterations, costs_per_iteration, 'b-', linewidth=2)
    plt.scatter(iterations, costs_per_iteration, color='red', s=30, alpha=0.6)
    
    # Mark the best cost
    best_cost_idx = np.argmin(costs_per_iteration)
    best_cost = costs_per_iteration[best_cost_idx]
    plt.scatter([best_cost_idx], [best_cost], color='green', s=100, zorder=5, edgecolor='black')
    plt.text(best_cost_idx, best_cost * 1.03, f'Best: {best_cost:.4f} at iter {best_cost_idx}', 
             fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
    
    # Add a horizontal line at the best cost
    plt.axhline(y=best_cost, color='green', linestyle='--', alpha=0.5)
    
    # Add annotations for initial and final costs
    plt.annotate(f'Initial: {costs_per_iteration[0]:.4f}', 
                 xy=(0, costs_per_iteration[0]),
                 xytext=(5, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.annotate(f'Final: {costs_per_iteration[-1]:.4f}', 
                 xy=(len(costs_per_iteration)-1, costs_per_iteration[-1]),
                 xytext=(-5, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='black'))
    
    # Add title and labels
    plt.title('Cost History Over Iterations', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Tour Length', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Add improvement percentage
    improvement = ((costs_per_iteration[0] - costs_per_iteration[-1]) / costs_per_iteration[0]) * 100
    plt.figtext(0.5, 0.01, f'Total improvement: {improvement:.2f}%', 
                ha='center', fontsize=12, bbox=dict(facecolor='lightyellow', alpha=0.8))
    
    if save_fig:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_distance_matrix(coordinates, best_solution, save_fig=False, output_dir='visualization_output', filename='distance_matrix.png'):
    """Create a heatmap visualization of the node-to-node distances in the TSP solution"""
    # Get coordinates
    coords = coordinates.numpy()
    n_nodes = len(coords)
    
    # Get solution order
    problem = TSP(p_size=n_nodes)
    solution_order = problem.get_order(best_solution, return_solution=True)[0].cpu().numpy()
    
    # Calculate distance matrix
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            # Euclidean distance
            distance_matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
    
    # Create matrix where connections in the tour are marked
    tour_connections = np.zeros((n_nodes, n_nodes))
    for idx in range(n_nodes):
        current_node = solution_order[idx]
        next_node = solution_order[(idx + 1) % n_nodes]
        tour_connections[current_node, next_node] = 1
    
    # Create combined heatmap
    plt.figure(figsize=(14, 6))
    
    # Distance matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(distance_matrix, cmap='viridis', annot=False)
    plt.title('Distance Matrix Between Nodes', fontsize=14)
    plt.xlabel('Node ID', fontsize=12)
    plt.ylabel('Node ID', fontsize=12)
    
    # Tour connections
    plt.subplot(1, 2, 2)
    mask = tour_connections == 0
    sns.heatmap(tour_connections, cmap='Reds', mask=mask, annot=False, cbar=False)
    plt.title('Tour Connections', fontsize=14)
    plt.xlabel('Node ID', fontsize=12)
    plt.ylabel('Node ID', fontsize=12)
    
    plt.tight_layout()
    
    if save_fig:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    plt.show()

def animate_optimization(tsp_instance, solution_history, cost_history, best_solution, save_fig=False, output_dir='visualization_output', filename='optimization_animation.gif'):
    """Create an animation of the optimization process"""
    # Get coordinates
    coords = tsp_instance['coordinates'].numpy()
    n_nodes = len(coords)
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Setup for TSP tour animation
    problem = TSP(p_size=n_nodes)
    
    # Initial plot for cost history
    x = np.arange(len(cost_history))
    line, = ax2.plot(x, cost_history, 'b-', linewidth=2)
    ax2.scatter(x, cost_history, color='red', s=30, alpha=0.6)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Tour Length", fontsize=12)
    ax2.set_title("Cost History", fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, len(cost_history)-1)
    ax2.set_ylim(min(cost_history) * 0.95, max(cost_history) * 1.05)
    
    # Add best point marker
    best_idx = np.argmin(cost_history)
    best_point, = ax2.plot([best_idx], [cost_history[best_idx]], 'go', markersize=10)
    
    # Function to create the tour plot
    def draw_tour(solution, axis):
        axis.clear()
        # Draw nodes
        axis.scatter(coords[:, 0], coords[:, 1], s=100, color='blue', alpha=0.7, edgecolor='black')
        
        # Draw tour
        solution_order = problem.get_order(solution, return_solution=True)[0].cpu().numpy()
        for idx in range(n_nodes):
            current_node = solution_order[idx]
            next_node = solution_order[(idx + 1) % n_nodes]
            x1, y1 = coords[current_node]
            x2, y2 = coords[next_node]
            axis.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=1.5)
        
        # Label nodes
        for i, coord in enumerate(coords):
            axis.annotate(str(i), (coord[0] + 0.01, coord[1] + 0.01), fontsize=12)
        
        axis.set_title("TSP Solution", fontsize=14)
        axis.set_xlim(-0.05, 1.05)
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.3)
        
    # Initialize animation function
    iterations = min(100, len(solution_history))
    
    def init():
        draw_tour(solution_history[0], ax1)
        return [line, best_point]
    
    def update(frame):
        i = frame
        # Update tour
        draw_tour(solution_history[i], ax1)
        
        # Update cost history line
        line.set_data(np.arange(i+1), cost_history[:i+1])
        
        # Update best point if needed
        curr_best_idx = np.argmin(cost_history[:i+1])
        best_point.set_data([curr_best_idx], [cost_history[curr_best_idx]])
        
        # Add iteration counter
        ax1.text(0.02, 0.02, f"Iteration: {i}", transform=ax1.transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        # Add cost info
        ax1.text(0.75, 0.02, f"Cost: {cost_history[i]:.4f}", transform=ax1.transAxes,
                fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        return [line, best_point]
    
    ani = FuncAnimation(fig, update, frames=iterations, init_func=init, blit=False, interval=200)
    plt.tight_layout()
    
    if save_fig:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            ani.save(os.path.join(output_dir, filename), writer='pillow', fps=5, dpi=100)
            print(f"Animation saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving animation: {e}")
    
    plt.show()

class SimpleTspSolver:
    """Simple TSP solver using 2-opt local search"""
    
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def solve(self, tsp_instance, max_iterations=100):
        """
        Solve a TSP instance using 2-opt local search
        
        Args:
            tsp_instance: A dictionary containing 'coordinates' tensor of shape [graph_size, 2]
            max_iterations: Maximum number of 2-opt iterations
            
        Returns:
            best_solution: The best solution found
            best_cost: The cost of the best solution
            cost_history: List of costs at each iteration
            solution_history: List of solutions at each iteration
        """
        # Create TSP problem instance
        problem = TSP(p_size=tsp_instance['coordinates'].size(0))
        
        # Convert to batch format and move to device
        batch = {'coordinates': tsp_instance['coordinates'].unsqueeze(0).to(self.device)}
        
        # Generate initial solution (greedy nearest neighbor)
        solution = self._greedy_tsp(batch)
        
        # Get initial cost
        original_cost = problem.get_costs(batch, solution)
        
        # Initialize tracking variables
        best_solution = solution.clone()
        best_cost = original_cost.clone()
        solution_history = [solution.clone().cpu()]
        cost_history = [original_cost.item()]
        
        print(f"Initial cost: {original_cost.item():.6f}")
        
        # Main optimization loop using 2-opt
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            # Try to improve with 2-opt
            solution, current_cost, improved = self._two_opt_step(solution, batch, problem)
            
            # Track progress
            solution_history.append(solution.clone().cpu())
            cost_history.append(current_cost.item())
            
            # Update best solution if improved
            if current_cost < best_cost:
                best_solution = solution.clone()
                best_cost = current_cost.clone()
            
            iteration += 1
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Cost: {current_cost.item():.6f}")
        
        print(f"Final cost: {current_cost.item():.6f}")
        print(f"Best cost: {best_cost.item():.6f}")
        print(f"Total iterations: {iteration}")
        
        return best_solution.cpu(), best_cost.item(), cost_history, solution_history
    
    def _greedy_tsp(self, batch):
        """
        Implement a greedy nearest neighbor solution for TSP
        """
        coordinates = batch['coordinates']
        batch_size, graph_size, _ = coordinates.size()
        
        # Initialize an empty solution
        solution = torch.zeros(batch_size, graph_size).long().to(self.device)
        
        # Start from node 0 for simplicity
        current_nodes = torch.zeros(batch_size, 1).long().to(self.device)
        
        # Track visited nodes
        visited = torch.zeros(batch_size, graph_size).bool().to(self.device)
        visited.scatter_(1, current_nodes, 1)
        
        # Build tour greedily
        for i in range(graph_size - 1):
            # Calculate distances from current nodes to all other nodes
            current_coords = coordinates.gather(1, current_nodes.unsqueeze(-1).expand(batch_size, 1, 2))
            distances = ((coordinates - current_coords) ** 2).sum(-1).sqrt()
            
            # Set distance to visited nodes to infinity
            distances.masked_fill_(visited, float('inf'))
            
            # Select closest unvisited node
            _, next_nodes = distances.min(dim=1)
            next_nodes = next_nodes.unsqueeze(1)
            
            # Update solution and mark as visited
            solution.scatter_(1, current_nodes, next_nodes)
            visited.scatter_(1, next_nodes, 1)
            current_nodes = next_nodes
        
        # Connect last node to first node (0)
        solution.scatter_(1, current_nodes, torch.zeros(batch_size, 1).long().to(self.device))
        
        return solution
    
    def _two_opt_step(self, solution, batch, problem):
        """
        Apply a single 2-opt improvement step
        
        Returns:
            solution: Updated solution
            cost: New solution cost
            improved: Whether any improvement was made
        """
        batch_size, graph_size = solution.size()
        
        # Get current solution order as a route
        route = problem.get_order(solution, return_solution=True)
        
        # Get current cost
        current_cost = problem.get_costs(batch, solution)
        
        # Try all possible 2-opt swaps
        best_delta = 0
        best_i, best_j = -1, -1
        
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
                new_cost = problem.get_costs(batch, new_solution)
                
                # If better, update
                delta = current_cost - new_cost
                if delta > best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        # Apply best swap if improvement found
        improved = best_delta > 0
        if improved:
            # Apply the best swap by extracting and flipping the segment
            segment = route[0, best_i+1:best_j+1].clone()
            route[0, best_i+1:best_j+1] = segment.flip(0)
            
            # Convert route to solution format
            for k in range(graph_size):
                idx = (k + 1) % graph_size
                solution[0, route[0, k]] = route[0, idx]
            
            current_cost -= best_delta
        
        return solution, current_cost, improved

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Generate a new random TSP instance
    print(f"Generating a random TSP instance with {args.graph_size} nodes...")
    tsp_instance = generate_random_tsp(args.graph_size)
    
    # Create a simple TSP solver
    print("Creating a simple 2-opt TSP solver...")
    solver = SimpleTspSolver()
    
    # Solve the TSP instance
    print("Solving TSP instance...")
    start_time = time.time()
    best_solution, best_cost, cost_history, solution_history = solver.solve(
        tsp_instance, max_iterations=args.max_iterations
    )
    solve_time = time.time() - start_time
    
    # Print results
    print(f"\nResults:")
    print(f"Best tour length: {best_cost:.6f}")
    print(f"Solve time: {solve_time:.2f} seconds")
    
    # Calculate improvement
    improvement = ((cost_history[0] - best_cost) / cost_history[0]) * 100
    print(f"Improvement: {improvement:.2f}%")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    # Create visualization directory if saving figures
    if args.save_figs and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 1. Visualize the TSP solution
    print("1. TSP Solution Visualization")
    visualize_tsp(
        tsp_instance['coordinates'], 
        best_solution, 
        f"TSP Solution (Length: {best_cost:.6f}, {args.graph_size} nodes)",
        save_fig=args.save_figs,
        output_dir=args.output_dir,
        filename=f"tsp{args.graph_size}_solution.png"
    )
    
    # 2. Plot the cost history
    print("2. Cost History Visualization")
    plot_cost_history(
        cost_history, 
        save_fig=args.save_figs,
        output_dir=args.output_dir,
        filename=f"tsp{args.graph_size}_cost_history.png"
    )
    
    # 3. Plot the distance matrix and tour connections
    print("3. Distance Matrix Visualization")
    plot_distance_matrix(
        tsp_instance['coordinates'], 
        best_solution, 
        save_fig=args.save_figs,
        output_dir=args.output_dir,
        filename=f"tsp{args.graph_size}_distance_matrix.png"
    )
    
    # 4. Animate the optimization process
    print("4. Optimization Animation")
    animate_optimization(
        tsp_instance, 
        solution_history,
        cost_history, 
        best_solution,
        save_fig=args.save_figs,
        output_dir=args.output_dir,
        filename=f"tsp{args.graph_size}_animation.gif"
    )

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main() 