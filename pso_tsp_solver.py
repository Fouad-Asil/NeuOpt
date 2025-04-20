import torch
import numpy as np
import random
import pickle
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import NeuOpt components
from agent.ppo import PPO
from problems.problem_tsp import TSP
from utils import augmentation
from options import get_options

class Particle:
    def __init__(self, num_cities, device):
        self.position = torch.randperm(num_cities).to(device)  # Current tour (permutation)
        self.velocity = []  # List of swaps to apply
        self.best_position = self.position.clone()  # Personal best tour
        self.fitness = float('inf')  # Current tour length
        self.best_fitness = float('inf')  # Personal best tour length
    
    def update_personal_best(self, new_fitness):
        if new_fitness < self.best_fitness:
            self.best_fitness = new_fitness
            self.best_position = self.position.clone()
            return True
        return False

class PsoTspSolver:
    def __init__(self, 
                 num_cities, 
                 coordinates, 
                 num_particles=50, 
                 max_iterations=100,
                 inertia_weight=0.5,
                 c1=1.5,  # personal best influence
                 c2=1.5,  # global best influence
                 neuopt_model_path=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.num_cities = num_cities
        self.coordinates = coordinates
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.c1 = c1
        self.c2 = c2
        self.device = device
        
        # Initialize NeuOpt components if model path is provided
        self.use_neuopt = neuopt_model_path is not None
        if self.use_neuopt:
            # Parse options for NeuOpt
            parser = argparse.ArgumentParser()
            opts = parser.parse_args([])
            
            # Problem configuration
            opts.problem = 'tsp'
            opts.graph_size = num_cities
            opts.k = 4  # Maximum k for k-opt moves
            
            # Device configuration
            opts.no_cuda = not torch.cuda.is_available()
            opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
            opts.device = self.device
            opts.distributed = False
            
            # Network architecture parameters
            opts.embedding_dim = 128
            opts.hidden_dim = 128
            opts.actor_head_num = 4
            opts.critic_head_num = 4
            opts.n_encode_layers = 3
            opts.normalization = 'layer'
            opts.v_range = 6.0
            
            # Feature flags
            opts.wo_RNN = False
            opts.wo_feature1 = True
            opts.wo_feature3 = True
            opts.wo_regular = True
            opts.wo_bonus = True
            opts.wo_MDP = True
            
            # Evaluation and training settings
            opts.eval_only = True
            opts.val_m = 1  # Number of augmentations
            opts.stall_limit = 10  # T_D2A in the paper
            opts.init_val_met = 'random'
            
            # Other required parameters
            opts.gamma = 0.999
            opts.T_max = 1000
            opts.no_progress_bar = False
            
            # Initialize the TSP problem
            self.problem = TSP(p_size=num_cities, init_val_met='random', k=opts.k)
            
            # Set the device for PyTorch operations
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.set_device(0)
            
            # Initialize the NeuOpt agent with the proper options object
            self.agent = PPO(
                self.problem,
                opts
            )
            
            # Load the pre-trained model using PPO's load method
            self.agent.load(neuopt_model_path)
            self.agent.eval()
        
        # Initialize particles
        self.particles = [Particle(num_cities, device) for _ in range(num_particles)]
        
        # Calculate initial fitness for all particles
        for particle in self.particles:
            particle.fitness = self.calculate_fitness(particle.position)
            particle.best_fitness = particle.fitness
        
        # Find the global best
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.update_global_best()
        
        # Tracking performance
        self.best_fitness_history = []
        
    def calculate_fitness(self, tour):
        """Calculate the total distance of the tour"""
        coords = self.coordinates[tour]
        shift_coords = torch.roll(coords, shifts=1, dims=0)
        dists = torch.sqrt(((coords - shift_coords) ** 2).sum(dim=1))
        return dists.sum().item()
    
    def apply_velocity(self, particle):
        """Apply the velocity (swaps) to the particle's position"""
        position = particle.position.clone()
        for i, j in particle.velocity:
            position[i], position[j] = position[j], position[i]
        return position
    
    def generate_velocity(self, source, target, probability):
        """Generate a velocity (list of swaps) to move from source towards target"""
        velocity = []
        if random.random() < probability:
            source_clone = source.clone()
            # Create mapping from city index to position in the tour
            pos_mapping = {city.item(): pos for pos, city in enumerate(source_clone)}
            
            # Find misplaced cities
            for i in range(len(source_clone)):
                if source_clone[i] != target[i]:
                    # City currently at position i
                    current_city = source_clone[i].item()
                    # City that should be at position i
                    target_city = target[i].item()
                    
                    # Safely find position of the target city in the current tour
                    if target_city not in pos_mapping:
                        continue  # Skip this city if not in the mapping
                        
                    target_city_pos = pos_mapping[target_city]
                    
                    # Swap the cities
                    source_clone[i], source_clone[target_city_pos] = source_clone[target_city_pos], source_clone[i]
                    
                    # Update the mapping
                    pos_mapping[current_city] = target_city_pos
                    pos_mapping[target_city] = i
                    
                    # Add this swap to velocity
                    velocity.append((i, target_city_pos))
        return velocity
    
    def update_velocities(self):
        """Update velocities for all particles"""
        # Ensure global_best_position is initialized
        if self.global_best_position is None:
            self.update_global_best()
            # If still None, can't update velocities
            if self.global_best_position is None:
                return
                
        for particle in self.particles:
            # Apply inertia (keep some previous velocity)
            new_velocity = particle.velocity.copy() if random.random() < self.inertia_weight else []
            
            # Move towards personal best
            personal_best_influence = self.generate_velocity(
                particle.position, 
                particle.best_position, 
                self.c1 * random.random()
            )
            new_velocity.extend(personal_best_influence)
            
            # Move towards global best
            global_best_influence = self.generate_velocity(
                particle.position, 
                self.global_best_position, 
                self.c2 * random.random()
            )
            new_velocity.extend(global_best_influence)
            
            # Update particle's velocity
            particle.velocity = new_velocity
    
    def apply_neuopt_local_search(self, particle):
        """Apply NeuOpt to enhance the current solution"""
        if not self.use_neuopt:
            return False
        
        # Batch size greater than 1 to avoid indexing issues in NeuOpt
        batch_size = 4
        
        # Prepare the batch data for NeuOpt
        batch = {
            'coordinates': self.coordinates.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)  # Create batch of identical coordinates
        }
        
        # Prepare initial solution for NeuOpt
        # Convert particle position to NeuOpt's expected format (next-node representation)
        rec = torch.zeros(batch_size, self.num_cities, dtype=torch.long, device=self.device)
        particle_pos = particle.position.unsqueeze(0).repeat(batch_size, 1)  # Add batch dimension and repeat
        
        for i in range(self.num_cities):
            current_idx = i
            next_idx = (i + 1) % self.num_cities
            for b in range(batch_size):
                rec[b, particle_pos[b, current_idx]] = particle_pos[b, next_idx]
        
        # Initialize NeuOpt state
        T_max = 50  # Number of steps for local search
        obj = torch.zeros(batch_size, 3, device=self.device)  # [current, best-so-far, tsp_best-so-far]
        cost, context = self.problem.get_costs(batch, rec, get_context=True)
        obj[:, 0] = cost  # Current cost
        obj[:, 1] = cost  # Best-so-far cost
        
        # Set up context2 tensor (required by the NeuOpt actor)
        context2 = torch.zeros(batch_size, 9, device=self.device)
        context2[:, -1] = 1
        
        # Apply NeuOpt for local search
        with torch.no_grad():
            action = None
            for _ in range(T_max):
                # Use actor directly to get action
                action = self.agent.actor(
                    self.problem,
                    batch,
                    self.problem.input_feature_encoding(batch),
                    rec,
                    context,
                    context2,
                    action
                )[0]  # Get first element from returned tuple (action)
                
                # Apply the action and get the updated state
                rec, _, obj, _, context, context2, _ = self.problem.step(
                    batch=batch,
                    rec=rec,
                    action=action,
                    obj=obj,
                    feasible_history=None,
                    t=0
                )
        
        # Find the best solution from the batch
        best_batch_idx = obj[:, 1].argmin()
        best_rec = rec[best_batch_idx]
        
        # Convert back to particle position format
        improved_solution = self.problem.get_order(best_rec.unsqueeze(0), return_solution=True)[0]
        
        # Calculate the new fitness
        new_fitness = self.calculate_fitness(improved_solution)
        
        # If NeuOpt found a better solution, update the particle
        if new_fitness < particle.fitness:
            particle.position = improved_solution
            particle.fitness = new_fitness
            particle.update_personal_best(new_fitness)
            return True
        
        return False
    
    def update_global_best(self):
        """Update the global best solution"""
        # Initialize global_best_position if it's None
        if self.global_best_position is None and self.particles:
            self.global_best_position = self.particles[0].position.clone()
            self.global_best_fitness = self.particles[0].best_fitness
        
        # Update with the best solution found
        for particle in self.particles:
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = particle.best_position.clone()
    
    def run(self):
        """Run the PSO algorithm"""
        progress_bar = tqdm(range(self.max_iterations))
        for iteration in progress_bar:
            # Apply NeuOpt local search to each particle
            if self.use_neuopt:
                for particle in self.particles:
                    self.apply_neuopt_local_search(particle)
            
            # Update velocities
            self.update_velocities()
            
            # Move particles
            for particle in self.particles:
                # Apply velocity to get new position
                new_position = self.apply_velocity(particle)
                particle.position = new_position
                
                # Calculate new fitness
                particle.fitness = self.calculate_fitness(particle.position)
                
                # Update personal best
                particle.update_personal_best(particle.fitness)
            
            # Update global best
            self.update_global_best()
            
            # Record best fitness for this iteration
            self.best_fitness_history.append(self.global_best_fitness)
            
            # Update progress bar
            progress_bar.set_description(f"Best fitness: {self.global_best_fitness:.4f}")
        
        return self.global_best_position, self.global_best_fitness, self.best_fitness_history
    
    def plot_fitness_history(self):
        """Plot the fitness history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history)
        plt.title('Best Fitness Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (Tour Length)')
        plt.grid(True)
        plt.show()
    
    def plot_solution(self, solution=None):
        """Plot the best solution found"""
        if solution is None:
            solution = self.global_best_position
        
        plt.figure(figsize=(10, 10))
        
        # Get coordinates for the tour
        tour_coords = self.coordinates[solution].cpu().numpy()
        
        # Plot cities
        plt.scatter(tour_coords[:, 0], tour_coords[:, 1], c='blue', s=50)
        
        # Plot tour
        for i in range(self.num_cities):
            plt.plot([tour_coords[i][0], tour_coords[(i+1) % self.num_cities][0]],
                     [tour_coords[i][1], tour_coords[(i+1) % self.num_cities][1]], 'k-', alpha=0.5)
        
        plt.title(f'TSP Solution: {self.global_best_fitness:.4f}')
        plt.grid(True)
        plt.show()

def load_tsp_data(file_path, num_instances=1):
    """Load TSP data from a pickle file"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract a subset of instances
    instances = data[:num_instances]
    
    return instances

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PSO TSP Solver with NeuOpt')
    parser.add_argument('--tsp_size', type=int, default=20, help='Number of cities for TSP')
    parser.add_argument('--data_path', type=str, default=None, help='Path to TSP data file')
    parser.add_argument('--num_particles', type=int, default=50, help='Number of particles for PSO')
    parser.add_argument('--max_iterations', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--inertia', type=float, default=0.5, help='Inertia weight')
    parser.add_argument('--c1', type=float, default=1.5, help='Personal best influence coefficient')
    parser.add_argument('--c2', type=float, default=1.5, help='Global best influence coefficient')
    parser.add_argument('--neuopt_model', type=str, default=None, help='Path to pre-trained NeuOpt model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load or generate TSP data
    if args.data_path:
        instances = load_tsp_data(args.data_path)
        coordinates = torch.tensor(instances[0]['coordinates'], dtype=torch.float32, device=args.device)
        num_cities = coordinates.shape[0]
    else:
        # Generate random TSP instance
        num_cities = args.tsp_size
        coordinates = torch.rand((num_cities, 2), device=args.device)
    
    # Initialize and run PSO solver
    solver = PsoTspSolver(
        num_cities=num_cities,
        coordinates=coordinates,
        num_particles=args.num_particles,
        max_iterations=args.max_iterations,
        inertia_weight=args.inertia,
        c1=args.c1,
        c2=args.c2,
        neuopt_model_path=args.neuopt_model,
        device=args.device
    )
    
    best_position, best_fitness, _ = solver.run()
    
    print(f"Best solution found: {best_fitness}")
    
    # Plot results
    solver.plot_fitness_history()
    solver.plot_solution()

if __name__ == "__main__":
    main() 