import os
import json
import torch
import pprint
import numpy as np
import random
from tensorboard_logger import Logger as TbLogger
import warnings
from options import get_options
import argparse
from abc_comparison import run_comparison
from problems.problem_tsp import TSPDataset

from problems.problem_tsp import TSP
from problems.problem_cvrp import CVRP
from agent.ppo import PPO

def load_problem(name):
    problem = {
        'tsp': TSP,
        'cvrp': CVRP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def run(opts):

    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tb and not opts.distributed:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))
    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        
    # Save arguments so exact configuration can always be found
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    # Figure out what's the problem
    problem = load_problem(opts.problem)(
                            p_size = opts.graph_size,
                            init_val_met = opts.init_val_met,
                            with_assert = opts.use_assert,
                            DUMMY_RATE = opts.dummy_rate,
                            k = opts.k,
                            with_bonus = not opts.wo_bonus,
                            with_regular = not opts.wo_regular)
    
    # Figure out the RL algorithm
    agent = PPO(problem, opts)

    # Load data from load_path
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        agent.load(load_path)

    # Do validation only
    if opts.eval_only:
        # Load the validation datasets
        agent.start_inference(problem, opts.val_dataset, tb_logger)
        
    else:
        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
            agent.opts.epoch_start = epoch_resume + 1
    
        # Start the actual training loop
        agent.start_training(problem, opts.val_dataset, tb_logger)


def parse_args():
    parser = argparse.ArgumentParser(description='Run ABC vs NeuOpt-enhanced ABC comparison')
    
    parser.add_argument('--dataset', type=str, default='datasets/tsp_20.pkl',
                        help='Path to the TSP dataset')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of TSP instances to solve')
    parser.add_argument('--colony_size', type=int, default=100,
                        help='Size of the colony (number of employed/onlooker bees)')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='Maximum number of iterations')
    parser.add_argument('--limit', type=int, default=50,
                        help='Maximum number of trials before abandonment')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='Directory to save results')
    parser.add_argument('--model_path', type=str, default='pre-trained/tsp20.pt',
                        help='Path to the pretrained NeuOpt model')
    parser.add_argument('--seed', type=int, default=6666,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Run comparison
    print(f"Running comparison with colony size: {args.colony_size}, max iterations: {args.max_iterations}")
    run_comparison(
        dataset=args.dataset,
        num_samples=args.num_samples,
        colony_size=args.colony_size,
        max_iterations=args.max_iterations,
        limit=args.limit,
        output_dir=args.output_dir,
        model_path=args.model_path
    )
    
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    warnings.filterwarnings("ignore")
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main()
