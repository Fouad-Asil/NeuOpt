#!/bin/bash

# Activate the DL conda environment
echo "Activating DL conda environment..."
eval "$(conda shell.bash hook)"
conda activate DL

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate the DL conda environment."
    echo "Please make sure the conda environment 'DL' exists."
    echo "You can create it with: conda create -n DL python=3.8 pytorch torchvision matplotlib seaborn tqdm"
    exit 1
fi

# Set up CUDA environment variables if needed
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the simple TSP solver with the provided arguments
echo "Running Simple TSP solver with 2-opt..."
python simple_tsp_solver.py "$@"

echo "Done." 