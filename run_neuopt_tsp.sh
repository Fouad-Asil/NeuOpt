#!/bin/bash

# Activate the DL conda environment
echo "Activating DL conda environment..."
eval "$(conda shell.bash hook)"
conda activate DL

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate the DL conda environment."
    echo "Please make sure the conda environment 'DL' exists."
    echo "You can create it with: conda create -n DL python=3.8 pytorch torchvision matplotlib seaborn tqdm tensorboard pillow"
    exit 1
fi

# Set up CUDA environment variables if needed
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the NeuOpt TSP script with the provided arguments
echo "Running NeuOpt TSP solver with pretrained model..."
python neuopt_pretrained_tsp.py "$@"

echo "Done." 