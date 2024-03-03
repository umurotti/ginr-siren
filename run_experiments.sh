#!/bin/bash

# Path to the Python script
PYTHON_SCRIPT="./main_stage_inr.py"

# Default directory containing the experiment configurations
DEFAULT_EXPERIMENTS_DIR="./experiments/overfit_relu_full_test"

# Use the supplied directory as the experiments directory if one is provided, otherwise use the default
EXPERIMENTS_DIR=${1:-$DEFAULT_EXPERIMENTS_DIR}

# Check if the experiments directory exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Experiments directory does not exist."
    exit 1
fi

# Function to run experiments given a directory
run_experiments() {
    local dir=$1
    echo "Processing experiment in directory: $dir"

    # Iterate over each YAML file in the directory
    for CONFIG_FILE in "$dir"/*.yaml; do
        # Check if the file exists
        if [ -f "$CONFIG_FILE" ]; then
            echo "Running experiment with config: $CONFIG_FILE"
            python "$PYTHON_SCRIPT" "-m=$CONFIG_FILE" "-t=$dir"
        fi
    done
}

# Check if the experiments directory contains YAML files directly
if ls "$EXPERIMENTS_DIR"/*.yaml 1> /dev/null 2>&1; then
    # If there are YAML files directly in the directory, run experiments on them
    run_experiments "$EXPERIMENTS_DIR"
else
    # Otherwise, iterate over each subdirectory and run experiments
    for EXP_DIR in "$EXPERIMENTS_DIR"/*; do
        if [ -d "$EXP_DIR" ]; then
            run_experiments "$EXP_DIR"
        fi
    done
fi
