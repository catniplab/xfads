#!/bin/bash

# Get the environment name (already specified in environment.yaml)
ENV_NAME=$(grep '^name:' environment.yaml | awk '{print $2}')

# Get the directory of the specified environment
ENV_PATH=$(conda env list | grep "${ENV_NAME}" | awk '{print $3}')
POStACTIVATION_PATH=$ENV_PATH/etc/conda/activate.d

# Create a directory for the activation script
mkdir -p $ENV_PATH/etc/conda/activate.d

# Get the current directory using $PWD
PROJECT_DIR="$PWD"

# Assign the value of PROJECT_DIR to a txt file
echo "$PROJECT_DIR" > $POStACTIVATION_PATH/project_dir.txt

# Create the post-activation script
echo 'PYTHONPATH=$(cat "$(dirname "$0")/project_dir.txt")'> $ENV_PATH/etc/conda/activate.d/set_pythonpath.sh
