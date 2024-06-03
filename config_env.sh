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
echo "$PROJECT_DIR" > $POStACTIVATION_PATH/set_pythonpath.txt

# Create the post-activation script
echo 'PYTHONPATH=$(cat "$(dirname "$0")/set_pythonpath.txt")'> $POStACTIVATION_PATH/set_pythonpath.sh

# Set Pytorch device
echo '#!/bin/bash' > $POStACTIVATION_PATH/set_pytorch_device.sh
echo '' >> $POStACTIVATION_PATH/set_pytorch_device.sh
echo '# Check if CUDA is available' >> $POStACTIVATION_PATH/set_pytorch_device.sh
echo 'if command -v nvidia-smi &> /dev/null' >> $POStACTIVATION_PATH/set_pytorch_device.sh
echo 'then' >> $POStACTIVATION_PATH/set_pytorch_device.sh
echo '    PYTORCH_DEVICE="cuda"' >> $POStACTIVATION_PATH/set_pytorch_device.sh
echo 'else' >> $POStACTIVATION_PATH/set_pytorch_device.sh
echo '    PYTORCH_DEVICE="cpu"' >> $POStACTIVATION_PATH/set_pytorch_device.sh
echo 'fi' >> $POStACTIVATION_PATH/set_pytorch_device.sh

echo '' >> $POStACTIVATION_PATH/set_pytorch_device.sh
echo '    PYTORCH_ENABLE_MPS_FALLBACK="0"' >> $POStACTIVATION_PATH/set_pytorch_device.sh

echo '' >> $POStACTIVATION_PATH/set_pytorch_device.sh
echo '    CUDA_VISIBLE_DEVICES=""' >> $POStACTIVATION_PATH/set_pytorch_device.sh

