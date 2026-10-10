#!/bin/bash
# Create fresh environment with CUDA 12.1 compatible packages

echo "Creating fresh environment with CUDA 12.1 support"
echo "================================================"

# Environment name
ENV_NAME="oriole_cuda121"

# Create new conda environment (if using conda)
if command -v conda &> /dev/null; then
    echo "Creating conda environment..."
    conda create -n $ENV_NAME python=3.11 -y
    conda activate $ENV_NAME
elif command -v uv &> /dev/null; then
    echo "Creating uv environment..."
    uv venv $ENV_NAME --python 3.11
    source $ENV_NAME/bin/activate
else
    echo "Creating standard venv..."
    python3.11 -m venv $ENV_NAME
    source $ENV_NAME/bin/activate
fi

# Install PyTorch with CUDA 12.1
echo -e "\nInstalling PyTorch with CUDA 12.1..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric and dependencies
echo -e "\nInstalling PyTorch Geometric packages..."
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# Install other required packages
echo -e "\nInstalling other dependencies..."
pip install pandas numpy scikit-learn sentence-transformers transformers tqdm easydict pyyaml

# Verify installation
echo -e "\nVerifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

try:
    import torch_scatter
    print('torch-scatter: OK')
except:
    print('torch-scatter: FAILED')

try:
    import torch_sparse
    print('torch-sparse: OK')
except:
    print('torch-sparse: FAILED')

try:
    import torch_geometric
    print('torch-geometric: OK')
except:
    print('torch-geometric: FAILED')
"

echo -e "\nEnvironment setup complete!"
echo "To activate: source $ENV_NAME/bin/activate"