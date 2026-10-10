#!/bin/bash
# Direct wheel installation for PyTorch Geometric

echo "Installing PyTorch Geometric via direct wheel download..."
echo "======================================================"

# Create temp directory for downloads
mkdir -p tmp_wheels
cd tmp_wheels

# Download the wheels directly
echo "Downloading wheels for PyTorch 2.1.2 + CUDA 12.1..."

# torch-scatter
wget https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp311-cp311-linux_x86_64.whl

# torch-sparse  
wget https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_sparse-0.6.18%2Bpt21cu121-cp311-cp311-linux_x86_64.whl

# torch-cluster
wget https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_cluster-1.6.3%2Bpt21cu121-cp311-cp311-linux_x86_64.whl

# torch-spline-conv
wget https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt21cu121-cp311-cp311-linux_x86_64.whl

# Install the wheels
echo -e "\nInstalling wheels..."
cd ..
uv pip install tmp_wheels/*.whl
uv pip install torch-geometric

# Clean up
rm -rf tmp_wheels

# Test
echo -e "\nTesting installation..."
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')

from torch_scatter import scatter
print('torch-scatter: OK')

from torch_sparse import coalesce
print('torch-sparse: OK')

import torch_geometric
print(f'torch-geometric: {torch_geometric.__version__}')

print('\\nAll packages installed successfully!')
"