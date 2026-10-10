#!/bin/bash
# Fix NumPy version and install PyTorch Geometric

echo "Fixing environment issues..."
echo "==========================="

# First, downgrade NumPy to 1.x
echo "Downgrading NumPy to version 1.x..."
uv pip install "numpy<2"

# Install PyTorch Geometric packages using pip directly in the venv
echo -e "\nInstalling PyTorch Geometric packages..."
source .venv/bin/activate

# Remove any existing installations
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -y

# Install specific versions that work with PyTorch 2.1.2 and CUDA 12.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric

# Verify the fix
echo -e "\nVerifying fixes..."
python -c "
import numpy as np
print(f'NumPy version: {np.__version__}')

import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

try:
    import torch_scatter
    from torch_scatter import scatter
    print('torch-scatter: OK')
    
    # Test scatter operation
    x = torch.randn(10, 3)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    out = scatter(x, index, dim=0, reduce='mean')
    print('scatter operation: OK')
except Exception as e:
    print(f'torch-scatter: FAILED - {e}')

try:
    import torch_geometric
    print('torch-geometric: OK')
except Exception as e:
    print(f'torch-geometric: FAILED - {e}')
"

echo -e "\nEnvironment fixed! Now you can run:"
echo "uv run python research/test_ultra_simple_working.py"