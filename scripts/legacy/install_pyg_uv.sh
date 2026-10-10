#!/bin/bash
# Install PyTorch Geometric with uv

echo "Installing PyTorch Geometric packages with uv..."
echo "=============================================="

# First, let's check what's already installed
echo "Current packages:"
uv pip list | grep -E "torch|scatter|sparse|geometric"

# Install the PyG packages with the correct index URL
echo -e "\nInstalling torch-scatter..."
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo -e "\nInstalling torch-sparse..."
uv pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo -e "\nInstalling torch-cluster..."
uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo -e "\nInstalling torch-spline-conv..."
uv pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo -e "\nInstalling torch-geometric..."
uv pip install torch-geometric

# Verify installation
echo -e "\nVerifying installation..."
uv run python -c "
import sys
print(f'Python executable: {sys.executable}')

packages = ['torch', 'torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_geometric']
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f'✓ {pkg}: {mod.__version__ if hasattr(mod, \"__version__\") else \"OK\"}')
    except ImportError as e:
        print(f'✗ {pkg}: {e}')

# Test scatter operation
try:
    from torch_scatter import scatter
    import torch
    x = torch.randn(10, 3)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    out = scatter(x, index, dim=0, reduce='mean')
    print(f'\\n✓ Scatter operation works! Output shape: {out.shape}')
except Exception as e:
    print(f'\\n✗ Scatter operation failed: {e}')
"

echo -e "\nReady to test ULTRA!"
echo "Run: uv run python research/test_ultra_simple_working.py"