#!/bin/bash
# Fix PyTorch Geometric installation

echo "Fixing PyTorch Geometric installation..."
echo "========================================"

# First, check Python and PyTorch versions
echo "Checking versions..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Install PyTorch Geometric packages for PyTorch 2.1.1
echo -e "\nInstalling PyTorch Geometric packages..."

# Since uv is having issues, let's use pip directly in the venv
source .venv/bin/activate

# Install packages one by one to avoid conflicts
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric

echo -e "\nVerifying installation..."
python -c "import torch_scatter; print('torch-scatter OK')"
python -c "import torch_sparse; print('torch-sparse OK')"
python -c "import torch_geometric; print('torch-geometric OK')"