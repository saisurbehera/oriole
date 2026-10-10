#!/bin/bash
# Setup fresh UV environment with CUDA 12.1

echo "Setting up fresh UV environment with CUDA 12.1"
echo "=============================================="

# Remove old venv if exists
if [ -d ".venv_old" ]; then
    rm -rf .venv_old
fi

# Backup current venv
if [ -d ".venv" ]; then
    echo "Backing up current .venv to .venv_old..."
    mv .venv .venv_old
fi

# Create new venv with Python 3.11
echo "Creating new virtual environment..."
uv venv --python 3.11

# Create requirements file for CUDA 12.1
cat > requirements_cuda121.txt << EOF
# PyTorch with CUDA 12.1
torch==2.1.2+cu121
torchvision==0.16.2+cu121
torchaudio==2.1.2+cu121

# Core dependencies
pandas
numpy
scikit-learn
sentence-transformers
transformers
tqdm
easydict
pyyaml

# For testing
pytest
jupyter
matplotlib
EOF

# Install PyTorch first with CUDA 12.1
echo -e "\nInstalling PyTorch with CUDA 12.1..."
uv pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
echo -e "\nInstalling PyTorch Geometric..."
# We need to use pip for PyG packages as uv might have issues with custom indices
source .venv/bin/activate
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric

# Install other dependencies
echo -e "\nInstalling other dependencies..."
uv pip install pandas numpy scikit-learn sentence-transformers transformers tqdm easydict pyyaml

# Create verification script
cat > research/verify_cuda121.py << 'EOF'
import torch
import sys

print("="*50)
print("CUDA 12.1 Environment Verification")
print("="*50)

print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test imports
imports = [
    "torch_scatter",
    "torch_sparse", 
    "torch_cluster",
    "torch_geometric",
    "transformers",
    "sentence_transformers"
]

print("\nPackage imports:")
for pkg in imports:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError as e:
        print(f"  ✗ {pkg}: {str(e)}")

# Test scatter operation
try:
    from torch_scatter import scatter
    x = torch.randn(10, 3)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    out = scatter(x, index, dim=0, reduce='mean')
    print("\n✓ torch-scatter operations working!")
except Exception as e:
    print(f"\n✗ torch-scatter operations failed: {e}")

print("\nEnvironment ready for ULTRA and NetHack KG processing!")
EOF

# Run verification
echo -e "\nVerifying installation..."
uv run python research/verify_cuda121.py

echo -e "\nSetup complete! Your new CUDA 12.1 environment is ready."
echo "To use: uv run python <your_script.py>"