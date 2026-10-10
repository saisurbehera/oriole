#!/bin/bash
# Setup using conda for better CUDA compatibility

echo "Setting up Conda environment with CUDA 12.1"
echo "==========================================="

ENV_NAME="oriole_cuda121"

# Create conda environment
conda create -n $ENV_NAME python=3.11 -y

# Create activation script to use with your project
cat > activate_oriole.sh << EOF
#!/bin/bash
conda activate $ENV_NAME
EOF
chmod +x activate_oriole.sh

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch with CUDA 12.1 from conda
echo "Installing PyTorch from conda-forge..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install NumPy 1.x
conda install numpy"<2" -y

# Install PyTorch Geometric from conda
echo "Installing PyTorch Geometric..."
conda install pyg -c pyg -y

# Install other dependencies
echo "Installing other dependencies..."
pip install pandas scikit-learn sentence-transformers transformers tqdm easydict pyyaml

# Create test script
cat > test_conda_env.py << 'EOF'
import sys
import numpy as np
import torch

print("Environment Test")
print("="*50)
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test imports
try:
    import torch_scatter
    from torch_scatter import scatter
    print("torch-scatter: OK")
    
    # Quick test
    x = torch.randn(10, 3)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    out = scatter(x, index, dim=0, reduce='mean')
    print(f"Scatter test passed, output shape: {out.shape}")
except Exception as e:
    print(f"torch-scatter: {e}")

try:
    import torch_geometric
    print("torch-geometric: OK")
except Exception as e:
    print(f"torch-geometric: {e}")
EOF

# Run test
python test_conda_env.py

echo -e "\nSetup complete!"
echo "To activate: conda activate $ENV_NAME"
echo "Or: source activate_oriole.sh"