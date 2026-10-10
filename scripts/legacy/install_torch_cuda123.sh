#!/bin/bash
# Install PyTorch with CUDA 12.3 support

echo "Installing PyTorch with CUDA 12.3 support..."
echo "================================================"

# For uv users
echo "Using uv:"
echo "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123"

# Or using regular pip
echo -e "\nOr using pip:"
echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123"

# Try the installation
if command -v uv &> /dev/null; then
    echo -e "\nRunning uv installation..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123
else
    echo -e "\nRunning pip installation..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123
fi

# After installation, install PyTorch Geometric
echo -e "\nNext, install PyTorch Geometric:"
echo "pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.1+cu121.html"