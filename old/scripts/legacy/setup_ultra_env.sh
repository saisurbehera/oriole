#!/bin/bash
# Setup ULTRA conda environment

echo "🚀 Setting up ULTRA conda environment..."

# Create conda environment with Python 3.9
conda create -n ultra python=3.9 -y

# Activate environment
conda activate ultra

# Install PyTorch, CUDA, and PyG
echo "Installing PyTorch and PyG dependencies..."
conda install pytorch=2.1.0 pytorch-cuda=11.8 cudatoolkit=11.8 pytorch-scatter=2.1.2 pyg=2.4.0 -c pytorch -c nvidia -c pyg -c conda-forge -y

# Install additional dependencies
echo "Installing additional dependencies..."
conda install ninja easydict pyyaml -c conda-forge -y

# Install other required packages
conda install pandas numpy requests -y

# Install FastAPI for server
pip install fastapi uvicorn

echo "✅ ULTRA environment setup complete!"
echo ""
echo "To activate: conda activate ultra"
echo "To test: python research/test_ultra_simple_working.py"