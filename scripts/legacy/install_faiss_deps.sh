#!/bin/bash

echo "Installing FAISS and dependencies for NetHack QA system..."

# Install FAISS (CPU version - faster to install)
echo "Installing FAISS..."
pip install faiss-cpu

# Ensure sentence-transformers is up to date
echo "Installing/updating sentence-transformers..."
pip install --upgrade sentence-transformers

# Install other dependencies
echo "Installing other dependencies..."
pip install pandas numpy scikit-learn

echo "Done! You can now run: python3 research/test_faiss_debug.py"