#!/bin/bash
# Fix NumPy version and run real ULTRA+LLM

echo "🔧 Fixing NumPy version..."
pip install "numpy<2"

echo "🚀 Running Real ULTRA + LLM Integration..."
python research/real_ultra_with_llm.py