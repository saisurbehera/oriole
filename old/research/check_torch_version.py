#!/usr/bin/env python3
"""
Check PyTorch and CUDA versions for installing compatible packages
"""

import torch
import sys

print("PyTorch Installation Info")
print("="*50)

# PyTorch version
torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")

# CUDA availability and version
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    print(f"CUDA available: Yes")
    print(f"CUDA version: {cuda_version}")
    
    # Extract major.minor version for PyG compatibility
    cuda_major_minor = '.'.join(cuda_version.split('.')[:2])
    print(f"CUDA major.minor: {cuda_major_minor}")
    
    # Get PyTorch major.minor version
    torch_major_minor = '.'.join(torch_version.split('.')[:2])
    if '+' in torch_major_minor:
        torch_major_minor = torch_major_minor.split('+')[0]
    print(f"PyTorch major.minor: {torch_major_minor}")
    
    # Construct CUDA tag for PyG
    cuda_tag = f"cu{cuda_major_minor.replace('.', '')}"
    
    print(f"\nInstall commands for PyTorch Geometric:")
    print(f"pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch_major_minor}+{cuda_tag}.html")
    print(f"pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch_major_minor}+{cuda_tag}.html")
    print(f"pip install torch-geometric")
    
    print(f"\nOr all at once:")
    print(f"pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-{torch_major_minor}+{cuda_tag}.html")
    
else:
    print(f"CUDA available: No")
    cpu_version = '.'.join(torch_version.split('.')[:2])
    if '+' in cpu_version:
        cpu_version = cpu_version.split('+')[0]
    
    print(f"\nInstall commands for PyTorch Geometric (CPU):")
    print(f"pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-{cpu_version}+cpu.html")

# Additional system info
print(f"\nPython version: {sys.version}")
print(f"PyTorch built with: {torch.version.git_version if hasattr(torch.version, 'git_version') else 'N/A'}")

# Check if we have GPU
if torch.cuda.is_available():
    print(f"\nGPU Info:")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")