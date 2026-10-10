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
