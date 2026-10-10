#!/usr/bin/env python3
"""
Check GPU availability and create faster test
"""

import torch
import os

print("🔍 Checking GPU Status:")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n🔍 Environment:")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Quick test
if torch.cuda.is_available():
    print("\n✅ GPU is available! ULTRA should use it.")
    print("Make sure to move tensors to GPU in the code.")
else:
    print("\n❌ No GPU detected. ULTRA will be VERY slow on CPU.")
    print("For 20k+ entities, it needs to score 20k candidates per query!")
    
print("\nFor faster testing without GPU:")
print("1. Test on smaller subset (10-20 queries only)")
print("2. Limit candidates to top 100 instead of all 20k")
print("3. Use heuristics to pre-filter candidates")