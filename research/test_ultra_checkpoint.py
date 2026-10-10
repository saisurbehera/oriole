"""
Test ULTRA checkpoint files without full dependencies
"""

import os
import pickle

def examine_checkpoint(checkpoint_path):
    """Examine ULTRA checkpoint structure"""
    print(f"\nExamining checkpoint: {checkpoint_path}")
    print(f"File size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
    
    try:
        # Try to load with pickle to see structure
        import torch
        
        print("\nAttempting to load checkpoint...")
        # Load with map_location to CPU to avoid CUDA issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("✓ Checkpoint loaded successfully!")
        
        # Examine structure
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")
            
        if 'model' in checkpoint:
            print("\nModel state dict keys (first 10):")
            model_keys = list(checkpoint['model'].keys())
            for key in model_keys[:10]:
                print(f"  - {key}")
            print(f"  ... ({len(model_keys)} total keys)")
            
        if 'config' in checkpoint:
            print("\nConfig information available")
            
        if 'epoch' in checkpoint:
            print(f"\nTraining epoch: {checkpoint['epoch']}")
            
    except Exception as e:
        print(f"Could not fully load checkpoint: {e}")
        print("This is expected without PyTorch Geometric installed")


def check_environment():
    """Check what's available in the environment"""
    print("=== Environment Check ===")
    
    # Check for PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
    
    # Check for PyG
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("✗ PyTorch Geometric not found")
    
    # Check for other dependencies
    try:
        import yaml
        print("✓ PyYAML available")
    except ImportError:
        print("✗ PyYAML not found")
        
    try:
        import easydict
        print("✓ easydict available")
    except ImportError:
        print("✗ easydict not found")


def test_simple_inference_setup():
    """Test if we can set up for inference without full dependencies"""
    print("\n=== Testing Simple Setup ===")
    
    # Check config files
    config_path = "/home/sai/Desktop/oriole/ultra/config/transductive/inference.yaml"
    if os.path.exists(config_path):
        print(f"✓ Config file found: {config_path}")
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print("\nConfig structure:")
            for key in config.keys():
                print(f"  - {key}")
                
        except Exception as e:
            print(f"Could not load config: {e}")
    else:
        print("✗ Config file not found")


if __name__ == "__main__":
    print("ULTRA Checkpoint Examination")
    print("=" * 50)
    
    # Check environment
    check_environment()
    
    # Check checkpoints
    checkpoint_dir = "/home/sai/Desktop/oriole/ultra/ckpts"
    checkpoints = ["ultra_3g.pth", "ultra_4g.pth", "ultra_50g.pth", "ultraquery.pth"]
    
    print(f"\n=== Checkpoints in {checkpoint_dir} ===")
    for ckpt in checkpoints:
        path = os.path.join(checkpoint_dir, ckpt)
        if os.path.exists(path):
            examine_checkpoint(path)
        else:
            print(f"\n✗ {ckpt} not found")
    
    # Test setup
    test_simple_inference_setup()
    
    print("\n" + "=" * 50)
    print("Examination completed!")
    print("\nTo use ULTRA for inference, you would need:")
    print("1. pip install torch torch-geometric easydict pyyaml")
    print("2. The checkpoints are ready in ultra/ckpts/")
    print("3. The kg_qa_wrapper.py would then work with these checkpoints")