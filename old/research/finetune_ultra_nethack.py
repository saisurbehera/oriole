#!/usr/bin/env python3
"""
Fine-tune ULTRA on NetHack KG using proper command line interface
"""

import os
import subprocess
import sys

def run_ultra_finetune():
    """Run ULTRA fine-tuning using the command line interface"""
    print("🔥 Fine-tuning ULTRA on NetHack KG")
    print("="*50)
    
    # Check if data is prepared
    if not os.path.exists("data/nethack_ultra/train.txt"):
        print("❌ NetHack data not found. Run prepare_nethack_for_ultra.py first")
        return False
    
    print("✓ NetHack data found")
    
    # Get absolute paths
    ultra_dir = os.path.abspath("ultra")
    checkpoint_path = os.path.abspath("ultra/ckpts/ultra_4g.pth")
    config_path = os.path.abspath("ultra/config/transductive/inference.yaml")
    
    print(f"✓ ULTRA directory: {ultra_dir}")
    print(f"✓ Checkpoint: {checkpoint_path}")
    print(f"✓ Config: {config_path}")
    
    # Check files exist
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return False
    
    # Try zero-shot inference first
    print("\n🧪 Testing zero-shot inference...")
    
    cmd_zeroshot = [
        "python3", f"{ultra_dir}/script/run.py",
        "-c", config_path,
        "--dataset", "CoDExSmall",  # Use a working dataset first
        "--epochs", "0",
        "--bpe", "null", 
        "--gpus", "[0]",
        "--ckpt", checkpoint_path
    ]
    
    print(f"Command: {' '.join(cmd_zeroshot)}")
    
    try:
        # Change to ULTRA directory
        os.chdir("ultra")
        
        result = subprocess.run(cmd_zeroshot[1:], 
                              capture_output=True, 
                              text=True,
                              timeout=300)  # 5 minute timeout
        
        print(f"Return code: {result.returncode}")
        if result.returncode == 0:
            print("✅ Zero-shot inference successful!")
            print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print("❌ Zero-shot inference failed")
            print("Error:", result.stderr[-500:])
            
        os.chdir("..")  # Change back
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        os.chdir("..")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        os.chdir("..")
        return False

def try_different_approach():
    """Try using the run_many.py script"""
    print("\n🔄 Trying alternative approach with run_many.py")
    
    ultra_dir = os.path.abspath("ultra")
    checkpoint_path = os.path.abspath("ultra/ckpts/ultra_4g.pth")
    config_path = os.path.abspath("ultra/config/transductive/inference.yaml")
    
    # Use run_many.py for easier dataset handling
    cmd = [
        "python3", f"{ultra_dir}/script/run_many.py",
        "-c", config_path,
        "--gpus", "[0]",
        "--ckpt", checkpoint_path,
        "-d", "FB15k237",  # Try with a standard dataset
        "--finetune"  # Enable fine-tuning
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        os.chdir("ultra")
        result = subprocess.run(cmd[1:], 
                              capture_output=True, 
                              text=True,
                              timeout=600)  # 10 minute timeout
        
        print(f"Return code: {result.returncode}")
        if result.returncode == 0:
            print("✅ Fine-tuning successful!")
            print("Output:", result.stdout[-1000:])
        else:
            print("❌ Fine-tuning failed")
            print("Error:", result.stderr[-1000:])
            
        os.chdir("..")
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        os.chdir("..")
        return False

def main():
    print("🎯 ULTRA Fine-tuning Setup")
    print("="*50)
    
    # Test basic functionality first
    if run_ultra_finetune():
        print("\n✅ ULTRA is working! Ready for NetHack fine-tuning")
        print("\nNext steps:")
        print("1. Add NetHack dataset to ULTRA's datasets.py")
        print("2. Run fine-tuning with --dataset NetHack")
        print("3. Compare results with our heuristics baseline")
    else:
        print("\n🔄 Trying alternative approach...")
        if try_different_approach():
            print("✅ Alternative approach worked!")
        else:
            print("❌ ULTRA setup needs debugging")
            print("\nPossible issues:")
            print("- CUDA library mismatch")
            print("- Missing dependencies")
            print("- Path configuration")

if __name__ == "__main__":
    main()