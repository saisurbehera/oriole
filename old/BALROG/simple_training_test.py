#!/usr/bin/env python3
"""
Simple Training Test
===================

Minimal test without complex dependencies.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

def simple_test():
    print("🎮 Simple NetHack Training Test")
    print("=" * 30)
    
    # Check basics
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🔥 CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    
    # Test dataset
    print("📚 Testing dataset...")
    try:
        dataset = load_from_disk("enhanced_nethack_dataset")
        print(f"✅ Dataset: {len(dataset['train'])} train, {len(dataset['validation'])} val")
        
        sample = dataset['train'][0]
        print(f"📋 Sample:")
        print(f"   Output: {sample['output']}")
        print(f"   Subgoal: {sample['subgoal']}")
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return
    
    # Test tokenizer
    print("🤖 Testing tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded")
        
        # Test tokenization
        test_text = "### Instruction:\nChoose a NetHack action\n\n### Response:\nnorth"
        tokens = tokenizer(test_text)
        print(f"✅ Tokenization: {len(tokens['input_ids'])} tokens")
        
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        return
    
    # Test model loading (simple)
    print("🧠 Testing model...")
    try:
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        if torch.cuda.is_available():
            model = model.cuda()
        print("✅ Model loaded")
        
        # Test inference
        inputs = tokenizer("Hello", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        print("✅ Model inference works")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"📊 GPU Memory: {memory_used:.2f} GB")
        
    except Exception as e:
        print(f"❌ Model error: {e}")
        return
    
    print("\n🎯 All basic tests passed!")
    print("💡 Ready for training setup")

if __name__ == "__main__":
    simple_test()