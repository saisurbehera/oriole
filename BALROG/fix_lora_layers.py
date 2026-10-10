#!/usr/bin/env python3
"""
Check DialoGPT layer names for LoRA
"""

from transformers import AutoModelForCausalLM

def check_model_layers():
    """Check what layers DialoGPT actually has"""
    print("🔍 Checking DialoGPT layer structure...")
    
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    
    print("📋 Model layers:")
    for name, module in model.named_modules():
        if 'attn' in name.lower() or 'proj' in name.lower() or 'linear' in name.lower():
            print(f"   {name}: {type(module).__name__}")
    
    print("\n🎯 Suitable LoRA targets for DialoGPT:")
    targets = []
    for name, module in model.named_modules():
        if any(keyword in name for keyword in ['c_attn', 'c_proj', 'attn']):
            targets.append(name.split('.')[-1])
    
    unique_targets = list(set(targets))
    print(f"   {unique_targets}")
    
    return unique_targets

if __name__ == "__main__":
    targets = check_model_layers()