#!/usr/bin/env python3
"""
Check if dataset matches the expected detailed format
"""

from datasets import load_from_disk
import json

def check_actual_format():
    """Check the actual format of the dataset"""
    print("🔍 Checking actual dataset format...")
    
    # Load the JSON format dataset
    dataset = load_from_disk("./enhanced_nethack_dataset_json")
    
    # Check first example
    example = dataset['train'][0]
    
    print(f"📋 Current instruction format:")
    print(f"Length: {len(example['instruction'])} characters")
    print(f"First 200 chars: {example['instruction'][:200]}...")
    print(f"Last 200 chars: ...{example['instruction'][-200:]}")
    
    print(f"\n🔍 Does it contain expected format elements?")
    instruction = example['instruction']
    
    # Check for specific elements
    checks = [
        ("=== NetHack Strategic Gameplay ===", "=== NetHack Strategic Gameplay ===" in instruction),
        ("CURRENT STATUS:", "CURRENT STATUS:" in instruction),
        ("ATTRIBUTES:", "ATTRIBUTES:" in instruction),
        ("MAP VIEW:", "MAP VIEW:" in instruction),
        ("GAMEPLAY PRIORITIES:", "GAMEPLAY PRIORITIES:" in instruction),
        ("SITUATION ANALYSIS:", "SITUATION ANALYSIS:" in instruction),
        ("DECISION PRIORITIES:", "DECISION PRIORITIES:" in instruction),
        ("Respond with JSON:", "Respond with JSON:" in instruction),
    ]
    
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}")
    
    print(f"\n🔍 Current output format:")
    print(f"Output: {example['output']}")
    
    # Check if it's the simple format or detailed format
    if len(instruction) < 500:
        print(f"\n⚠️  ISSUE: Current format is SIMPLE, but you want DETAILED format")
        print(f"Current format appears to be the basic instruction format")
        print(f"Expected format should be much longer with detailed strategic analysis")
    else:
        print(f"\n✅ Format appears to be detailed strategic format")

if __name__ == "__main__":
    check_actual_format()