#!/usr/bin/env python3
"""
Check the current dataset format
"""

from datasets import load_from_disk
import json

def check_dataset_format():
    """Check what format the dataset actually has"""
    print("🔍 Checking current dataset format...")
    
    # Load dataset
    dataset = load_from_disk("./enhanced_nethack_dataset")
    
    print(f"📊 Dataset info:")
    print(f"   - Train samples: {len(dataset['train'])}")
    print(f"   - Validation samples: {len(dataset['validation'])}")
    
    # Check columns
    print(f"\n📋 Column names:")
    for col in dataset['train'].column_names:
        print(f"   - {col}")
    
    # Check first few examples
    print(f"\n🔍 First 3 training examples:")
    for i in range(min(3, len(dataset['train']))):
        example = dataset['train'][i]
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {example['instruction'][:100]}...")
        print(f"Output: {example['output']}")
        
        # Check if output is JSON format
        try:
            parsed = json.loads(example['output'])
            print(f"✅ Output is valid JSON: {type(parsed)}")
        except json.JSONDecodeError:
            print(f"❌ Output is NOT JSON format")
    
    # Check what the actual outputs look like
    print(f"\n🎯 Sample outputs (first 10):")
    for i in range(min(10, len(dataset['train']))):
        output = dataset['train'][i]['output']
        print(f"   {i+1}: {output}")

if __name__ == "__main__":
    check_dataset_format()