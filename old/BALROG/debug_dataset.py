#!/usr/bin/env python3
"""
Debug dataset format to see what's wrong
"""

from datasets import load_from_disk

def debug_dataset():
    """Debug the dataset format"""
    print("🔍 Loading dataset...")
    dataset = load_from_disk("./json_no_reasoning_nethack_dataset")
    
    print(f"📊 Dataset info:")
    print(f"   - Train samples: {len(dataset['train'])}")
    print(f"   - Columns: {dataset['train'].column_names}")
    
    # Check first example
    example = dataset['train'][0]
    print(f"\n🔍 First example:")
    print(f"   - Type of instruction: {type(example['instruction'])}")
    print(f"   - Type of output: {type(example['output'])}")
    print(f"   - Instruction length: {len(example['instruction'])}")
    print(f"   - Output: {example['output']}")
    
    # Format for SFT
    def format_for_sft(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            text = examples['instruction'][i] + examples['output'][i]
            texts.append(text)
        return {"text": texts}
    
    # Apply formatting
    formatted = dataset.map(format_for_sft, batched=True, remove_columns=dataset['train'].column_names)
    
    print(f"\n🔍 After formatting:")
    print(f"   - Columns: {formatted['train'].column_names}")
    
    first_formatted = formatted['train'][0]
    print(f"   - Type of text: {type(first_formatted['text'])}")
    print(f"   - Text length: {len(first_formatted['text'])}")
    print(f"   - First 200 chars: {first_formatted['text'][:200]}...")
    print(f"   - Last 100 chars: ...{first_formatted['text'][-100:]}")

if __name__ == "__main__":
    debug_dataset()