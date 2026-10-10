#!/usr/bin/env python3
"""
Show example of converted JSON format dataset
"""

from datasets import load_from_disk
import json

def show_json_example():
    """Show a detailed example of the JSON format dataset"""
    print("🔍 Loading JSON format dataset...")
    
    # Load the converted dataset
    dataset = load_from_disk("./enhanced_nethack_dataset_json")
    
    print(f"📊 Dataset info:")
    print(f"   - Train samples: {len(dataset['train'])}")
    print(f"   - Validation samples: {len(dataset['validation'])}")
    
    # Show one complete example
    print(f"\n📋 Complete example:")
    example = dataset['train'][0]
    
    print(f"🔹 Instruction:")
    print(f"   {example['instruction']}")
    
    print(f"\n🔹 Output (JSON format):")
    print(f"   {example['output']}")
    
    # Parse and show the JSON structure
    try:
        parsed_output = json.loads(example['output'])
        print(f"\n🔹 Parsed JSON structure:")
        print(f"   Action: {parsed_output['action']}")
        print(f"   Reasoning: {parsed_output['reasoning']}")
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
    
    # Show the Llama chat format that would be used for training
    print(f"\n🔹 Llama chat format for training:")
    instruction = example['instruction']
    response = example['output']
    
    chat_format = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert NetHack player. Respond with JSON format containing action and reasoning.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    
    print(chat_format)
    
    # Show metadata that's preserved
    print(f"\n🔹 Preserved metadata:")
    metadata_fields = [
        'session_id', 'user_id', 'episode_num', 'episode_points', 
        'character_role', 'character_race', 'expert_skill_level', 
        'subgoal', 'episode_outcome'
    ]
    
    for field in metadata_fields:
        if field in example:
            print(f"   {field}: {example[field]}")

if __name__ == "__main__":
    show_json_example()