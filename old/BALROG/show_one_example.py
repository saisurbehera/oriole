#!/usr/bin/env python3
"""
Show one complete example from strategic dataset
"""

from datasets import load_from_disk

def show_one_example():
    """Show one complete example"""
    print("🔍 Loading strategic dataset...")
    
    # Load dataset
    dataset = load_from_disk("./strategic_nethack_dataset")
    
    # Get one example
    example = dataset['train'][0]
    
    print("=" * 80)
    print("📋 COMPLETE STRATEGIC FORMAT EXAMPLE")
    print("=" * 80)
    
    print(example['instruction'])
    
    print("\n" + "=" * 80)
    print("📊 METADATA")
    print("=" * 80)
    
    metadata_fields = [
        'session_id', 'user_id', 'episode_num', 'episode_points',
        'character_role', 'character_race', 'expert_skill_level',
        'subgoal', 'episode_outcome'
    ]
    
    for field in metadata_fields:
        if field in example:
            print(f"{field}: {example[field]}")

if __name__ == "__main__":
    show_one_example()