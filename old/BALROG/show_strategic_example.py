#!/usr/bin/env python3
"""
Show complete strategic format example
"""

from datasets import load_from_disk

def show_strategic_example():
    """Show complete strategic format example"""
    print("🔍 Loading strategic format dataset...")
    
    # Load the strategic dataset
    dataset = load_from_disk("./strategic_nethack_dataset")
    
    print(f"📊 Strategic Dataset:")
    print(f"   - Train: {len(dataset['train'])}")
    print(f"   - Validation: {len(dataset['validation'])}")
    
    # Show complete example
    example = dataset['train'][0]
    
    print(f"\n📋 COMPLETE STRATEGIC FORMAT EXAMPLE:")
    print(f"Instruction length: {len(example['instruction'])} characters")
    print(f"\n{example['instruction']}")
    
    print(f"\n🔍 Verification:")
    instruction = example['instruction']
    
    # Check all required elements
    required_elements = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
        "Cutting Knowledge Date: December 2023",
        "Today Date: 07 Jul 2025",
        "<|eot_id|><|start_header_id|>user<|end_header_id|>",
        "=== NetHack Strategic Gameplay ===",
        "CURRENT STATUS:",
        "ATTRIBUTES:",
        "MAP VIEW:",
        "GAMEPLAY PRIORITIES:",
        "SITUATION ANALYSIS:",
        "DECISION PRIORITIES:",
        "STRATEGIC DECISIONS:",
        "IMMEDIATE ACTIONS:",
        "Choose the MOST IMPORTANT action",
        'Respond with JSON:\n{"action": "chosen_action", "reasoning": "why_this_action_is_most_important_for_nethack_survival_and_progression"}',
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    ]
    
    print(f"\n✅ Format verification:")
    all_good = True
    for element in required_elements:
        if element in instruction:
            print(f"   ✅ {element[:50]}...")
        else:
            print(f"   ❌ Missing: {element[:50]}...")
            all_good = False
    
    if all_good:
        print(f"\n🎉 SUCCESS! Dataset matches your exact format specification!")
    else:
        print(f"\n⚠️  Some elements missing")

if __name__ == "__main__":
    show_strategic_example()