#!/usr/bin/env python3
"""
Debug the template matching
"""

from datasets import load_from_disk

def debug_templates():
    """Debug what the templates should be"""
    dataset = load_from_disk("./json_no_reasoning_nethack_dataset")
    
    # Apply the same formatting as in training
    def create_simple_format(example):
        original_instruction = example['instruction']
        
        # Find the strategic gameplay part
        start_marker = "=== NetHack Strategic Gameplay ==="
        if start_marker in original_instruction:
            strategic_part = original_instruction[original_instruction.find(start_marker):]
            strategic_part = strategic_part.replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "")
        else:
            strategic_part = "Choose NetHack action based on game state."
        
        instruction_part = f"### Instruction:\n{strategic_part}\n\n"
        response_part = f"### Response:\n{example['output']}"
        
        text = instruction_part + response_part
        return text
    
    # Format one example
    example_text = create_simple_format(dataset['train'][0])
    
    print("🔍 Formatted example:")
    print("=" * 80)
    print(example_text)
    print("=" * 80)
    
    # Check for template matches
    instruction_template = "### Instruction:"
    response_template = "### Response:"
    
    print(f"\n📋 Template analysis:")
    print(f"   - Instruction template '{instruction_template}' found: {instruction_template in example_text}")
    print(f"   - Response template '{response_template}' found: {response_template in example_text}")
    
    if response_template in example_text:
        response_start = example_text.find(response_template)
        print(f"   - Response starts at position: {response_start}")
        print(f"   - Response content: {example_text[response_start:]}")
    
    # Count total length
    print(f"\n📏 Length analysis:")
    print(f"   - Total length: {len(example_text)} characters")
    print(f"   - Estimated tokens: ~{len(example_text) // 4}")

if __name__ == "__main__":
    debug_templates()