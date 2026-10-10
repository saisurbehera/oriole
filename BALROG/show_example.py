#!/usr/bin/env python3
"""
Show one example of the formatted dataset
"""

from datasets import load_from_disk

def show_example():
    """Show one complete example"""
    print("🔍 Loading dataset...")
    
    # Load original dataset
    dataset = load_from_disk("./json_no_reasoning_nethack_dataset")
    
    # Apply the same formatting as in the training script
    def create_simple_format(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            # Extract the strategic context from the original instruction
            original_instruction = examples['instruction'][i]
            
            # Find the strategic gameplay part (skip the chat headers)
            start_marker = "=== NetHack Strategic Gameplay ==="
            if start_marker in original_instruction:
                # Extract from strategic gameplay onwards
                strategic_part = original_instruction[original_instruction.find(start_marker):]
                # Remove the chat format ending
                strategic_part = strategic_part.replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "")
            else:
                # Fallback to simple instruction
                strategic_part = "Choose NetHack action based on game state."
            
            # Create simple format with game context
            instruction_part = f"### Instruction:\n{strategic_part}\n\n"
            response_part = f"### Response:\n{examples['output'][i]}"
            
            text = instruction_part + response_part
            texts.append(text)
        
        return {"text": texts}
    
    # Format one example
    first_example = {"instruction": [dataset['train'][0]['instruction']], "output": [dataset['train'][0]['output']]}
    formatted = create_simple_format(first_example)
    
    print("=" * 80)
    print("📋 COMPLETE FORMATTED EXAMPLE")
    print("=" * 80)
    print(formatted['text'][0])
    print("=" * 80)
    print(f"📊 Example stats:")
    print(f"   - Length: {len(formatted['text'][0])} characters")
    print(f"   - Contains strategic context: {'=== NetHack Strategic Gameplay ===' in formatted['text'][0]}")

if __name__ == "__main__":
    show_example()