#!/usr/bin/env python3
"""
Create dataset with JSON format but no reasoning (just action)
"""

from datasets import load_from_disk, Dataset, DatasetDict
import json

def create_json_no_reasoning_instruction(example):
    """Create strategic instruction format with JSON action-only output"""
    
    # Extract action from existing data
    original_output = json.loads(example['output'])
    action = original_output['action']
    episode_points = example.get('episode_points', 0)
    
    # Create strategic format with JSON prompt but no reasoning
    strategic_instruction = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 07 Jul 2025

<|eot_id|><|start_header_id|>user<|end_header_id|>

=== NetHack Strategic Gameplay ===

CURRENT STATUS:
- Dungeon Level: 1
- HP: 16/16
- Score: {episode_points}
- Gold: 0
- XP Level: 1
- AC: 10 (lower is better)
- Turn: 17
- Hunger: Normal

ATTRIBUTES:
- STR: 18 DEX: 15 CON: 14
- INT: 10 WIS: 8 CHA: 12

MAP VIEW:
-----                 
                                           
                 |...|                 
                                                          +...|                 
                                                          |...|                 
 
                                                           |.dd.                 
                                                          |..@|                 
                                         
                   -+---
- Player Position: (61, 5)
- Stairs Position: not found

ITEMS NEARBY: []
MONSTERS: YES

MAP LEGEND:
@ = You  < = Stairs down  > = Stairs up  . = Floor  # = Wall  + = Door  
  d = Pet
) = Weapons  [ = Armor  ! = Potions  ? = Scrolls  = = Rings  % = Food

GAMEPLAY PRIORITIES:
1. SURVIVAL: Manage HP, hunger, avoid death
2. EQUIPMENT: Pick up weapons/armor, wield/wear them  
3. RESOURCES: Collect gold, potions, scrolls, food
4. COMBAT: Fight monsters for XP and loot
5. EXPLORATION: Find all items and map layout
6. PROGRESSION: Only descend when well-equipped

SITUATION ANALYSIS:
- On stairs: None
- Stairs visible: False  
- Items to collect: 0 items
- Combat ready: Needs armor
- Prepared for level 2: NO - get equipment first!

ACTIONS ALREADY TRIED: []

AVAILABLE ACTIONS:
MOVEMENT: north, south, east, west, northeast, northwest, southeast, southwest, up, down
BASIC: wait, search, look, pickup, drop, inventory  
ITEMS: eat, drink, read, wear, wield, apply
COMBAT: fight, fire, throw, zap
SPECIAL: pray, offer, engrave, open, kick

DECISION PRIORITIES:
EXPLORE FIRST!

1. EXPLORATION: 🔍 CRITICAL! Must explore to find stairs and monsters for score!
2. IMMEDIATE DANGER: If monsters attacking → fight or flee  
3. ITEMS ON GROUND: If items nearby → pickup valuable ones
4. EQUIPMENT NEEDS: If unequipped → wield weapons, wear armor
5. RESOURCE MANAGEMENT: If hungry → eat food, if injured → drink healing
6. PROGRESSION: If well-equipped and stairs found → descend

STRATEGIC DECISIONS:
🎯 STAIRS NOT FOUND - EXPLORATION MODE:

⭐ PRIMARY GOAL: Find stairs by systematic exploration!
- Move in unexplored directions: "north", "south", "east", "west"  
- Search corners and rooms: "search"
- Look for '<' symbol on map (stairs down)
- Kill any monsters encountered for score and XP

IMMEDIATE ACTIONS:
- If items nearby: "pickup" to collect them
- If holding weapon but not wielding: "wield" it
- If holding armor but not wearing: "wear" it  
- If monsters present: "fight" for score and XP!
- If injured (HP < 75%): look for healing potions to "drink"
- If hungry: "eat" food if available
- If stairs visible AND well-equipped: move toward stairs
- If on stairs AND prepared: "down" to descend
- If no stairs found: MOVE to explore! Try 'north', 'south', 'east', 'west'

Choose the MOST IMPORTANT action for long-term NetHack success:

Respond with JSON:
{{"action": "chosen_action"}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{"action": "{action}"}}"""

    return strategic_instruction, f'{{"action": "{action}"}}'

def create_json_no_reasoning_dataset():
    """Create dataset with JSON format but no reasoning"""
    print("🔄 Creating JSON format dataset (no reasoning)...")
    
    # Load JSON format dataset
    dataset = load_from_disk("./enhanced_nethack_dataset_json")
    
    print(f"📊 Original dataset:")
    print(f"   - Train: {len(dataset['train'])}")
    print(f"   - Validation: {len(dataset['validation'])}")
    
    # Convert training set
    print("🔄 Converting to JSON-no-reasoning format...")
    train_examples = []
    for i, example in enumerate(dataset['train']):
        if i % 200 == 0:
            print(f"   Progress: {i}/{len(dataset['train'])}")
        
        # Create JSON no-reasoning instruction
        strategic_instruction, json_output = create_json_no_reasoning_instruction(example)
        
        # Create new example
        new_example = {
            'instruction': strategic_instruction,
            'output': json_output,  # JSON with just action, no reasoning
            # Keep metadata
            **{k: v for k, v in example.items() if k not in ['instruction', 'output']}
        }
        
        train_examples.append(new_example)
    
    # Convert validation set
    print("🔄 Converting validation set...")
    val_examples = []
    for i, example in enumerate(dataset['validation']):
        if i % 50 == 0:
            print(f"   Progress: {i}/{len(dataset['validation'])}")
        
        strategic_instruction, json_output = create_json_no_reasoning_instruction(example)
        
        new_example = {
            'instruction': strategic_instruction,
            'output': json_output,
            **{k: v for k, v in example.items() if k not in ['instruction', 'output']}
        }
        
        val_examples.append(new_example)
    
    # Create new dataset
    json_no_reasoning_dataset = DatasetDict({
        'train': Dataset.from_list(train_examples),
        'validation': Dataset.from_list(val_examples)
    })
    
    # Save dataset
    output_path = "./json_no_reasoning_nethack_dataset"
    json_no_reasoning_dataset.save_to_disk(output_path)
    
    print(f"✅ JSON-no-reasoning dataset saved to: {output_path}")
    print(f"📊 New dataset:")
    print(f"   - Train: {len(json_no_reasoning_dataset['train'])}")
    print(f"   - Validation: {len(json_no_reasoning_dataset['validation'])}")
    
    # Show example
    print(f"\n🔍 Sample JSON-no-reasoning format:")
    example = json_no_reasoning_dataset['train'][0]
    print(f"Output: {example['output']}")
    
    # Verify JSON format
    try:
        parsed = json.loads(example['output'])
        print(f"✅ Valid JSON with keys: {list(parsed.keys())}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
    
    # Show instruction ending
    instruction_end = example['instruction'][-300:]
    print(f"\nInstruction ending:\n...{instruction_end}")
    
    return json_no_reasoning_dataset

if __name__ == "__main__":
    create_json_no_reasoning_dataset()