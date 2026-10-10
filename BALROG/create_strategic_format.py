#!/usr/bin/env python3
"""
Create dataset with detailed strategic format
"""

from datasets import load_from_disk, Dataset, DatasetDict
import json
import random

def create_strategic_instruction(example):
    """Create detailed strategic instruction format"""
    
    # Extract basic info from existing data
    action = json.loads(example['output'])['action']
    episode_points = example.get('episode_points', 0)
    character_role = example.get('character_role', 'Rog')
    character_race = example.get('character_race', 'Hum')
    
    # Create strategic format
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
{{"action": "chosen_action", "reasoning": "why_this_action_is_most_important_for_nethack_survival_and_progression"}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}"""

    return strategic_instruction

def create_strategic_dataset():
    """Create dataset with strategic format"""
    print("🔄 Creating strategic format dataset...")
    
    # Load JSON format dataset
    dataset = load_from_disk("./enhanced_nethack_dataset_json")
    
    print(f"📊 Original dataset:")
    print(f"   - Train: {len(dataset['train'])}")
    print(f"   - Validation: {len(dataset['validation'])}")
    
    # Convert training set
    print("🔄 Converting to strategic format...")
    train_examples = []
    for i, example in enumerate(dataset['train']):
        if i % 200 == 0:
            print(f"   Progress: {i}/{len(dataset['train'])}")
        
        # Create strategic instruction
        strategic_instruction = create_strategic_instruction(example)
        
        # Create new example with strategic format
        new_example = {
            'instruction': strategic_instruction,
            'output': example['output'],
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
        
        strategic_instruction = create_strategic_instruction(example)
        
        new_example = {
            'instruction': strategic_instruction,
            'output': example['output'],
            **{k: v for k, v in example.items() if k not in ['instruction', 'output']}
        }
        
        val_examples.append(new_example)
    
    # Create new dataset
    strategic_dataset = DatasetDict({
        'train': Dataset.from_list(train_examples),
        'validation': Dataset.from_list(val_examples)
    })
    
    # Save strategic dataset
    output_path = "./strategic_nethack_dataset"
    strategic_dataset.save_to_disk(output_path)
    
    print(f"✅ Strategic dataset saved to: {output_path}")
    print(f"📊 New dataset:")
    print(f"   - Train: {len(strategic_dataset['train'])}")
    print(f"   - Validation: {len(strategic_dataset['validation'])}")
    
    # Show example
    print(f"\n🔍 Sample strategic format:")
    example = strategic_dataset['train'][0]
    print(f"Instruction length: {len(example['instruction'])} characters")
    print(f"First 500 chars:\n{example['instruction'][:500]}...")
    print(f"Output: {example['output']}")
    
    return strategic_dataset

if __name__ == "__main__":
    create_strategic_dataset()