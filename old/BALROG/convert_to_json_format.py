#!/usr/bin/env python3
"""
Convert NetHack dataset to JSON format with reasoning
"""

from datasets import load_from_disk, Dataset, DatasetDict
import json
import random
from typing import Dict, List

# NetHack action explanations for reasoning
ACTION_REASONING = {
    "search": "Search for hidden doors, traps, or secret passages to find alternate routes and avoid dangers",
    "north": "Move north to explore uncharted areas and potentially find stairs or valuable items",
    "south": "Move south to continue exploration and search for dungeon features",
    "east": "Move east to systematically explore the current level",
    "west": "Move west to check unexplored areas for important features",
    "northeast": "Move northeast to reach diagonal areas efficiently",
    "northwest": "Move northwest to explore corner areas that might contain secrets",
    "southeast": "Move southeast to continue diagonal exploration",
    "southwest": "Move southwest to check remaining unexplored areas",
    "up": "Go up stairs to return to previous level or escape dangerous situations",
    "down": "Go down stairs to progress deeper into the dungeon",
    "pickup": "Pick up items that could be useful for survival or progression",
    "drop": "Drop unnecessary items to manage inventory space",
    "eat": "Eat food to maintain nutrition and avoid starvation",
    "drink": "Drink potions to gain beneficial effects or heal",
    "read": "Read scrolls or books to learn spells or gain magical effects",
    "wear": "Wear armor or accessories to improve defense and survival",
    "wield": "Wield weapons to be prepared for combat encounters",
    "open": "Open doors or containers to access new areas or items",
    "close": "Close doors to block monsters or create tactical advantages",
    "kick": "Kick doors or objects to open them when keys are unavailable",
    "wait": "Wait to let monsters approach or regenerate health safely",
    "rest": "Rest to recover health and mana when in a safe location",
    "look": "Look around to gather information about the current situation",
    "inventory": "Check inventory to manage items and plan next actions",
    "quit": "Quit the game when facing unsurmountable danger",
    "help": "Get help information to understand available commands",
    "save": "Save the game to preserve progress",
    "pray": "Pray to gods for divine intervention in desperate situations",
    "cast": "Cast spells to deal with monsters or solve puzzles",
    "zap": "Zap wands to use their magical effects",
    "apply": "Apply tools or special items to solve specific problems",
    "throw": "Throw items at monsters or to trigger mechanisms",
    "fire": "Fire ranged weapons at distant enemies",
    "dip": "Dip items in fountains or other liquids for special effects",
    "engrave": "Engrave messages or symbols for various purposes",
    "enhance": "Enhance weapon or spell skills through practice",
    "force": "Force locks or doors when other methods fail",
    "jump": "Jump over obstacles or to reach distant locations",
    "loot": "Loot containers or corpses for valuable items",
    "monster": "Interact with monsters through various means",
    "name": "Name items or monsters for better identification",
    "offer": "Offer items at altars for divine favor",
    "pay": "Pay for items or services in shops",
    "sit": "Sit on chairs or thrones for potential effects",
    "turn": "Turn undead creatures if you have the ability",
    "untrap": "Disarm traps to safely pass through dangerous areas",
    "version": "Check game version information",
    "wipe": "Wipe face to clear blindness or other effects",
    "yes": "Confirm actions when prompted by the game",
    "no": "Decline actions when prompted by the game"
}

def get_action_reasoning(action: str, context: Dict) -> str:
    """Get reasoning for an action based on context"""
    base_reasoning = ACTION_REASONING.get(action, f"Perform {action} action to progress in the game")
    
    # Add context-specific reasoning
    subgoal = context.get('subgoal', '')
    skill_level = context.get('expert_skill_level', 'novice')
    
    contextual_additions = []
    
    if subgoal == 'find_stairs':
        if action == 'search':
            contextual_additions.append("especially important for finding hidden stairs")
        elif action in ['north', 'south', 'east', 'west']:
            contextual_additions.append("to systematically search for stairs")
    
    if subgoal == 'explore':
        if action in ['north', 'south', 'east', 'west']:
            contextual_additions.append("to map out the current level")
    
    if skill_level == 'expert':
        contextual_additions.append("using expert knowledge of NetHack mechanics")
    elif skill_level == 'intermediate':
        contextual_additions.append("applying intermediate game strategy")
    
    if contextual_additions:
        return f"{base_reasoning}, {', '.join(contextual_additions)}"
    
    return base_reasoning

def convert_example_to_json(example: Dict) -> Dict:
    """Convert a single example to JSON format"""
    action = example['output']
    
    # Create context for reasoning
    context = {
        'subgoal': example.get('subgoal', ''),
        'expert_skill_level': example.get('expert_skill_level', 'novice'),
        'episode_points': example.get('episode_points', 0),
        'character_role': example.get('character_role', ''),
        'character_race': example.get('character_race', '')
    }
    
    reasoning = get_action_reasoning(action, context)
    
    # Create JSON output
    json_output = {
        "action": action,
        "reasoning": reasoning
    }
    
    # Update the example
    new_example = example.copy()
    new_example['output'] = json.dumps(json_output)
    
    return new_example

def convert_dataset_to_json_format():
    """Convert the entire dataset to JSON format"""
    print("🔄 Converting dataset to JSON format with reasoning...")
    
    # Load original dataset
    dataset = load_from_disk("./enhanced_nethack_dataset")
    
    print(f"📊 Original dataset:")
    print(f"   - Train: {len(dataset['train'])}")
    print(f"   - Validation: {len(dataset['validation'])}")
    
    # Convert training set
    print("🔄 Converting training set...")
    train_examples = []
    for i, example in enumerate(dataset['train']):
        if i % 200 == 0:
            print(f"   Progress: {i}/{len(dataset['train'])}")
        
        converted = convert_example_to_json(example)
        train_examples.append(converted)
    
    # Convert validation set
    print("🔄 Converting validation set...")
    val_examples = []
    for i, example in enumerate(dataset['validation']):
        if i % 50 == 0:
            print(f"   Progress: {i}/{len(dataset['validation'])}")
        
        converted = convert_example_to_json(example)
        val_examples.append(converted)
    
    # Create new dataset
    new_dataset = DatasetDict({
        'train': Dataset.from_list(train_examples),
        'validation': Dataset.from_list(val_examples)
    })
    
    # Save converted dataset
    output_path = "./enhanced_nethack_dataset_json"
    new_dataset.save_to_disk(output_path)
    
    print(f"✅ Converted dataset saved to: {output_path}")
    print(f"📊 New dataset:")
    print(f"   - Train: {len(new_dataset['train'])}")
    print(f"   - Validation: {len(new_dataset['validation'])}")
    
    # Show examples
    print(f"\n🔍 Sample converted examples:")
    for i in range(min(3, len(new_dataset['train']))):
        example = new_dataset['train'][i]
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {example['instruction'][:100]}...")
        print(f"Output: {example['output']}")
        
        # Verify JSON format
        try:
            parsed = json.loads(example['output'])
            print(f"✅ Valid JSON with keys: {list(parsed.keys())}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
    
    return new_dataset

if __name__ == "__main__":
    convert_dataset_to_json_format()