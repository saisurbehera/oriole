#!/usr/bin/env python3
"""
Clean Expert Data Pipeline
===========================

Cleans up TTY escape sequences and converts expert data to proper HuggingFace format.
"""

import json
import re
import logging
from typing import Dict, List, Optional
from pathlib import Path
import datasets
from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpertDataCleaner:
    """Cleans and processes expert NetHack data"""
    
    def __init__(self):
        # NetHack action mappings for consistency
        self.action_standardization = {
            'h': 'west', 'j': 'south', 'k': 'north', 'l': 'east',
            'y': 'northwest', 'u': 'northeast', 'b': 'southwest', 'n': 'southeast',
            ',': 'pickup', 'd': 'drop', 'i': 'inventory', 'w': 'wield', 'W': 'wear',
            'e': 'eat', 'q': 'drink', 'r': 'read', 'z': 'zap', 'f': 'fire',
            's': 'search', '.': 'wait', '<': 'up', '>': 'down',
            'o': 'open', 'c': 'close', 'a': 'apply', 'p': 'pay'
        }
        
    def clean_tty_escape_sequences(self, text: str) -> str:
        """Remove TTY escape sequences and binary data"""
        if not text:
            return ""
            
        # Remove ANSI escape sequences
        text = re.sub(r'\x1b\[[0-9;]*[mH]', '', text)
        text = re.sub(r'\x1b\[[ABCD]', '', text)  # Arrow keys
        text = re.sub(r'\x1b\[K', '', text)       # Clear to end of line
        text = re.sub(r'\x1b\[2J', '', text)     # Clear screen
        
        # Remove other control characters and binary data
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Remove timing data patterns (common in TTY recordings)
        text = re.sub(r'\d+\x00\x00\x00\x00', '', text)
        text = re.sub(r'[àèðñ].*?[\x00-\x1f]', '', text)
        
        # Clean up extra whitespace and newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def extract_game_state(self, observation: str) -> Dict[str, str]:
        """Extract structured game state from cleaned observation"""
        cleaned_obs = self.clean_tty_escape_sequences(observation)
        
        # Split into map and status sections
        sections = cleaned_obs.split('status:')
        
        map_section = ""
        status_section = ""
        
        if len(sections) >= 1:
            map_section = sections[0].replace('map:', '').strip()
            
        if len(sections) >= 2:
            status_section = sections[1].strip()
        
        # Extract meaningful map data
        map_lines = []
        for line in map_section.split('\n'):
            line = line.strip()
            # Keep lines that look like NetHack map content
            if line and any(c in line for c in '@#.+-|<>)[]!?%'):
                map_lines.append(line[:80])  # NetHack map width limit
        
        # Extract status information
        status_lines = []
        for line in status_section.split('\n'):
            line = line.strip()
            # Keep lines with game status info
            if line and any(pattern in line for pattern in ['HP:', 'Dlvl:', 'St:', 'T:']):
                status_lines.append(line)
        
        return {
            'map': '\n'.join(map_lines) if map_lines else 'No map data',
            'status': '\n'.join(status_lines) if status_lines else 'No status data'
        }
    
    def standardize_action(self, action: str) -> str:
        """Standardize action names"""
        action = action.strip().lower()
        
        # Direct mapping
        if action in self.action_standardization:
            return self.action_standardization[action]
        
        # Already standardized
        if action in ['north', 'south', 'east', 'west', 'northeast', 'northwest', 
                     'southeast', 'southwest', 'up', 'down', 'wait', 'search', 
                     'pickup', 'drop', 'inventory', 'wield', 'wear', 'eat', 'drink',
                     'read', 'zap', 'fire', 'open', 'close', 'apply', 'pay']:
            return action
        
        # Fallback mapping
        action_keywords = {
            'move': 'wait',
            'look': 'search', 
            'get': 'pickup',
            'take': 'pickup',
            'equip': 'wield',
            'use': 'apply'
        }
        
        for keyword, standard_action in action_keywords.items():
            if keyword in action:
                return standard_action
        
        return 'wait'  # Safe default
    
    def is_valid_game_step(self, step: Dict) -> bool:
        """Check if a game step has valid data"""
        try:
            # Must have required fields
            if not all(key in step for key in ['observation', 'action', 'subgoal']):
                return False
            
            # Observation must have meaningful content
            game_state = self.extract_game_state(step['observation'])
            if not game_state['map'] or game_state['map'] == 'No map data':
                return False
            
            # Action must be valid
            clean_action = self.standardize_action(step['action'])
            if clean_action == 'wait' and step['action'] not in ['wait', '.']:
                # Only allow wait if it was explicitly wait
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Invalid step: {e}")
            return False
    
    def clean_expert_data(self, input_file: str) -> List[Dict]:
        """Clean the expert training data"""
        logger.info(f"Loading expert data from {input_file}")
        
        with open(input_file, 'r') as f:
            raw_data = json.load(f)
        
        logger.info(f"Loaded {len(raw_data)} raw training examples")
        
        cleaned_data = []
        skipped_count = 0
        
        for i, step in enumerate(raw_data):
            try:
                if not self.is_valid_game_step(step):
                    skipped_count += 1
                    continue
                
                # Extract and clean game state
                game_state = self.extract_game_state(step['observation'])
                
                # Standardize action
                clean_action = self.standardize_action(step['action'])
                
                # Create cleaned step
                cleaned_step = {
                    'observation': f"Map:\n{game_state['map']}\n\nStatus:\n{game_state['status']}",
                    'action': clean_action,
                    'subgoal': step.get('subgoal', 'exploration'),
                    'reasoning': step.get('reasoning', f"Taking '{clean_action}' as strategic action."),
                    'step_num': step.get('step_num', i),
                    'episode_id': step.get('episode_id', f'episode_{i}')
                }
                
                cleaned_data.append(cleaned_step)
                
            except Exception as e:
                logger.debug(f"Failed to clean step {i}: {e}")
                skipped_count += 1
        
        logger.info(f"Cleaned {len(cleaned_data)} examples, skipped {skipped_count} invalid examples")
        return cleaned_data
    
    def format_for_sft_training(self, cleaned_data: List[Dict]) -> List[Dict]:
        """Format data for supervised fine-tuning"""
        sft_data = []
        
        for step in cleaned_data:
            # Create instruction-following format
            instruction = f"""You are an expert NetHack player. Given the current game state, choose the best action to achieve the subgoal.

Current Game State:
{step['observation']}

Current Subgoal: {step['subgoal']}

Choose the best action from: north, south, east, west, northeast, northwest, southeast, southwest, up, down, wait, search, pickup, drop, inventory, wield, wear, eat, drink, read, zap, fire, open, close, apply, pay

Action:"""

            response = f"{step['action']}"
            
            # Add reasoning for more detailed training
            if step.get('reasoning'):
                response += f"\n\nReasoning: {step['reasoning']}"
            
            sft_example = {
                'instruction': instruction,
                'input': '',  # No additional input needed
                'output': response,
                'episode_id': step['episode_id'],
                'step_num': step['step_num'],
                'subgoal': step['subgoal']
            }
            
            sft_data.append(sft_example)
        
        return sft_data
    
    def create_huggingface_dataset(self, sft_data: List[Dict]) -> DatasetDict:
        """Create HuggingFace dataset with train/validation split"""
        logger.info(f"Creating HuggingFace dataset from {len(sft_data)} examples")
        
        # Split data (90% train, 10% validation)
        split_idx = int(0.9 * len(sft_data))
        train_data = sft_data[:split_idx]
        val_data = sft_data[split_idx:]
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        logger.info(f"Created dataset: {len(train_data)} train, {len(val_data)} validation examples")
        return dataset_dict
    
    def save_cleaned_data(self, cleaned_data: List[Dict], output_file: str):
        """Save cleaned data to JSON"""
        with open(output_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        logger.info(f"Saved {len(cleaned_data)} cleaned examples to {output_file}")


def main():
    """Main cleaning pipeline"""
    input_file = "/home/sai/Desktop/oriole/BALROG/expert_training_data.json"
    cleaned_file = "/home/sai/Desktop/oriole/BALROG/expert_training_data_cleaned.json"
    hf_dataset_dir = "/home/sai/Desktop/oriole/BALROG/nethack_expert_dataset"
    
    cleaner = ExpertDataCleaner()
    
    # Step 1: Clean the raw data
    logger.info("Step 1: Cleaning expert data...")
    cleaned_data = cleaner.clean_expert_data(input_file)
    cleaner.save_cleaned_data(cleaned_data, cleaned_file)
    
    # Step 2: Format for SFT training
    logger.info("Step 2: Formatting for SFT training...")
    sft_data = cleaner.format_for_sft_training(cleaned_data)
    
    # Step 3: Create HuggingFace dataset
    logger.info("Step 3: Creating HuggingFace dataset...")
    dataset = cleaner.create_huggingface_dataset(sft_data)
    
    # Step 4: Save to disk
    logger.info("Step 4: Saving HuggingFace dataset...")
    dataset.save_to_disk(hf_dataset_dir)
    
    print(f"\n✅ Data cleaning complete!")
    print(f"📊 Dataset Statistics:")
    print(f"   - Total cleaned examples: {len(cleaned_data)}")
    print(f"   - Training examples: {len(dataset['train'])}")
    print(f"   - Validation examples: {len(dataset['validation'])}")
    print(f"📁 Files created:")
    print(f"   - Cleaned JSON: {cleaned_file}")
    print(f"   - HF Dataset: {hf_dataset_dir}/")
    
    # Show sample
    print(f"\n📋 Sample training example:")
    sample = dataset['train'][0]
    print(f"Instruction: {sample['instruction'][:200]}...")
    print(f"Output: {sample['output']}")


if __name__ == "__main__":
    main()