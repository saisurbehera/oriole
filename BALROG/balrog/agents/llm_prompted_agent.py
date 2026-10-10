"""
LLM-Prompted Agent Framework for BALROG
======================================

This agent uses structured prompting to get clean action decisions from LLMs,
especially optimized for smaller models like Llama 3.2 1B.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse


class PromptStrategy(Enum):
    """Different prompting strategies"""
    MULTIPLE_CHOICE = "multiple_choice"
    CONSTRAINED_GENERATION = "constrained_generation"
    EXAMPLE_BASED = "example_based"
    STEP_BY_STEP = "step_by_step"


@dataclass
class ActionContext:
    """Structured context for action selection"""
    # Map features
    stairs_visible: bool = False
    stairs_direction: Optional[str] = None  # "north", "east", etc.
    door_present: bool = False
    door_direction: Optional[str] = None
    
    # Entities
    monster_present: bool = False
    monster_adjacent: bool = False
    monster_direction: Optional[str] = None
    pet_present: bool = False
    item_visible: bool = False
    
    # Player state
    hp_critical: bool = False  # < 30%
    hp_low: bool = False      # < 50%
    hungry: bool = False
    
    # Exploration
    in_room: bool = True
    in_corridor: bool = False
    dead_end: bool = False
    junction: bool = False
    
    # Recent history
    last_action: Optional[str] = None
    repeated_actions: int = 0
    steps_without_progress: int = 0


class LLMPromptedAgent(BaseAgent):
    """Agent that uses structured prompting for better LLM responses"""
    
    def __init__(self, client_factory, prompt_builder, prompt_strategy=PromptStrategy.MULTIPLE_CHOICE):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.prompt_strategy = prompt_strategy
        
        # Memory
        self.action_history = deque(maxlen=20)
        self.position_history = deque(maxlen=10)
        self.successful_sequences = []
        
        # Tracking
        self.last_score = 0
        self.last_depth = 1
        self.steps_since_progress = 0
        
    def act(self, obs, prev_action=None):
        """Main action selection with structured prompting"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        # Extract context
        context = self._extract_context(obs)
        
        # Check progress
        self._update_progress(obs)
        
        # Get action using appropriate prompt strategy
        if self.prompt_strategy == PromptStrategy.MULTIPLE_CHOICE:
            action = self._get_action_multiple_choice(context)
        elif self.prompt_strategy == PromptStrategy.CONSTRAINED_GENERATION:
            action = self._get_action_constrained(context)
        elif self.prompt_strategy == PromptStrategy.EXAMPLE_BASED:
            action = self._get_action_with_examples(context)
        else:  # STEP_BY_STEP
            action = self._get_action_step_by_step(context)
            
        # Record action
        self.action_history.append(action)
        
        return LLMResponse(
            model_id=f"llm_prompted_{self.prompt_strategy.value}",
            completion=action,
            stop_reason="complete",
            input_tokens=0,
            output_tokens=0,
            reasoning=self._get_reasoning(context)
        )
        
    def _extract_context(self, obs: Dict) -> ActionContext:
        """Extract structured context from observation"""
        context = ActionContext()
        
        text = obs.get("text", {})
        long_term = text.get("long_term_context", "").lower()
        short_term = text.get("short_term_context", "").lower()
        
        # Find stairs - only > (down) and < (up) are stairs in NetHack
        stairs_chars = {'>', '<'}
        player_y, player_x = None, None
        stairs_y, stairs_x = None, None
        
        lines = long_term.split('\n')
        for y, line in enumerate(lines):
            if '@' in line:
                player_y, player_x = y, line.index('@')
            for stair in stairs_chars:
                if stair in line:
                    context.stairs_visible = True
                    stairs_y, stairs_x = y, line.index(stair)
                    break
                    
        if context.stairs_visible and player_y is not None and stairs_y is not None:
            if stairs_y < player_y:
                context.stairs_direction = "north"
            elif stairs_y > player_y:
                context.stairs_direction = "south"
            elif stairs_x < player_x:
                context.stairs_direction = "west"
            elif stairs_x > player_x:
                context.stairs_direction = "east"
                        
        # Detect doors
        context.door_present = '+' in long_term
        
        # Detect monsters (excluding pets)
        # Check for "tame" in the text to identify pets
        is_pet_present = "tame" in long_term
        
        monster_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        monster_chars -= {'@', 'I', 'd', 'f'}  # Remove player, common non-monsters, and typical pets
        
        # Only detect as monster if not a pet
        if not is_pet_present:
            for char in monster_chars:
                if char in long_term:
                    context.monster_present = True
                    # Check if adjacent (simplified)
                    if '@' in long_term and player_y is not None:
                        # Check actual adjacency
                        for line_idx, line in enumerate(lines):
                            if char in line:
                                char_x = line.index(char)
                                char_y = line_idx
                                # Check if within 1 square
                                if abs(char_x - player_x) <= 1 and abs(char_y - player_y) <= 1:
                                    context.monster_adjacent = True
                                    break
                    break
                
        # Items
        item_chars = {')', '[', '?', '!', '*', '(', '%', '$'}
        context.item_visible = any(c in long_term for c in item_chars)
        
        # Player state
        hp_match = re.search(r'hp:\s*(\d+)\((\d+)\)', short_term)
        if hp_match:
            current_hp = int(hp_match.group(1))
            max_hp = int(hp_match.group(2))
            context.hp_critical = current_hp < max_hp * 0.3
            context.hp_low = current_hp < max_hp * 0.5
            
        context.hungry = "hungry" in short_term or "weak" in short_term
        
        # Location type
        if '.' in long_term:
            context.in_room = True
            context.in_corridor = False
        elif '#' in long_term:
            context.in_corridor = True
            context.in_room = False
            
        # Recent actions
        if self.action_history:
            context.last_action = self.action_history[-1]
            # Count repetitions
            if len(self.action_history) >= 3:
                if all(a == context.last_action for a in list(self.action_history)[-3:]):
                    context.repeated_actions = 3
                    
        context.steps_without_progress = self.steps_since_progress
        
        return context
        
    def _get_action_multiple_choice(self, context: ActionContext) -> str:
        """Multiple choice prompt strategy"""
        prompt = "NETHACK ACTION SELECTION\n\n"
        
        # Describe situation
        prompt += "CURRENT SITUATION:\n"
        if context.hp_critical:
            prompt += "- CRITICAL: HP very low!\n"
        elif context.hp_low:
            prompt += "- WARNING: HP low\n"
            
        if context.stairs_visible:
            prompt += f"- Stairs visible to the {context.stairs_direction or 'unknown'}\n"
        if context.monster_adjacent:
            prompt += "- Monster adjacent!\n"
        elif context.monster_present:
            prompt += "- Monster in room\n"
        if context.door_present:
            prompt += "- Door present\n"
        if context.hungry:
            prompt += "- You are hungry\n"
            
        if context.repeated_actions >= 3:
            prompt += f"- You've repeated '{context.last_action}' 3 times\n"
            
        # Generate options based on context
        prompt += "\nCHOOSE YOUR ACTION:\n"
        
        options = []
        if context.monster_adjacent:
            options = ["A) wait", "B) kick", "C) north", "D) south", "E) east"]
        elif context.stairs_visible and context.stairs_direction:
            if context.stairs_direction == "north":
                options = ["A) north", "B) search", "C) wait", "D) east", "E) west"]
            elif context.stairs_direction == "south":
                options = ["A) south", "B) search", "C) wait", "D) east", "E) west"]
            elif context.stairs_direction == "east":
                options = ["A) east", "B) search", "C) wait", "D) north", "E) south"]
            else:  # west
                options = ["A) west", "B) search", "C) wait", "D) north", "E) south"]
        elif context.door_present:
            options = ["A) kick", "B) open", "C) search", "D) north", "E) south"]
        else:
            # General exploration
            options = ["A) north", "B) east", "C) south", "D) west", "E) search"]
            
        prompt += "\n".join(options)
        prompt += "\n\nRespond with ONLY the letter (A, B, C, D, or E):"
        
        # Get response
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.generate(messages)
            answer = response.completion.strip().upper()
            
            # Extract action from letter
            if answer in ['A', 'B', 'C', 'D', 'E']:
                idx = ord(answer) - ord('A')
                if idx < len(options):
                    # Extract action from option text
                    action_text = options[idx].split(')')[1].strip()
                    return action_text
                    
            # Fallback
            return "search"
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return "wait"
            
    def _get_action_constrained(self, context: ActionContext) -> str:
        """Constrained generation prompt strategy"""
        valid_actions = ["north", "south", "east", "west", "northeast", "southeast", 
                        "southwest", "northwest", "wait", "search", "kick", "open", 
                        "down", "up", "pickup"]
                        
        prompt = f"Valid actions: {', '.join(valid_actions)}\n\n"
        
        if context.stairs_visible:
            prompt += f"Stairs are {context.stairs_direction}. "
        if context.monster_adjacent:
            prompt += "Monster adjacent! "
        if context.hp_critical:
            prompt += "HP critical! "
            
        prompt += "\nBest action:"
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.generate(messages)
            action = response.completion.strip().lower()
            
            # Validate
            if action in valid_actions:
                return action
                
            # Try to find closest match
            for valid in valid_actions:
                if valid in action or action in valid:
                    return valid
                    
            return "search"
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return "wait"
            
    def _get_action_with_examples(self, context: ActionContext) -> str:
        """Example-based prompt strategy"""
        prompt = "Learn from these examples:\n\n"
        
        # Provide examples
        examples = [
            ("Stairs visible to north, no threats", "north"),
            ("Monster adjacent, in corridor", "wait"),
            ("Door blocking path", "kick"),
            ("Nothing special visible", "search"),
            ("HP low, monster present", "north"),  # flee
            ("Stairs visible to east", "east"),
            ("Item on ground", "pickup"),
        ]
        
        for situation, action in examples:
            prompt += f"Situation: {situation} → Action: {action}\n"
            
        # Current situation
        prompt += "\nCurrent situation: "
        
        parts = []
        if context.stairs_visible:
            parts.append(f"Stairs visible to {context.stairs_direction or 'unknown'}")
        if context.monster_adjacent:
            parts.append("Monster adjacent")
        elif context.monster_present:
            parts.append("Monster present")
        if context.hp_low:
            parts.append("HP low")
        if context.door_present:
            parts.append("Door present")
            
        if not parts:
            parts.append("Nothing special visible")
            
        prompt += ", ".join(parts)
        prompt += " → Action:"
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.generate(messages)
            action = response.completion.strip().lower()
            
            # Clean up response
            action = action.split()[0]  # Take first word
            return action if len(action) < 10 else "search"
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return "wait"
            
    def _get_action_step_by_step(self, context: ActionContext) -> str:
        """Step-by-step reasoning prompt strategy"""
        prompt = "THINK STEP BY STEP:\n\n"
        
        prompt += "1) What do I see?\n"
        if context.stairs_visible:
            prompt += f"   - Stairs to the {context.stairs_direction}\n"
        if context.monster_present:
            prompt += "   - Monster present\n"
        if context.door_present:
            prompt += "   - Door blocking path\n"
        if not any([context.stairs_visible, context.monster_present, context.door_present]):
            prompt += "   - Empty room\n"
            
        prompt += "\n2) What is my goal?\n"
        prompt += "   - Find stairs and descend\n"
        
        prompt += "\n3) What should I do?\n"
        if context.stairs_visible and context.stairs_direction:
            prompt += f"   - Move {context.stairs_direction} toward stairs\n"
        elif context.monster_adjacent:
            prompt += "   - Deal with monster (wait or kick)\n"
        elif context.door_present and not context.stairs_visible:
            prompt += "   - Open door to explore (kick or open)\n"
        elif not context.stairs_visible:
            prompt += "   - Explore the area (move west through doorway)\n"
        else:
            prompt += "   - Continue exploring\n"
            
        prompt += "\nTherefore, my action is:"
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.generate(messages)
            # Extract action from response
            action_line = response.completion.strip().lower()
            
            # Common patterns - check for exact words first
            words = action_line.split()
            valid_actions = ["north", "south", "east", "west", "northeast", "southeast", 
                           "southwest", "northwest", "wait", "kick", "search", "open", "down"]
            
            # Look for exact match first
            for word in words:
                if word in valid_actions:
                    return word
                    
            # Then check if action is mentioned in the text
            for action in valid_actions:
                if action in action_line:
                    return action
                    
            # Default to west if we need to explore
            if not context.stairs_visible:
                return "west"
            else:
                return "search"
                
        except Exception as e:
            print(f"LLM Error: {e}")
            return "wait"
            
    def _update_progress(self, obs: Dict):
        """Track progress"""
        score = obs.get("score", 0)
        depth = obs.get("depth", 1)
        
        if score > self.last_score or depth > self.last_depth:
            self.steps_since_progress = 0
            self.last_score = score
            self.last_depth = depth
        else:
            self.steps_since_progress += 1
            
    def _get_reasoning(self, context: ActionContext) -> str:
        """Get reasoning summary"""
        parts = []
        
        if context.stairs_visible:
            parts.append(f"Stairs {context.stairs_direction or 'visible'}")
        if context.monster_adjacent:
            parts.append("Monster adjacent")
        if context.hp_critical:
            parts.append("HP critical!")
        elif context.hp_low:
            parts.append("HP low")
            
        parts.append(f"Progress: {self.steps_since_progress} steps")
        
        return " | ".join(parts)
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.action_history.clear()
        self.position_history.clear()
        self.last_score = 0
        self.last_depth = 1
        self.steps_since_progress = 0