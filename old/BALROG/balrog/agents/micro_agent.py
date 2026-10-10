"""
Micro Agent - Extreme bootstrapping for 1B models
=================================================

This agent uses the LLM only for the absolute minimum:
1. Choosing between 2 pre-selected options
2. Everything else is deterministic

The key insight: We can get good performance by:
- Pre-computing most decisions
- Using the LLM as a simple tie-breaker
- Learning from successful patterns
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse


# ============================================================================
# MICRO DECISION FRAMEWORK
# ============================================================================

@dataclass
class MicroDecision:
    """A single micro-decision the LLM needs to make"""
    option_a: str
    option_b: str
    context: str  # Very short context


class MicroAgent(BaseAgent):
    """Ultra-minimal LLM usage agent"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Memory
        self.action_history = deque(maxlen=10)
        self.position_memory = deque(maxlen=20)
        self.last_score = 0
        self.last_depth = 0
        self.steps_without_progress = 0
        
        # Current plan
        self.current_plan = None
        self.plan_index = 0
        
        # Exploration patterns
        self.exploration_patterns = [
            # Systematic room exploration
            ["west", "west", "west", "north", "north", "east", "east", "east", "south", "south"],
            # Door checking pattern
            ["west", "kick", "west", "search", "east", "east", "kick", "east"],
            # Spiral out
            ["north", "east", "south", "south", "west", "west", "north", "north"],
        ]
        self.current_pattern_index = 0
        
    def act(self, obs, prev_action=None):
        """Main action selection"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        # Quick context extraction
        stairs_pos = self._find_stairs(obs)
        player_pos = self._find_player(obs)
        door_adjacent = self._check_door_adjacent(obs)
        
        # Update progress tracking
        score = obs.get("score", 0)
        depth = obs.get("depth", 0)
        
        if score > self.last_score or depth > self.last_depth:
            self.steps_without_progress = 0
            self.last_score = score
            self.last_depth = depth
            # Clear plan on progress
            self.current_plan = None
        else:
            self.steps_without_progress += 1
            
        # Decision tree with minimal LLM usage
        action = self._decide_action(stairs_pos, player_pos, door_adjacent, obs)
        
        # Record action
        self.action_history.append(action)
        if player_pos:
            self.position_memory.append(player_pos)
            
        return LLMResponse(
            model_id="micro_1b",
            completion=action,
            stop_reason="complete",
            input_tokens=0,
            output_tokens=0,
            reasoning=self._get_reasoning(stairs_pos, door_adjacent)
        )
        
    def _decide_action(self, stairs_pos, player_pos, door_adjacent, obs):
        """Main decision logic with minimal LLM calls"""
        
        # Priority 1: If at stairs, use them
        if stairs_pos and player_pos and self._at_location(player_pos, stairs_pos):
            # Determine stair type
            stair_char = self._get_char_at(obs, stairs_pos)
            return "up" if stair_char == '<' else "down"
            
        # Priority 2: If stairs visible, navigate to them
        if stairs_pos and player_pos:
            return self._simple_pathfind(player_pos, stairs_pos)
            
        # Priority 3: If door blocking, handle it
        if door_adjacent:
            # Simple door handling
            if self.action_history and self.action_history[-1] == "kick":
                return "open"  # Try open after kick
            else:
                return "kick"
                
        # Priority 4: If stuck, use LLM to choose direction
        if self.steps_without_progress > 20:
            return self._llm_choose_direction(obs)
            
        # Priority 5: Execute current plan if any
        if self.current_plan and self.plan_index < len(self.current_plan):
            action = self.current_plan[self.plan_index]
            self.plan_index += 1
            return action
            
        # Priority 6: Start new exploration pattern
        return self._start_exploration_pattern()
        
    def _find_stairs(self, obs):
        """Find stairs position"""
        text = obs.get("text", {})
        map_text = text.get("long_term_context", "")
        lines = map_text.split('\n')
        
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char in ['>', '<']:
                    return (y, x)
        return None
        
    def _find_player(self, obs):
        """Find player position"""
        text = obs.get("text", {})
        map_text = text.get("long_term_context", "")
        lines = map_text.split('\n')
        
        for y, line in enumerate(lines):
            if '@' in line:
                return (y, line.index('@'))
        return None
        
    def _check_door_adjacent(self, obs):
        """Check if door is adjacent"""
        text = obs.get("text", {})
        map_text = text.get("long_term_context", "").lower()
        
        # Simple check - if + is near @ in the text
        if '+' in map_text and '@' in map_text:
            # Very simple proximity check
            plus_idx = map_text.index('+')
            at_idx = map_text.index('@')
            return abs(plus_idx - at_idx) < 5  # Close enough
        return False
        
    def _at_location(self, pos1, pos2):
        """Check if two positions are the same"""
        return abs(pos1[0] - pos2[0]) <= 0 and abs(pos1[1] - pos2[1]) <= 0
        
    def _get_char_at(self, obs, pos):
        """Get character at position"""
        text = obs.get("text", {})
        map_text = text.get("long_term_context", "")
        lines = map_text.split('\n')
        
        if 0 <= pos[0] < len(lines) and 0 <= pos[1] < len(lines[pos[0]]):
            return lines[pos[0]][pos[1]]
        return ' '
        
    def _simple_pathfind(self, from_pos, to_pos):
        """Very simple pathfinding"""
        dy = to_pos[0] - from_pos[0]
        dx = to_pos[1] - from_pos[1]
        
        # Prefer moving in the dimension with larger distance
        if abs(dy) > abs(dx):
            return "north" if dy < 0 else "south"
        elif dx != 0:
            return "west" if dx < 0 else "east"
        else:
            return "wait"
            
    def _llm_choose_direction(self, obs):
        """Use LLM only to choose between 2 directions"""
        # Find which directions are likely safe
        safe_dirs = self._find_safe_directions(obs)
        
        if len(safe_dirs) >= 2:
            # Ask LLM to choose between first two options
            option_a = safe_dirs[0]
            option_b = safe_dirs[1]
            
            # Ultra-simple prompt
            prompt = f"A={option_a} B={option_b}\nChoose:"
            
            messages = [{"role": "user", "content": prompt}]
            
            try:
                response = self.client.generate(messages)
                answer = response.completion.strip().upper()
                
                if 'A' in answer:
                    return option_a
                elif 'B' in answer:
                    return option_b
            except:
                pass
                
        # Fallback
        return safe_dirs[0] if safe_dirs else "north"
        
    def _find_safe_directions(self, obs):
        """Find likely safe directions"""
        # Look for walls and obstacles
        text = obs.get("text", {})
        map_text = text.get("long_term_context", "")
        
        # Very simple heuristic - avoid recently used directions
        all_dirs = ["north", "south", "east", "west"]
        safe = []
        
        for d in all_dirs:
            # Avoid if we just came from there
            if self.action_history and len(self.action_history) >= 3:
                recent = list(self.action_history)[-3:]
                if recent.count(d) >= 2:
                    continue
            safe.append(d)
            
        return safe if safe else all_dirs
        
    def _start_exploration_pattern(self):
        """Start a new exploration pattern"""
        # Pick next pattern
        pattern = self.exploration_patterns[self.current_pattern_index]
        self.current_pattern_index = (self.current_pattern_index + 1) % len(self.exploration_patterns)
        
        # Set as current plan
        self.current_plan = pattern
        self.plan_index = 0
        
        # Return first action
        if self.current_plan:
            action = self.current_plan[0]
            self.plan_index = 1
            return action
            
        return "search"
        
    def _get_reasoning(self, stairs_pos, door_adjacent):
        """Get simple reasoning string"""
        if stairs_pos:
            return "Stairs found"
        elif door_adjacent:
            return "Door adjacent"
        elif self.steps_without_progress > 20:
            return "Stuck - choosing direction"
        elif self.current_plan:
            return f"Pattern {self.plan_index}/{len(self.current_plan)}"
        else:
            return "Exploring"
            
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.action_history.clear()
        self.position_memory.clear()
        self.last_score = 0
        self.last_depth = 0
        self.steps_without_progress = 0
        self.current_plan = None
        self.plan_index = 0
        self.current_pattern_index = 0