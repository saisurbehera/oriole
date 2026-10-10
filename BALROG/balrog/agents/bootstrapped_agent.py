"""
Bootstrapped Agent for Small LLMs (1B parameters)
=================================================

This agent uses heavy scaffolding to make small LLMs effective at NetHack.
Key principles:
1. Binary decisions instead of open-ended generation
2. State machine with clear transitions
3. Heuristic pre-filtering of options
4. Minimal LLM calls only for ambiguous cases
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Set
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse


# ============================================================================
# STATE MACHINE FOR SMALL MODELS
# ============================================================================

class GameState(Enum):
    """High-level game states that determine behavior"""
    EXPLORING = "exploring"          # No stairs found yet
    NAVIGATING = "navigating"        # Moving to known stairs
    COMBAT = "combat"               # Monster nearby
    DOOR_BLOCKED = "door_blocked"   # Path blocked by door
    STUCK = "stuck"                 # No progress for many turns
    AT_STAIRS = "at_stairs"         # Ready to descend


@dataclass
class GameContext:
    """Simplified context for decision making"""
    state: GameState
    
    # Key features
    stairs_pos: Optional[Tuple[int, int]] = None
    player_pos: Optional[Tuple[int, int]] = None
    door_direction: Optional[str] = None
    safe_directions: List[str] = None
    
    # Simple flags
    monster_adjacent: bool = False
    at_stairs: bool = False
    blocked: bool = False
    
    def __post_init__(self):
        if self.safe_directions is None:
            self.safe_directions = []


# ============================================================================
# BOOTSTRAPPED AGENT
# ============================================================================

class BootstrappedAgent(BaseAgent):
    """Agent optimized for small LLMs through heavy scaffolding"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # State tracking
        self.current_state = GameState.EXPLORING
        self.action_history = deque(maxlen=20)
        self.position_history = deque(maxlen=10)
        self.stuck_counter = 0
        self.last_score = 0
        
        # Learned patterns (will be populated over time)
        self.successful_sequences = []
        
        # Pre-computed action sequences for common situations
        self.action_sequences = {
            "spiral_explore": ["north", "north", "east", "east", "south", "south", "south", "west", "west", "west", "north"],
            "door_sequence": ["kick", "kick", "kick", "open"],
            "combat_corridor": ["wait", "wait", "kick"],
            "escape_pattern": ["northwest", "north", "northeast", "east"],
        }
        
        # Current sequence being executed
        self.current_sequence = None
        self.sequence_index = 0
        
    def act(self, obs, prev_action=None):
        """Main action selection with heavy scaffolding"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        # Store last observation
        self.last_obs = obs
        
        # Extract simplified context
        context = self._extract_game_context(obs)
        
        # Update state machine
        self._update_state_machine(context, obs)
        
        # Get action based on current state
        action = self._get_state_based_action(context)
        
        # Record action
        self.action_history.append(action)
        
        return LLMResponse(
            model_id="bootstrapped_1b",
            completion=action,
            stop_reason="complete",
            input_tokens=0,
            output_tokens=0,
            reasoning=f"State: {self.current_state.value}"
        )
        
    def _extract_game_context(self, obs: Dict) -> GameContext:
        """Extract simplified context from observation"""
        context = GameContext(state=self.current_state)
        
        text = obs.get("text", {})
        map_text = text.get("long_term_context", "").lower()
        
        # Find player position
        lines = map_text.split('\n')
        for y, line in enumerate(lines):
            if '@' in line:
                context.player_pos = (y, line.index('@'))
                break
                
        # Find stairs
        for y, line in enumerate(lines):
            if '>' in line or '<' in line:
                x = line.index('>') if '>' in line else line.index('<')
                context.stairs_pos = (y, x)
                # Check if at stairs
                if context.player_pos and context.stairs_pos:
                    if abs(context.player_pos[0] - context.stairs_pos[0]) <= 1 and \
                       abs(context.player_pos[1] - context.stairs_pos[1]) <= 1:
                        context.at_stairs = True
                break
                
        # Check for monsters (excluding pets)
        context.monster_adjacent = self._check_monster_adjacent(lines, context.player_pos)
        
        # Find safe directions (no walls)
        context.safe_directions = self._find_safe_directions(lines, context.player_pos)
        
        # Check for doors
        if '+' in map_text and context.player_pos:
            context.door_direction = self._find_door_direction(lines, context.player_pos)
            
        return context
        
    def _update_state_machine(self, context: GameContext, obs: Dict):
        """Update state based on context"""
        score = obs.get("score", 0)
        
        # Check if we're making progress
        if score > self.last_score:
            self.stuck_counter = 0
            self.last_score = score
        else:
            self.stuck_counter += 1
            
        # State transitions
        if context.at_stairs:
            self.current_state = GameState.AT_STAIRS
        elif context.monster_adjacent:
            self.current_state = GameState.COMBAT
        elif context.door_direction and not context.stairs_pos:
            self.current_state = GameState.DOOR_BLOCKED
        elif context.stairs_pos:
            self.current_state = GameState.NAVIGATING
        elif self.stuck_counter > 15:
            self.current_state = GameState.STUCK
            self.stuck_counter = 0
        else:
            self.current_state = GameState.EXPLORING
            
    def _get_state_based_action(self, context: GameContext) -> str:
        """Get action based on current state"""
        
        # First check if we're executing a sequence
        if self.current_sequence:
            action = self._execute_sequence()
            if action:
                return action
                
        # Handle each state
        if self.current_state == GameState.AT_STAIRS:
            # Check if we're at down stairs (>) or up stairs (<)
            if context.stairs_pos:
                # Look at the actual stair character
                text = self.last_obs.get("text", {}) if hasattr(self, 'last_obs') else {}
                map_text = text.get("long_term_context", "")
                lines = map_text.split('\n')
                if context.stairs_pos[0] < len(lines):
                    line = lines[context.stairs_pos[0]]
                    if context.stairs_pos[1] < len(line):
                        stair_char = line[context.stairs_pos[1]]
                        if stair_char == '<':
                            return "up"  # Up stairs
            return "down"  # Default to down stairs
            
        elif self.current_state == GameState.COMBAT:
            return self._handle_combat(context)
            
        elif self.current_state == GameState.DOOR_BLOCKED:
            return self._handle_door(context)
            
        elif self.current_state == GameState.NAVIGATING:
            return self._navigate_to_stairs(context)
            
        elif self.current_state == GameState.STUCK:
            return self._get_unstuck(context)
            
        else:  # EXPLORING
            return self._explore_efficiently(context)
            
    def _handle_combat(self, context: GameContext) -> str:
        """Simple combat handling"""
        # In corridor: wait for monster to approach
        if len(context.safe_directions) <= 2:
            return "wait"
        # In room: kick if adjacent
        else:
            return "kick"
            
    def _handle_door(self, context: GameContext) -> str:
        """Handle door blocking path"""
        if context.door_direction:
            # Start door sequence
            self.current_sequence = self.action_sequences["door_sequence"]
            self.sequence_index = 0
            return self._execute_sequence()
        return "kick"
        
    def _navigate_to_stairs(self, context: GameContext) -> str:
        """Simple pathfinding to stairs"""
        if not context.player_pos or not context.stairs_pos:
            return "search"
            
        py, px = context.player_pos
        sy, sx = context.stairs_pos
        
        # Simple Manhattan distance navigation
        if abs(py - sy) > abs(px - sx):
            # Move vertically first
            if py > sy and "north" in context.safe_directions:
                return "north"
            elif py < sy and "south" in context.safe_directions:
                return "south"
        else:
            # Move horizontally
            if px > sx and "west" in context.safe_directions:
                return "west"
            elif px < sx and "east" in context.safe_directions:
                return "east"
                
        # If direct path blocked, try diagonal
        if py > sy and px > sx and "northwest" in context.safe_directions:
            return "northwest"
        elif py > sy and px < sx and "northeast" in context.safe_directions:
            return "northeast"
        elif py < sy and px > sx and "southwest" in context.safe_directions:
            return "southwest"
        elif py < sy and px < sx and "southeast" in context.safe_directions:
            return "southeast"
            
        # Fallback
        return context.safe_directions[0] if context.safe_directions else "search"
        
    def _get_unstuck(self, context: GameContext) -> str:
        """Try to get unstuck"""
        # Use LLM only for truly ambiguous cases
        if len(context.safe_directions) > 3:
            # Too many options, ask LLM for help
            return self._ask_llm_binary_choice(context)
        else:
            # Start escape sequence
            self.current_sequence = self.action_sequences["escape_pattern"]
            self.sequence_index = 0
            return self._execute_sequence()
            
    def _explore_efficiently(self, context: GameContext) -> str:
        """Efficient exploration pattern"""
        # If we haven't moved much, start spiral exploration
        if len(set(self.position_history)) < 3:
            self.current_sequence = self.action_sequences["spiral_explore"]
            self.sequence_index = 0
            return self._execute_sequence()
            
        # Otherwise, prefer unexplored directions
        if context.safe_directions:
            # Avoid repeating recent actions
            for direction in context.safe_directions:
                if direction not in list(self.action_history)[-3:]:
                    return direction
                    
        return "search"
        
    def _ask_llm_binary_choice(self, context: GameContext) -> str:
        """Use LLM only for simple binary choices"""
        # Create ultra-simple prompt
        if len(context.safe_directions) == 2:
            prompt = f"Choose: A) {context.safe_directions[0]} or B) {context.safe_directions[1]}?"
            prompt += "\nReply with just A or B:"
        else:
            # For exploration, ask about general direction
            prompt = "Explore: A) left side or B) right side?"
            prompt += "\nReply with just A or B:"
            
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.generate(messages)
            answer = response.completion.strip().upper()
            
            if len(context.safe_directions) == 2:
                if answer == 'A':
                    return context.safe_directions[0]
                elif answer == 'B':
                    return context.safe_directions[1]
            else:
                # Map to actual directions
                if answer == 'A':
                    return "west" if "west" in context.safe_directions else "northwest"
                else:
                    return "east" if "east" in context.safe_directions else "northeast"
                    
        except Exception as e:
            print(f"LLM Error: {e}")
            
        # Fallback
        return context.safe_directions[0] if context.safe_directions else "search"
        
    def _execute_sequence(self) -> Optional[str]:
        """Execute current action sequence"""
        if not self.current_sequence or self.sequence_index >= len(self.current_sequence):
            self.current_sequence = None
            self.sequence_index = 0
            return None
            
        action = self.current_sequence[self.sequence_index]
        self.sequence_index += 1
        return action
        
    def _check_monster_adjacent(self, lines: List[str], player_pos: Optional[Tuple[int, int]]) -> bool:
        """Check if monster is adjacent (excluding pets)"""
        if not player_pos or not lines:
            return False
            
        py, px = player_pos
        
        # Check if any line mentions "tame" (indicates pet)
        map_text = '\n'.join(lines)
        if "tame" in map_text.lower():
            return False
            
        # Check 8 adjacent squares
        monster_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        monster_chars -= {'@', 'I', '-', '|', '+', '.', '#', ' ', '>', '<'}
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                    
                ny, nx = py + dy, px + dx
                if 0 <= ny < len(lines) and 0 <= nx < len(lines[ny]):
                    char = lines[ny][nx]
                    if char in monster_chars:
                        return True
                        
        return False
        
    def _find_safe_directions(self, lines: List[str], player_pos: Optional[Tuple[int, int]]) -> List[str]:
        """Find directions without walls"""
        if not player_pos or not lines:
            return ["north", "south", "east", "west"]
            
        py, px = player_pos
        safe = []
        
        # Check each direction
        checks = [
            ("north", -1, 0), ("south", 1, 0), ("east", 0, 1), ("west", 0, -1),
            ("northeast", -1, 1), ("northwest", -1, -1), ("southeast", 1, 1), ("southwest", 1, -1)
        ]
        
        for direction, dy, dx in checks:
            ny, nx = py + dy, px + dx
            if 0 <= ny < len(lines) and 0 <= nx < len(lines[ny]):
                char = lines[ny][nx]
                if char not in ['-', '|', ' ']:  # Not a wall or empty
                    safe.append(direction)
                    
        return safe
        
    def _find_door_direction(self, lines: List[str], player_pos: Optional[Tuple[int, int]]) -> Optional[str]:
        """Find direction to nearest door"""
        if not player_pos or not lines:
            return None
            
        py, px = player_pos
        
        # Check adjacent squares for doors
        checks = [
            ("north", -1, 0), ("south", 1, 0), ("east", 0, 1), ("west", 0, -1)
        ]
        
        for direction, dy, dx in checks:
            ny, nx = py + dy, px + dx
            if 0 <= ny < len(lines) and 0 <= nx < len(lines[ny]):
                if lines[ny][nx] == '+':
                    return direction
                    
        return None
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.current_state = GameState.EXPLORING
        self.action_history.clear()
        self.position_history.clear()
        self.stuck_counter = 0
        self.last_score = 0
        self.current_sequence = None
        self.sequence_index = 0


# ============================================================================
# LEARNING COMPONENT
# ============================================================================

class SequenceLearner:
    """Learn successful action sequences over episodes"""
    
    def __init__(self):
        self.episode_buffer = []
        self.learned_sequences = defaultdict(list)
        
    def record_episode(self, states: List[GameState], actions: List[str], 
                      final_score: int, final_depth: int):
        """Record an episode for learning"""
        if final_score > 0 or final_depth > 1:
            # Successful episode - extract patterns
            for i in range(len(states) - 3):
                if states[i] == states[i+1] == states[i+2]:
                    # Found stable state with actions
                    sequence = actions[i:i+3]
                    self.learned_sequences[states[i]].append(sequence)
                    
    def get_best_sequence(self, state: GameState) -> Optional[List[str]]:
        """Get best action sequence for a state"""
        if state in self.learned_sequences:
            # Return most common sequence
            sequences = self.learned_sequences[state]
            if sequences:
                return max(sequences, key=sequences.count)
        return None