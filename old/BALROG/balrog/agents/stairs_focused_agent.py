"""
Stairs-focused agent that prioritizes finding and using stairs
"""

import re
from typing import Dict, Any, List, Optional
from .base import BaseAgent
from balrog.client import LLMResponse


class StairsFocusedAgent(BaseAgent):
    """Agent focused on efficiently finding and using stairs"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # State tracking
        self.stairs_position = None
        self.player_position = None
        self.explored_positions = set()
        self.stuck_counter = 0
        self.last_positions = []
        self.stairs_found = False
        self.on_stairs = False
        self.attempted_down = False
        
    def reset(self):
        """Reset agent state"""
        self.stairs_position = None
        self.player_position = None
        self.explored_positions = set()
        self.stuck_counter = 0
        self.last_positions = []
        self.stairs_found = False
        self.on_stairs = False
        self.attempted_down = False
        
    def act(self, observation: Dict[str, Any], prev_action: str = None, return_reasoning: bool = False) -> LLMResponse:
        """Choose action based on current state"""
        # Parse current state
        state = self._parse_state(observation)
        self._update_positions(state)
        
        # Determine action
        action = self._choose_action(state, observation)
        
        # Create reasoning
        reasoning = self._get_reasoning(state, action)
        
        return LLMResponse(
            model_id="stairs_focused",
            completion=action,
            stop_reason="complete",
            input_tokens=50,
            output_tokens=10,
            reasoning=reasoning
        )
    
    def _parse_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Parse observation into structured state"""
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        lang_obs = text.get("language_observation", "")
        
        # Parse map to find positions
        map_section = long_text.split("map:")[-1].split("Agent")[0] if "map:" in long_text else ""
        
        # Check for stairs in observation
        stairs_visible = (
            "stairs" in lang_obs.lower() or 
            "staircase" in lang_obs.lower() or
            ">" in lang_obs or
            ">" in map_section
        )
        
        # Parse depth
        depth_match = re.search(r'Dlvl:(\d+)', long_text)
        depth = int(depth_match.group(1)) if depth_match else 1
        
        return {
            "map_section": map_section,
            "lang_obs": lang_obs,
            "long_text": long_text,
            "stairs_visible": stairs_visible,
            "depth": depth
        }
    
    def _update_positions(self, state: Dict[str, Any]):
        """Update player and stairs positions from map"""
        map_section = state["map_section"]
        lines = map_section.strip().split('\n')[:21]
        
        player_y, player_x = None, None
        stairs_y, stairs_x = None, None
        
        for y, line in enumerate(lines):
            if '@' in line:
                player_x = line.index('@')
                player_y = y
            if '>' in line:
                stairs_x = line.index('>')
                stairs_y = y
                
        if player_x is not None:
            self.player_position = (player_x, player_y)
            self.explored_positions.add(self.player_position)
            self.last_positions.append(self.player_position)
            if len(self.last_positions) > 5:
                self.last_positions.pop(0)
                
        if stairs_x is not None:
            self.stairs_position = (stairs_x, stairs_y)
            if not self.stairs_found:
                self.stairs_found = True
                
        # Check if on stairs
        if self.player_position and self.stairs_position:
            self.on_stairs = (self.player_position == self.stairs_position)
    
    def _choose_action(self, state: Dict[str, Any], obs: Dict[str, Any]) -> str:
        """Choose the best action based on current state"""
        
        # Priority 1: If on stairs, go down
        if self.on_stairs and not self.attempted_down:
            self.attempted_down = True
            return ">"
            
        # Priority 2: If stairs visible but not on them, navigate to stairs
        if self.stairs_found and self.stairs_position and not self.on_stairs:
            return self._navigate_to_stairs()
            
        # Priority 3: Explore systematically
        return self._explore_systematically(state)
    
    def _navigate_to_stairs(self) -> str:
        """Navigate to stairs position"""
        if not self.player_position or not self.stairs_position:
            return "wait"
            
        px, py = self.player_position
        sx, sy = self.stairs_position
        
        dx = sx - px
        dy = sy - py
        
        # Prefer cardinal directions first
        if abs(dx) > abs(dy):
            if dx > 0:
                return "east"
            else:
                return "west"
        elif dy != 0:
            if dy > 0:
                return "south"
            else:
                return "north"
        else:
            # We should be at stairs, try going down
            return ">"
    
    def _explore_systematically(self, state: Dict[str, Any]) -> str:
        """Systematic exploration to find stairs"""
        # Check if stuck
        if len(self.last_positions) >= 3:
            recent_positions = self.last_positions[-3:]
            if len(set(recent_positions)) == 1:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                
        # If stuck, try different strategies
        if self.stuck_counter > 2:
            # Try diagonal movements
            diagonals = ["northeast", "northwest", "southeast", "southwest"]
            for diag in diagonals:
                test_pos = self._get_new_position(self.player_position, diag)
                if test_pos not in self.explored_positions:
                    self.stuck_counter = 0
                    return diag
                    
            # Try kicking if really stuck
            if self.stuck_counter > 5:
                self.stuck_counter = 0
                return "kick"
                
        # Search in rooms
        if "room" in state["lang_obs"].lower():
            # In a room, search thoroughly
            if len(self.last_positions) < 3 or self.last_positions[-1] == self.last_positions[-3]:
                return "search"
                
        # Systematic exploration pattern
        # Try to explore unexplored directions
        directions = ["south", "east", "north", "west"]
        for direction in directions:
            new_pos = self._get_new_position(self.player_position, direction)
            if new_pos not in self.explored_positions:
                return direction
                
        # If all cardinal directions explored, try diagonals
        diagonals = ["southeast", "northeast", "southwest", "northwest"]
        for diagonal in diagonals:
            new_pos = self._get_new_position(self.player_position, diagonal)
            if new_pos not in self.explored_positions:
                return diagonal
                
        # Default to searching
        return "search"
    
    def _get_new_position(self, pos: tuple, direction: str) -> tuple:
        """Get new position after moving in direction"""
        if not pos:
            return (0, 0)
            
        x, y = pos
        moves = {
            "north": (0, -1),
            "south": (0, 1),
            "east": (1, 0),
            "west": (-1, 0),
            "northeast": (1, -1),
            "northwest": (-1, -1),
            "southeast": (1, 1),
            "southwest": (-1, 1)
        }
        
        if direction in moves:
            dx, dy = moves[direction]
            return (x + dx, y + dy)
        return pos
    
    def _get_reasoning(self, state: Dict[str, Any], action: str) -> str:
        """Generate reasoning for the action"""
        if self.on_stairs and action == ">":
            return "On stairs, going down"
        elif self.stairs_found and action in ["north", "south", "east", "west"]:
            return f"Moving {action} toward stairs at {self.stairs_position}"
        elif action == "search":
            return "Searching room for stairs"
        elif action in ["northeast", "northwest", "southeast", "southwest"]:
            return f"Exploring diagonally {action}"
        elif action == "kick":
            return "Kicking to break through obstacle"
        else:
            return f"Exploring {action}"