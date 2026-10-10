"""
Simple Heuristic Agent - Minimal LLM usage with robust heuristics
================================================================

This agent uses simple, robust heuristics and only calls the LLM
when absolutely necessary. The key is to handle common cases well.
"""

import random
from typing import Dict, List, Optional, Tuple
from collections import deque

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse


class SimpleHeuristicAgent(BaseAgent):
    """Agent that relies primarily on heuristics, minimal LLM"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Memory
        self.visited_positions = set()
        self.action_history = deque(maxlen=30)
        self.stuck_counter = 0
        self.exploration_mode = "systematic"
        
        # Progress tracking
        self.last_score = 0
        self.last_depth = 0
        self.turn_count = 0
        
        # Saved successful paths
        self.successful_path = None
        
    def act(self, obs, prev_action=None):
        """Select action using heuristics"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        self.turn_count += 1
        
        # Parse observation
        game_map = self._parse_map(obs)
        player_pos = self._find_player(game_map)
        stairs_info = self._find_stairs(game_map)
        
        # Track position
        if player_pos:
            self.visited_positions.add(player_pos)
            
        # Check progress
        score = obs.get("score", 0)
        depth = obs.get("depth", 0)
        
        made_progress = score > self.last_score or depth > self.last_depth
        if made_progress:
            self.stuck_counter = 0
            self.last_score = score
            self.last_depth = depth
        else:
            self.stuck_counter += 1
            
        # Decision logic
        action = self._decide_action(game_map, player_pos, stairs_info, made_progress)
        
        # Record action
        self.action_history.append(action)
        
        return LLMResponse(
            model_id="simple_heuristic",
            completion=action,
            stop_reason="complete",
            input_tokens=0,
            output_tokens=0,
            reasoning=self._get_reasoning(stairs_info)
        )
        
    def _parse_map(self, obs):
        """Parse the game map into a 2D array"""
        text = obs.get("text", {})
        map_text = text.get("long_term_context", "")
        lines = map_text.split('\n')
        
        # Find where the map starts (after "map:" line)
        map_start = -1
        for i, line in enumerate(lines):
            if line.strip() == "map:":
                map_start = i + 1
                break
                
        if map_start == -1:
            # Fallback: try to find map lines by content
            game_map = []
            for line in lines:
                # Keep lines that look like map content
                if any(c in line for c in ['|', '-', '.', '#', '@', '<', '>', '+', 'd']):
                    game_map.append(line)
            return game_map
            
        # Extract map lines (usually 21 lines after "map:")
        game_map = []
        for i in range(map_start, min(map_start + 21, len(lines))):
            if i < len(lines):
                # Stop at status line (starts with player name)
                if lines[i].strip().startswith('Agent'):
                    break
                game_map.append(lines[i])
                
        return game_map
        
    def _find_player(self, game_map):
        """Find player position (row, col)"""
        for row, line in enumerate(game_map):
            if '@' in line:
                col = line.index('@')
                return (row, col)
        return None
        
    def _find_stairs(self, game_map):
        """Find stairs and their type"""
        for row, line in enumerate(game_map):
            if '>' in line:
                col = line.index('>')
                return {'pos': (row, col), 'type': 'down'}
            elif '<' in line:
                col = line.index('<')
                return {'pos': (row, col), 'type': 'up'}
        return None
        
    def _decide_action(self, game_map, player_pos, stairs_info, made_progress):
        """Main decision logic"""
        
        # Case 1: At stairs - use the stair character to go up/down
        if stairs_info and player_pos and player_pos == stairs_info['pos']:
            # In NetHack, use '<' to go up and '>' to go down when on stairs
            return '<' if stairs_info['type'] == 'up' else '>'
            
        # Case 2: Stairs visible - navigate to them
        if stairs_info and player_pos:
            return self._navigate_to(player_pos, stairs_info['pos'], game_map)
            
        # Case 3: Door adjacent - handle it
        door_dir = self._check_adjacent_door(game_map, player_pos)
        if door_dir:
            # Alternate between kick and open
            if self.action_history and self.action_history[-1] == "kick":
                return "open"
            return "kick"
            
        # Case 4: Stuck - change strategy
        if self.stuck_counter > 30:
            self.stuck_counter = 0
            return self._change_exploration_strategy(game_map, player_pos)
            
        # Case 5: Normal exploration
        return self._explore(game_map, player_pos)
        
    def _adjacent(self, pos1, pos2):
        """Check if positions are adjacent or the same"""
        return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1
        
    def _navigate_to(self, from_pos, to_pos, game_map):
        """Navigate from one position to another"""
        d_row = to_pos[0] - from_pos[0]
        d_col = to_pos[1] - from_pos[1]
        
        # Try to move in the most needed direction
        candidates = []
        
        if d_row < 0:
            candidates.append(("north", from_pos[0] - 1, from_pos[1]))
        elif d_row > 0:
            candidates.append(("south", from_pos[0] + 1, from_pos[1]))
            
        if d_col < 0:
            candidates.append(("west", from_pos[0], from_pos[1] - 1))
        elif d_col > 0:
            candidates.append(("east", from_pos[0], from_pos[1] + 1))
            
        # Check which moves are valid
        for direction, new_row, new_col in candidates:
            if self._is_walkable(game_map, new_row, new_col):
                return direction
                
        # Try diagonals if cardinal directions blocked
        if d_row < 0 and d_col < 0 and self._is_walkable(game_map, from_pos[0] - 1, from_pos[1] - 1):
            return "northwest"
        elif d_row < 0 and d_col > 0 and self._is_walkable(game_map, from_pos[0] - 1, from_pos[1] + 1):
            return "northeast"
        elif d_row > 0 and d_col < 0 and self._is_walkable(game_map, from_pos[0] + 1, from_pos[1] - 1):
            return "southwest"
        elif d_row > 0 and d_col > 0 and self._is_walkable(game_map, from_pos[0] + 1, from_pos[1] + 1):
            return "southeast"
            
        # If all else fails, try any valid move
        return self._any_valid_move(game_map, from_pos)
        
    def _is_walkable(self, game_map, row, col):
        """Check if a position is walkable"""
        if 0 <= row < len(game_map) and 0 <= col < len(game_map[row]):
            char = game_map[row][col]
            # Walkable: floor, corridor, stairs, door (closed), spaces in unexplored areas
            return char in '.#<>+ '
        return False
        
    def _check_adjacent_door(self, game_map, player_pos):
        """Check for adjacent doors"""
        if not player_pos:
            return None
            
        row, col = player_pos
        directions = [
            ("north", row-1, col),
            ("south", row+1, col),
            ("east", row, col+1),
            ("west", row, col-1)
        ]
        
        for direction, new_row, new_col in directions:
            if 0 <= new_row < len(game_map) and 0 <= new_col < len(game_map[new_row]):
                if game_map[new_row][new_col] == '+':
                    return direction
                    
        return None
        
    def _explore(self, game_map, player_pos):
        """Exploration strategy"""
        if not player_pos:
            return "search"
            
        if self.exploration_mode == "systematic":
            # Systematic exploration - prefer unvisited areas
            valid_moves = self._get_valid_moves(game_map, player_pos)
            
            # Prefer unvisited positions
            unvisited = []
            for direction, new_pos in valid_moves:
                if new_pos not in self.visited_positions:
                    unvisited.append(direction)
                    
            if unvisited:
                return random.choice(unvisited)
                
            # Otherwise, move away from recent positions
            if valid_moves:
                # Avoid repeating recent actions
                non_recent = []
                recent_actions = list(self.action_history)[-5:] if len(self.action_history) >= 5 else []
                
                for direction, _ in valid_moves:
                    if direction not in recent_actions:
                        non_recent.append(direction)
                        
                if non_recent:
                    return random.choice(non_recent)
                    
                return valid_moves[0][0]
                
        # Search if no moves available
        return "search"
        
    def _get_valid_moves(self, game_map, player_pos):
        """Get all valid moves from current position"""
        if not player_pos:
            return []
            
        row, col = player_pos
        moves = []
        
        directions = [
            ("north", row-1, col),
            ("south", row+1, col),
            ("east", row, col+1),
            ("west", row, col-1),
            ("northeast", row-1, col+1),
            ("northwest", row-1, col-1),
            ("southeast", row+1, col+1),
            ("southwest", row+1, col-1)
        ]
        
        for direction, new_row, new_col in directions:
            if self._is_walkable(game_map, new_row, new_col):
                moves.append((direction, (new_row, new_col)))
                
        return moves
        
    def _any_valid_move(self, game_map, player_pos):
        """Return any valid move"""
        valid_moves = self._get_valid_moves(game_map, player_pos)
        if valid_moves:
            return valid_moves[0][0]
        return "search"
        
    def _change_exploration_strategy(self, game_map, player_pos):
        """Change strategy when stuck - minimal LLM usage"""
        # Only use LLM if we have multiple equally good options
        valid_moves = self._get_valid_moves(game_map, player_pos)
        
        if len(valid_moves) >= 3:
            # Too many options - use simple LLM query
            # Pick two most different directions
            if any(d[0] in ["north", "south"] for d in valid_moves) and \
               any(d[0] in ["east", "west"] for d in valid_moves):
                
                # Ask LLM: vertical or horizontal?
                prompt = "vertical or horizontal? Reply with one word:"
                messages = [{"role": "user", "content": prompt}]
                
                try:
                    response = self.client.generate(messages)
                    answer = response.completion.strip().lower()
                    
                    if "vert" in answer:
                        # Prefer north/south
                        for direction, _ in valid_moves:
                            if direction in ["north", "south"]:
                                return direction
                    else:
                        # Prefer east/west
                        for direction, _ in valid_moves:
                            if direction in ["east", "west"]:
                                return direction
                except:
                    pass
                    
        # Fallback: pick a random valid move
        if valid_moves:
            return random.choice(valid_moves)[0]
            
        return "search"
        
    def _get_reasoning(self, stairs_info):
        """Simple reasoning string"""
        if stairs_info:
            return f"Stairs {stairs_info['type']} at {stairs_info['pos']}"
        elif self.stuck_counter > 20:
            return "Stuck - exploring"
        else:
            return f"Exploring - visited {len(self.visited_positions)} positions"
            
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.visited_positions.clear()
        self.action_history.clear()
        self.stuck_counter = 0
        self.exploration_mode = "systematic"
        self.last_score = 0
        self.last_depth = 0
        self.turn_count = 0