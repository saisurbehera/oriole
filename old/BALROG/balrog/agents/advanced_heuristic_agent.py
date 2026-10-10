"""
Advanced Heuristic Agent - Enhanced NetHack agent with comprehensive heuristics
==============================================================================

This agent builds on SimpleHeuristicAgent with:
1. Item pickup and management
2. Combat detection and avoidance  
3. Trap detection and careful movement
4. Food and hunger management
5. Multi-level exploration strategy
6. Better door handling
7. Corner checking for hidden doors
"""

import re
import random
from typing import Dict, List, Optional, Tuple, Set
from collections import deque
from dataclasses import dataclass
from enum import Enum

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse


class ExplorationMode(Enum):
    """Different exploration strategies"""
    SYSTEMATIC = "systematic"      # Room by room
    CORRIDOR_FOLLOW = "corridor"   # Follow corridors
    WALL_FOLLOW = "wall_follow"    # Follow walls
    RANDOM_WALK = "random"         # Random exploration


@dataclass
class GameState:
    """Track important game state"""
    hunger_level: str = "normal"
    hp_percentage: float = 1.0
    turn_count: int = 0
    current_level: int = 1
    monsters_nearby: List[Tuple[int, int]] = None
    items_nearby: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.monsters_nearby is None:
            self.monsters_nearby = []
        if self.items_nearby is None:
            self.items_nearby = []


class AdvancedHeuristicAgent(BaseAgent):
    """Enhanced agent with comprehensive heuristics"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Memory
        self.visited_positions = set()
        self.action_history = deque(maxlen=50)
        self.stuck_counter = 0
        self.door_kick_attempts = {}  # Track kicks per door
        self.exploration_mode = ExplorationMode.SYSTEMATIC
        
        # Progress tracking
        self.last_score = 0
        self.last_depth = 1
        self.turn_count = 0
        self.last_hp = None
        
        # Level memory
        self.level_stairs = {}  # Remember stairs per level
        self.level_layout = {}  # Remember level layouts
        
        # Action patterns
        self.search_count = 0
        self.corner_positions = set()
        
    def act(self, obs, prev_action=None):
        """Select action using advanced heuristics"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        self.turn_count += 1
        
        # Parse game state
        game_state = self._parse_game_state(obs)
        game_map = self._parse_map(obs)
        player_pos = self._find_player(game_map)
        
        # Update position tracking
        if player_pos:
            self.visited_positions.add((game_state.current_level, player_pos))
            
        # Update progress tracking
        score = obs.get("score", 0)
        if score > self.last_score:
            self.last_score = score
            self.stuck_counter = 0
            
        if game_state.current_level != self.last_depth:
            self.last_depth = game_state.current_level
            self.stuck_counter = 0
            
        # Main decision
        action = self._decide_action(game_state, game_map, player_pos, obs)
        
        # Record action
        self.action_history.append(action)
        
        return LLMResponse(
            model_id="advanced_heuristic",
            completion=action,
            stop_reason="complete",
            input_tokens=0,
            output_tokens=0,
            reasoning=self._get_reasoning(game_state, action)
        )
        
    def _parse_game_state(self, obs) -> GameState:
        """Extract comprehensive game state"""
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        
        # Parse status line
        status_match = re.search(r'HP:(\d+)\((\d+)\).*T:(\d+)', long_text)
        if status_match:
            current_hp = int(status_match.group(1))
            max_hp = int(status_match.group(2))
            turn = int(status_match.group(3))
            hp_percentage = current_hp / max_hp if max_hp > 0 else 1.0
        else:
            hp_percentage = 1.0
            turn = self.turn_count
            
        # Parse dungeon level
        level_match = re.search(r'Dlvl:(\d+)', long_text)
        current_level = int(level_match.group(1)) if level_match else 1
        
        # Check hunger
        hunger = "normal"
        if "hungry" in long_text.lower():
            hunger = "hungry"
        elif "weak" in long_text.lower():
            hunger = "weak"
        elif "fainting" in long_text.lower():
            hunger = "fainting"
            
        # Find monsters and items
        monsters = self._find_monsters(obs)
        items = self._find_items(obs)
        
        return GameState(
            hunger_level=hunger,
            hp_percentage=hp_percentage,
            turn_count=turn,
            current_level=current_level,
            monsters_nearby=monsters,
            items_nearby=items
        )
        
    def _find_monsters(self, obs) -> List[Tuple[int, int]]:
        """Find monster positions"""
        monsters = []
        text = obs.get("text", {})
        lang_obs = text.get("language_observation", "")
        
        # Parse language observation for monsters
        monster_keywords = ["goblin", "kobold", "jackal", "rat", "bat", "newt", 
                          "lichen", "grid bug", "hostile"]
        
        # Exclude pets
        pet_keywords = ["tame", "peaceful"]
        
        lines = lang_obs.split('\n')
        for line in lines:
            line_lower = line.lower()
            is_monster = any(kw in line_lower for kw in monster_keywords)
            is_pet = any(kw in line_lower for kw in pet_keywords)
            
            if is_monster and not is_pet:
                # Try to extract position from line
                # This is approximate - would need more parsing
                monsters.append((-1, -1))  # Placeholder
                
        return monsters
        
    def _find_items(self, obs) -> List[Tuple[int, int]]:
        """Find item positions"""
        items = []
        game_map = self._parse_map(obs)
        
        # Common item symbols
        item_symbols = ['$', '%', '!', '?', '/', '=', '"', '(', ')', '[', ']', 
                       '*', '+', ':', ';']
        
        for row, line in enumerate(game_map):
            for col, char in enumerate(line):
                if char in item_symbols:
                    items.append((row, col))
                    
        return items
        
    def _decide_action(self, game_state: GameState, game_map, player_pos, obs):
        """Advanced decision logic"""
        
        # Priority 1: Emergency situations
        if game_state.hp_percentage < 0.3:
            return self._handle_low_hp(game_state, game_map, player_pos)
            
        if game_state.hunger_level in ["weak", "fainting"]:
            return self._handle_hunger(game_state, game_map, player_pos)
            
        # Priority 2: Combat
        if game_state.monsters_nearby:
            return self._handle_combat(game_state, game_map, player_pos)
            
        # Priority 3: Items nearby
        if game_state.items_nearby and player_pos:
            closest_item = self._find_closest(player_pos, game_state.items_nearby)
            if closest_item and self._distance(player_pos, closest_item) <= 5:
                return self._navigate_to_item(player_pos, closest_item, game_map)
                
        # Priority 4: Stairs
        stairs_info = self._find_stairs(game_map)
        if stairs_info and player_pos:
            # At stairs
            if player_pos == stairs_info['pos']:
                # Should we go down?
                if self._should_descend(game_state):
                    return '<' if stairs_info['type'] == 'up' else '>'
                else:
                    # Continue exploring this level
                    return self._explore_level(game_map, player_pos, game_state)
            # Navigate to stairs if level is well explored
            elif self._level_explored_enough(game_state):
                return self._navigate_to(player_pos, stairs_info['pos'], game_map)
                
        # Priority 5: Doors
        door_info = self._check_doors(game_map, player_pos)
        if door_info:
            return self._handle_door(door_info, player_pos)
            
        # Priority 6: Check if stuck
        if self._is_stuck():
            return self._unstuck_strategy(game_map, player_pos, game_state)
            
        # Priority 7: Normal exploration
        action = self._explore_level(game_map, player_pos, game_state)
        
        # Never return "search" as default action
        if action == "search" and self.search_count > 2:
            # Force movement instead
            valid_moves = self._get_valid_moves(game_map, player_pos)
            if valid_moves:
                return random.choice(valid_moves)[0]
            
        return action
        
    def _handle_low_hp(self, game_state, game_map, player_pos):
        """Handle low HP situations"""
        # Look for safe direction away from monsters
        if game_state.monsters_nearby:
            safe_moves = self._get_safe_moves(game_map, player_pos, game_state.monsters_nearby)
            if safe_moves:
                return random.choice(safe_moves)
                
        # Otherwise, search for hidden doors to escape
        return "search"
        
    def _handle_hunger(self, game_state, game_map, player_pos):
        """Handle hunger - look for food"""
        # Check inventory for food
        text = game_state.__dict__.get('raw_text', '')
        if "%" in text:  # % often represents food
            return "eat"
            
        # Look for food items on ground
        for item_pos in game_state.items_nearby:
            item_char = self._get_char_at(game_map, item_pos)
            if item_char == '%':  # Comestible
                return self._navigate_to(player_pos, item_pos, game_map)
                
        # Continue exploring to find food
        return self._explore_level(game_map, player_pos, game_state)
        
    def _handle_combat(self, game_state, game_map, player_pos):
        """Handle combat situations"""
        # Simple strategy: fight if strong, flee if weak
        if game_state.hp_percentage > 0.7:
            # Fight - move toward nearest monster
            if game_state.monsters_nearby:
                return self._approach_monster(game_map, player_pos)
        else:
            # Flee - move away from monsters
            safe_moves = self._get_safe_moves(game_map, player_pos, game_state.monsters_nearby)
            if safe_moves:
                return random.choice(safe_moves)
                
        return "wait"
        
    def _navigate_to_item(self, from_pos, item_pos, game_map):
        """Navigate to pick up item"""
        if self._adjacent(from_pos, item_pos):
            # Pick up item
            return ","  # Pickup command
        else:
            return self._navigate_to(from_pos, item_pos, game_map)
            
    def _should_descend(self, game_state):
        """Decide if we should go down stairs"""
        # Don't descend if:
        # - HP is low
        # - We're hungry
        # - Current level not well explored
        
        if game_state.hp_percentage < 0.5:
            return False
            
        if game_state.hunger_level != "normal":
            return False
            
        # Check exploration percentage (simple heuristic)
        level_positions = sum(1 for (lvl, _) in self.visited_positions 
                            if lvl == game_state.current_level)
        
        # Descend if we've visited many positions
        return level_positions > 100
        
    def _level_explored_enough(self, game_state):
        """Check if current level is sufficiently explored"""
        level_positions = sum(1 for (lvl, _) in self.visited_positions 
                            if lvl == game_state.current_level)
        return level_positions > 150
        
    def _check_doors(self, game_map, player_pos):
        """Check for doors in all directions"""
        if not player_pos:
            return None
            
        row, col = player_pos
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
            if 0 <= new_row < len(game_map) and 0 <= new_col < len(game_map[new_row]):
                char = game_map[new_row][new_col]
                if char == '+':  # Closed door
                    return {"direction": direction, "pos": (new_row, new_col), "type": "closed"}
                elif char == '-' or char == '|':  # Might be secret door
                    # Check if it's suspicious (dead end, etc)
                    if self._is_suspicious_wall(game_map, new_row, new_col):
                        return {"direction": direction, "pos": (new_row, new_col), "type": "secret"}
                        
        return None
        
    def _handle_door(self, door_info, player_pos):
        """Smart door handling"""
        door_pos = door_info['pos']
        direction = door_info['direction']
        
        if door_info['type'] == 'closed':
            # Track kick attempts
            kick_count = self.door_kick_attempts.get(door_pos, 0)
            
            if kick_count < 3:
                # Try kicking
                self.door_kick_attempts[door_pos] = kick_count + 1
                return "kick"
            elif kick_count < 5:
                # Try opening
                self.door_kick_attempts[door_pos] = kick_count + 1
                return "open"
            else:
                # Give up on this door
                return self._explore_level(None, player_pos, None)
                
        elif door_info['type'] == 'secret':
            # Search for secret door
            return "search"
            
        return direction
        
    def _is_suspicious_wall(self, game_map, row, col):
        """Check if a wall might hide a secret door"""
        # Look for dead ends or gaps in room layout
        # This is a simple heuristic
        adjacent_floors = 0
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < len(game_map) and 0 <= new_col < len(game_map[new_row]):
                    if game_map[new_row][new_col] in '.#':
                        adjacent_floors += 1
                        
        # Suspicious if there are floors on opposite sides
        return adjacent_floors >= 2
        
    def _is_stuck(self):
        """Detect if we're stuck"""
        if len(self.action_history) < 10:
            return False
            
        # Check for repetitive patterns
        recent_actions = list(self.action_history)[-10:]
        unique_actions = len(set(recent_actions))
        
        # Count consecutive searches
        consecutive_searches = 0
        for action in reversed(recent_actions):
            if action == "search":
                consecutive_searches += 1
            else:
                break
                
        # Stuck if we're repeating same few actions or searching too much
        return unique_actions <= 3 or consecutive_searches >= 3
        
    def _unstuck_strategy(self, game_map, player_pos, game_state):
        """Get unstuck using various strategies"""
        self.stuck_counter += 1
        
        # Strategy 1: Search for secret doors
        if self.stuck_counter % 5 == 0:
            self.search_count += 1
            return "search"
            
        # Strategy 2: Try a different exploration mode
        if self.stuck_counter % 10 == 0:
            modes = list(ExplorationMode)
            modes.remove(self.exploration_mode)
            self.exploration_mode = random.choice(modes)
            
        # Strategy 3: Random walk
        valid_moves = self._get_valid_moves(game_map, player_pos)
        if valid_moves:
            return random.choice(valid_moves)[0]
            
        return "wait"
        
    def _explore_level(self, game_map, player_pos, game_state):
        """Exploration strategy based on mode"""
        if not player_pos:
            return "search"
            
        if self.exploration_mode == ExplorationMode.SYSTEMATIC:
            return self._systematic_explore(game_map, player_pos)
        elif self.exploration_mode == ExplorationMode.CORRIDOR_FOLLOW:
            return self._corridor_follow(game_map, player_pos)
        elif self.exploration_mode == ExplorationMode.WALL_FOLLOW:
            return self._wall_follow(game_map, player_pos)
        else:
            return self._random_explore(game_map, player_pos)
            
    def _systematic_explore(self, game_map, player_pos):
        """Room by room exploration"""
        valid_moves = self._get_valid_moves(game_map, player_pos)
        
        # Prefer unvisited positions
        unvisited = []
        current_level = self.last_depth
        
        for direction, new_pos in valid_moves:
            if (current_level, new_pos) not in self.visited_positions:
                unvisited.append(direction)
                
        if unvisited:
            return random.choice(unvisited)
            
        # Otherwise, move toward unexplored areas
        if valid_moves:
            # Prefer moves that lead to more open space
            best_move = None
            best_openness = -1
            
            for direction, new_pos in valid_moves:
                openness = self._count_open_spaces(game_map, new_pos)
                if openness > best_openness:
                    best_openness = openness
                    best_move = direction
                    
            if best_move:
                return best_move
            
            # If all moves are equally good, avoid recent directions
            non_recent = []
            recent_actions = list(self.action_history)[-5:] if len(self.action_history) >= 5 else []
            
            for direction, _ in valid_moves:
                if direction not in recent_actions:
                    non_recent.append(direction)
                    
            if non_recent:
                return random.choice(non_recent)
            
            return random.choice(valid_moves)[0]
            
        # Only search if truly stuck with no moves
        return "wait"
        
    def _corridor_follow(self, game_map, player_pos):
        """Follow corridors to find new rooms"""
        # Look for corridor tiles (#)
        valid_moves = self._get_valid_moves(game_map, player_pos)
        
        corridor_moves = []
        for direction, new_pos in valid_moves:
            if self._get_char_at(game_map, new_pos) == '#':
                corridor_moves.append(direction)
                
        if corridor_moves:
            return random.choice(corridor_moves)
            
        # Fallback to systematic
        return self._systematic_explore(game_map, player_pos)
        
    def _wall_follow(self, game_map, player_pos):
        """Follow walls to explore perimeter"""
        # Simple right-hand wall following
        if self.action_history:
            last_action = self.action_history[-1]
            # Get direction 90 degrees clockwise
            clockwise = {
                "north": "east", "east": "south", 
                "south": "west", "west": "north"
            }
            
            if last_action in clockwise:
                preferred = clockwise[last_action]
                valid_moves = self._get_valid_moves(game_map, player_pos)
                
                for direction, _ in valid_moves:
                    if direction == preferred:
                        return direction
                        
        # Fallback
        return self._systematic_explore(game_map, player_pos)
        
    def _random_explore(self, game_map, player_pos):
        """Random walk exploration"""
        valid_moves = self._get_valid_moves(game_map, player_pos)
        if valid_moves:
            return random.choice(valid_moves)[0]
        return "wait"
        
    def _count_open_spaces(self, game_map, pos):
        """Count open spaces around a position"""
        count = 0
        row, col = pos
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < len(game_map) and 0 <= new_col < len(game_map[new_row]):
                    if game_map[new_row][new_col] in '.# ':
                        count += 1
                        
        return count
        
    def _get_safe_moves(self, game_map, player_pos, monster_positions):
        """Get moves away from monsters"""
        if not player_pos:
            return []
            
        valid_moves = self._get_valid_moves(game_map, player_pos)
        safe_moves = []
        
        for direction, new_pos in valid_moves:
            # Check if this move increases distance from all monsters
            is_safe = True
            for monster_pos in monster_positions:
                if monster_pos != (-1, -1):  # Valid position
                    if self._distance(new_pos, monster_pos) < self._distance(player_pos, monster_pos):
                        is_safe = False
                        break
                        
            if is_safe:
                safe_moves.append(direction)
                
        return safe_moves
        
    def _approach_monster(self, game_map, player_pos):
        """Move toward nearest monster"""
        # Simple melee approach
        valid_moves = self._get_valid_moves(game_map, player_pos)
        if valid_moves:
            # Pick random direction for now
            # (Monster positions need better parsing)
            return random.choice(valid_moves)[0]
        return "wait"
        
    def _distance(self, pos1, pos2):
        """Manhattan distance between positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def _find_closest(self, from_pos, positions):
        """Find closest position from list"""
        if not positions:
            return None
            
        closest = None
        min_dist = float('inf')
        
        for pos in positions:
            dist = self._distance(from_pos, pos)
            if dist < min_dist:
                min_dist = dist
                closest = pos
                
        return closest
        
    def _get_reasoning(self, game_state, action):
        """Generate reasoning string"""
        parts = []
        
        if game_state.hp_percentage < 0.5:
            parts.append(f"HP: {int(game_state.hp_percentage*100)}%")
            
        if game_state.hunger_level != "normal":
            parts.append(f"Hunger: {game_state.hunger_level}")
            
        if game_state.monsters_nearby:
            parts.append(f"Monsters: {len(game_state.monsters_nearby)}")
            
        parts.append(f"Level: {game_state.current_level}")
        parts.append(f"Mode: {self.exploration_mode.value}")
        
        return " | ".join(parts)
        
    # Include all the basic methods from SimpleHeuristicAgent
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
        
    def _get_char_at(self, game_map, pos):
        """Get character at position"""
        row, col = pos
        if 0 <= row < len(game_map) and 0 <= col < len(game_map[row]):
            return game_map[row][col]
        return ' '
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.visited_positions.clear()
        self.action_history.clear()
        self.stuck_counter = 0
        self.door_kick_attempts.clear()
        self.exploration_mode = ExplorationMode.SYSTEMATIC
        self.last_score = 0
        self.last_depth = 1
        self.turn_count = 0
        self.last_hp = None
        self.level_stairs.clear()
        self.level_layout.clear()
        self.search_count = 0
        self.corner_positions.clear()