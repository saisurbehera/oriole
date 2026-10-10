"""
Expert-Guided RL Agent - Learn from expert patterns and use LLM for decisions
=============================================================================

This agent:
1. Loads expert demonstrations from trajectories
2. Learns common patterns and successful sequences
3. Uses LLM to select between learned strategies
4. Executes with heuristics and tracks outcomes
"""

import json
import random
import os
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import deque, defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import re

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse


@dataclass
class ExpertPattern:
    """A pattern learned from expert play"""
    name: str
    context_conditions: Dict[str, Any]  # Conditions when pattern applies
    action_sequence: List[str]  # Sequence of actions
    success_rate: float = 0.0
    usage_count: int = 0
    

@dataclass
class ExpertStrategy:
    """High-level strategy composed of patterns"""
    name: str
    goal: str
    patterns: List[ExpertPattern]
    context_priority: List[str]  # Which context matters most
    

class ExpertSkill(Enum):
    """Expert-level skills based on analysis"""
    STAIRS_RUSH = "rush_to_stairs"
    SYSTEMATIC_EXPLORE = "systematic_exploration"
    DOOR_SEQUENCE = "door_handling"
    COMBAT_KITE = "kite_monster"
    ITEM_COLLECTION = "collect_items"
    CORRIDOR_FOLLOW = "follow_corridors"
    ROOM_CLEAR = "clear_room"
    TRAP_CHECK = "check_traps"
    HUNGER_MANAGE = "manage_hunger"
    TACTICAL_RETREAT = "tactical_retreat"
    

class ExpertGuidedRLAgent(BaseAgent):
    """Agent that learns from expert demonstrations"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Expert knowledge
        self.expert_patterns = self._load_expert_patterns()
        self.expert_strategies = self._build_expert_strategies()
        
        # Current execution
        self.current_strategy = None
        self.current_pattern_idx = 0
        self.pattern_step = 0
        
        # State tracking
        self.visited_positions = set()
        self.action_history = deque(maxlen=100)
        self.pattern_history = deque(maxlen=20)
        self.turn_count = 0
        
        # Learning from experience
        self.pattern_outcomes = defaultdict(list)  # pattern_name -> outcomes
        self.strategy_rewards = defaultdict(float)  # strategy -> avg reward
        
        # Game state
        self.last_score = 0
        self.last_hp = None
        self.last_depth = 1
        self.stuck_counter = 0
        
    def act(self, obs, prev_action=None):
        """Select action using expert guidance"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        self.turn_count += 1
        
        # Parse state
        game_state = self._parse_game_state(obs)
        
        # Check if we need new strategy
        if self._should_select_new_strategy(game_state):
            # Record outcome of previous strategy
            if self.current_strategy:
                self._record_strategy_outcome(game_state)
                
            # Select new strategy with LLM
            self.current_strategy = self._select_strategy_with_llm(game_state)
            self.current_pattern_idx = 0
            self.pattern_step = 0
            
        # Execute current pattern
        action = self._execute_current_pattern(game_state, obs)
        
        # Track action
        self.action_history.append(action)
        
        # Update position
        if game_state["player_pos"]:
            self.visited_positions.add((game_state["depth"], game_state["player_pos"]))
            
        return LLMResponse(
            model_id="expert_guided_rl",
            completion=action,
            stop_reason="complete",
            input_tokens=100,  # Approximate
            output_tokens=10,
            reasoning=self._get_reasoning()
        )
        
    def _load_expert_patterns(self) -> Dict[str, ExpertPattern]:
        """Load patterns from expert trajectories"""
        patterns = {}
        
        # Common expert patterns from analysis
        patterns["stairs_direct"] = ExpertPattern(
            name="stairs_direct",
            context_conditions={"stairs_visible": True, "hp_ok": True},
            action_sequence=["navigate_to_stairs", "<"]
        )
        
        patterns["door_kick_sequence"] = ExpertPattern(
            name="door_kick_sequence", 
            context_conditions={"door_adjacent": True},
            action_sequence=["kick", "kick", "open", "move_through"]
        )
        
        patterns["spiral_explore"] = ExpertPattern(
            name="spiral_explore",
            context_conditions={"in_room": True, "no_stairs": True},
            action_sequence=["north", "east", "south", "south", "west", "west", "north", "north"]
        )
        
        patterns["corridor_follow"] = ExpertPattern(
            name="corridor_follow",
            context_conditions={"in_corridor": True},
            action_sequence=["follow_corridor", "follow_corridor", "check_branches"]
        )
        
        patterns["combat_retreat"] = ExpertPattern(
            name="combat_retreat",
            context_conditions={"monster_adjacent": True, "hp_low": True},
            action_sequence=["move_away", "move_away", "wait"]
        )
        
        patterns["systematic_room"] = ExpertPattern(
            name="systematic_room",
            context_conditions={"in_room": True, "room_not_explored": True},
            action_sequence=["explore_perimeter", "explore_center", "check_corners"]
        )
        
        patterns["item_pickup"] = ExpertPattern(
            name="item_pickup",
            context_conditions={"item_visible": True},
            action_sequence=["navigate_to_item", ","]
        )
        
        # Load from actual expert data if available
        expert_file = "/home/sai/Desktop/oriole/BALROG/expert_nethack_abstractions.json"
        if os.path.exists(expert_file):
            try:
                with open(expert_file, 'r') as f:
                    expert_data = json.load(f)
                    # Convert to patterns
                    for name, data in expert_data.items():
                        if isinstance(data, dict) and 'actions' in data:
                            patterns[f"expert_{name}"] = ExpertPattern(
                                name=f"expert_{name}",
                                context_conditions=data.get('context', {}),
                                action_sequence=data['actions']
                            )
            except:
                pass
                
        return patterns
        
    def _build_expert_strategies(self) -> Dict[str, ExpertStrategy]:
        """Build high-level strategies from patterns"""
        strategies = {}
        
        strategies["speed_descent"] = ExpertStrategy(
            name="speed_descent",
            goal="Reach deeper levels quickly",
            patterns=[
                self.expert_patterns["stairs_direct"],
                self.expert_patterns["corridor_follow"],
                self.expert_patterns["door_kick_sequence"]
            ],
            context_priority=["stairs_visible", "hp_ok", "no_monsters"]
        )
        
        strategies["full_exploration"] = ExpertStrategy(
            name="full_exploration",
            goal="Explore level thoroughly",
            patterns=[
                self.expert_patterns["systematic_room"],
                self.expert_patterns["spiral_explore"],
                self.expert_patterns["item_pickup"]
            ],
            context_priority=["room_not_explored", "items_visible"]
        )
        
        strategies["tactical_combat"] = ExpertStrategy(
            name="tactical_combat",
            goal="Handle combat intelligently",
            patterns=[
                self.expert_patterns["combat_retreat"],
                self.expert_patterns["corridor_follow"]
            ],
            context_priority=["hp_low", "monster_count", "escape_route"]
        )
        
        return strategies
        
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Parse comprehensive game state"""
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        lang_obs = text.get("language_observation", "").lower()
        
        # Basic parsing
        game_map = self._parse_map(obs)
        player_pos = self._find_player(game_map)
        
        # HP and stats
        hp_match = re.search(r'HP:(\d+)\((\d+)\)', long_text)
        hp_percent = 1.0
        if hp_match:
            hp_percent = int(hp_match.group(1)) / int(hp_match.group(2))
            
        # Depth
        depth_match = re.search(r'Dlvl:(\d+)', long_text)
        depth = int(depth_match.group(1)) if depth_match else 1
        
        # Context conditions
        state = {
            "player_pos": player_pos,
            "depth": depth,
            "hp_percent": hp_percent,
            "hp_ok": hp_percent > 0.5,
            "hp_low": hp_percent < 0.3,
            "stairs_visible": self._find_stairs(game_map) is not None,
            "monster_adjacent": self._check_monsters_adjacent(game_map, player_pos),
            "door_adjacent": self._check_door_adjacent(game_map, player_pos),
            "in_room": self._in_room(game_map, player_pos),
            "in_corridor": self._in_corridor(game_map, player_pos),
            "item_visible": self._check_items_visible(game_map),
            "no_monsters": "hostile" not in lang_obs and "monster" not in lang_obs,
            "room_not_explored": len(self.visited_positions) < 50,
            "no_stairs": "stairs" not in lang_obs,
            "game_map": game_map
        }
        
        return state
        
    def _should_select_new_strategy(self, game_state) -> bool:
        """Decide if we need a new strategy"""
        if not self.current_strategy:
            return True
            
        # Strategy completed
        if self.current_pattern_idx >= len(self.current_strategy.patterns):
            return True
            
        # Emergency override
        if game_state["hp_low"] and self.current_strategy.name != "tactical_combat":
            return True
            
        # Stuck detection
        if self.stuck_counter > 20:
            return True
            
        return False
        
    def _select_strategy_with_llm(self, game_state) -> ExpertStrategy:
        """Use LLM to select strategy"""
        # Build expert-like prompt with learned knowledge
        prompt = f"""NetHack Expert Strategy Selection:

Game State:
- Depth: {game_state['depth']}
- HP: {int(game_state['hp_percent']*100)}%
- Stairs: {'visible' if game_state['stairs_visible'] else 'not visible'}
- Monsters: {'nearby' if game_state['monster_adjacent'] else 'clear'}
- Location: {'room' if game_state['in_room'] else 'corridor'}

Available Strategies:
1. speed_descent - Rush to deeper levels
2. full_exploration - Explore thoroughly 
3. tactical_combat - Handle threats carefully

Strategy Performance:
{self._format_strategy_performance()}

Previous strategies: {[s.name for s in self.pattern_history[-3:]] if self.pattern_history else 'None'}

Choose strategy (name only):"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.generate(messages)
            strategy_name = response.completion.strip().lower()
            
            # Find matching strategy
            for name, strategy in self.expert_strategies.items():
                if name in strategy_name or strategy_name in name:
                    return strategy
                    
        except:
            pass
            
        # Fallback logic
        if game_state["stairs_visible"] and game_state["hp_ok"]:
            return self.expert_strategies["speed_descent"]
        elif game_state["hp_low"]:
            return self.expert_strategies["tactical_combat"]
        else:
            return self.expert_strategies["full_exploration"]
            
    def _execute_current_pattern(self, game_state, obs) -> str:
        """Execute the current pattern in the strategy"""
        if not self.current_strategy:
            return "wait"
            
        # Get current pattern
        if self.current_pattern_idx >= len(self.current_strategy.patterns):
            return "wait"
            
        pattern = self.current_strategy.patterns[self.current_pattern_idx]
        
        # Check if pattern conditions are met
        if not self._check_pattern_conditions(pattern, game_state):
            # Skip to next pattern
            self.current_pattern_idx += 1
            self.pattern_step = 0
            return self._execute_current_pattern(game_state, obs)
            
        # Get action from pattern
        if self.pattern_step >= len(pattern.action_sequence):
            # Pattern completed
            self.current_pattern_idx += 1
            self.pattern_step = 0
            return self._execute_current_pattern(game_state, obs)
            
        action_template = pattern.action_sequence[self.pattern_step]
        action = self._resolve_action_template(action_template, game_state)
        
        self.pattern_step += 1
        
        return action
        
    def _check_pattern_conditions(self, pattern: ExpertPattern, game_state) -> bool:
        """Check if pattern conditions are met"""
        for condition, expected in pattern.context_conditions.items():
            if condition in game_state:
                if game_state[condition] != expected:
                    return False
        return True
        
    def _resolve_action_template(self, template: str, game_state) -> str:
        """Convert action template to concrete action"""
        # Handle special templates
        if template == "navigate_to_stairs":
            stairs_pos = self._find_stairs(game_state["game_map"])
            if stairs_pos and game_state["player_pos"]:
                return self._navigate_to(game_state["player_pos"], stairs_pos, game_state["game_map"])
                
        elif template == "navigate_to_item":
            item_pos = self._find_nearest_item(game_state["game_map"], game_state["player_pos"])
            if item_pos and game_state["player_pos"]:
                return self._navigate_to(game_state["player_pos"], item_pos, game_state["game_map"])
                
        elif template == "move_away":
            return self._move_away_from_danger(game_state["game_map"], game_state["player_pos"])
            
        elif template == "follow_corridor":
            return self._follow_corridor(game_state["game_map"], game_state["player_pos"])
            
        elif template == "explore_perimeter":
            return self._explore_perimeter(game_state["game_map"], game_state["player_pos"])
            
        elif template == "move_through":
            # Move through opened door
            valid = self._get_valid_moves(game_state["game_map"], game_state["player_pos"])
            if valid:
                return valid[0][0]
                
        # Direct actions
        elif template in ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]:
            return template
        elif template in ["kick", "open", "search", "wait", ",", "<", ">"]:
            return template
            
        # Default
        return "wait"
        
    def _record_strategy_outcome(self, end_state):
        """Record how well the strategy performed"""
        if not self.current_strategy:
            return
            
        # Calculate reward
        reward = 0.0
        
        # Depth progress
        if end_state["depth"] > self.last_depth:
            reward += 10.0
            
        # HP preservation  
        if end_state["hp_percent"] > 0.8:
            reward += 2.0
        elif end_state["hp_percent"] < 0.3:
            reward -= 5.0
            
        # Exploration
        new_positions = len(self.visited_positions) - getattr(self, '_strategy_start_positions', 0)
        reward += new_positions * 0.1
        
        # Update strategy performance
        self.strategy_rewards[self.current_strategy.name] = (
            0.9 * self.strategy_rewards[self.current_strategy.name] + 0.1 * reward
        )
        
    def _format_strategy_performance(self) -> str:
        """Format strategy performance for prompt"""
        lines = []
        for name, avg_reward in sorted(self.strategy_rewards.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {name}: {avg_reward:.1f} avg reward")
        return "\n".join(lines) if lines else "No performance data yet"
        
    def _get_reasoning(self) -> str:
        """Get current reasoning"""
        if self.current_strategy:
            pattern_name = "none"
            if self.current_pattern_idx < len(self.current_strategy.patterns):
                pattern_name = self.current_strategy.patterns[self.current_pattern_idx].name
            return f"Strategy: {self.current_strategy.name} | Pattern: {pattern_name}"
        return "No strategy"
        
    # Utility methods
    def _parse_map(self, obs):
        """Parse the game map"""
        text = obs.get("text", {})
        map_text = text.get("long_term_context", "")
        lines = map_text.split('\n')
        
        map_start = -1
        for i, line in enumerate(lines):
            if line.strip() == "map:":
                map_start = i + 1
                break
                
        if map_start == -1:
            return []
            
        game_map = []
        for i in range(map_start, min(map_start + 21, len(lines))):
            if i < len(lines):
                if lines[i].strip().startswith('Agent'):
                    break
                game_map.append(lines[i])
                
        return game_map
        
    def _find_player(self, game_map):
        """Find player position"""
        for row, line in enumerate(game_map):
            if '@' in line:
                return (row, line.index('@'))
        return None
        
    def _find_stairs(self, game_map):
        """Find stairs position"""
        for row, line in enumerate(game_map):
            if '>' in line:
                return (row, line.index('>'))
            elif '<' in line:
                return (row, line.index('<'))
        return None
        
    def _check_monsters_adjacent(self, game_map, player_pos):
        """Check for adjacent monsters"""
        if not player_pos:
            return False
            
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                row, col = player_pos[0] + dr, player_pos[1] + dc
                if 0 <= row < len(game_map) and 0 <= col < len(game_map[row]):
                    char = game_map[row][col]
                    # Common monster chars
                    if char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:;&':
                        return True
        return False
        
    def _check_door_adjacent(self, game_map, player_pos):
        """Check for adjacent doors"""
        if not player_pos:
            return False
            
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                row, col = player_pos[0] + dr, player_pos[1] + dc
                if 0 <= row < len(game_map) and 0 <= col < len(game_map[row]):
                    if game_map[row][col] == '+':
                        return True
        return False
        
    def _in_room(self, game_map, player_pos):
        """Check if in a room"""
        if not player_pos:
            return False
        char = self._get_char_at(game_map, player_pos)
        return char == '.'
        
    def _in_corridor(self, game_map, player_pos):
        """Check if in corridor"""
        if not player_pos:
            return False
        char = self._get_char_at(game_map, player_pos)
        return char == '#'
        
    def _check_items_visible(self, game_map):
        """Check if items are visible"""
        for line in game_map:
            if any(c in line for c in '$!?/="()[]*%'):
                return True
        return False
        
    def _find_nearest_item(self, game_map, player_pos):
        """Find nearest item position"""
        if not player_pos:
            return None
            
        items = []
        for row, line in enumerate(game_map):
            for col, char in enumerate(line):
                if char in '$!?/="()[]*%':
                    items.append((row, col))
                    
        if not items:
            return None
            
        # Find closest
        return min(items, key=lambda p: abs(p[0]-player_pos[0]) + abs(p[1]-player_pos[1]))
        
    def _navigate_to(self, from_pos, to_pos, game_map):
        """Simple navigation"""
        d_row = to_pos[0] - from_pos[0]
        d_col = to_pos[1] - from_pos[1]
        
        # Try direct approach
        if d_row < 0 and self._is_walkable(game_map, from_pos[0]-1, from_pos[1]):
            return "north"
        elif d_row > 0 and self._is_walkable(game_map, from_pos[0]+1, from_pos[1]):
            return "south"
        elif d_col < 0 and self._is_walkable(game_map, from_pos[0], from_pos[1]-1):
            return "west"
        elif d_col > 0 and self._is_walkable(game_map, from_pos[0], from_pos[1]+1):
            return "east"
            
        # Any valid move
        valid = self._get_valid_moves(game_map, from_pos)
        if valid:
            return valid[0][0]
        return "wait"
        
    def _move_away_from_danger(self, game_map, player_pos):
        """Move away from threats"""
        valid = self._get_valid_moves(game_map, player_pos)
        if valid:
            # Prefer moves that lead to open space
            best = max(valid, key=lambda m: self._count_open_neighbors(game_map, m[1]))
            return best[0]
        return "wait"
        
    def _follow_corridor(self, game_map, player_pos):
        """Follow corridor tiles"""
        valid = self._get_valid_moves(game_map, player_pos)
        
        # Prefer corridor tiles
        corridor_moves = [m for m in valid if self._get_char_at(game_map, m[1]) == '#']
        if corridor_moves:
            return corridor_moves[0][0]
            
        if valid:
            return valid[0][0]
        return "wait"
        
    def _explore_perimeter(self, game_map, player_pos):
        """Explore room perimeter"""
        # Simple wall following
        valid = self._get_valid_moves(game_map, player_pos)
        
        # Prefer moves along walls
        wall_moves = []
        for direction, pos in valid:
            if self._next_to_wall(game_map, pos):
                wall_moves.append(direction)
                
        if wall_moves:
            return random.choice(wall_moves)
            
        if valid:
            return valid[0][0]
        return "wait"
        
    def _get_valid_moves(self, game_map, pos):
        """Get valid moves from position"""
        if not pos:
            return []
            
        moves = []
        directions = [
            ("north", pos[0]-1, pos[1]),
            ("south", pos[0]+1, pos[1]),
            ("east", pos[0], pos[1]+1),
            ("west", pos[0], pos[1]-1)
        ]
        
        for direction, row, col in directions:
            if self._is_walkable(game_map, row, col):
                moves.append((direction, (row, col)))
                
        return moves
        
    def _is_walkable(self, game_map, row, col):
        """Check if position is walkable"""
        if 0 <= row < len(game_map) and 0 <= col < len(game_map[row]):
            return game_map[row][col] in '.#<>+ '
        return False
        
    def _get_char_at(self, game_map, pos):
        """Get character at position"""
        row, col = pos
        if 0 <= row < len(game_map) and 0 <= col < len(game_map[row]):
            return game_map[row][col]
        return ' '
        
    def _count_open_neighbors(self, game_map, pos):
        """Count open neighbors"""
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if self._is_walkable(game_map, pos[0]+dr, pos[1]+dc):
                    count += 1
        return count
        
    def _next_to_wall(self, game_map, pos):
        """Check if next to wall"""
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                row, col = pos[0]+dr, pos[1]+dc
                if 0 <= row < len(game_map) and 0 <= col < len(game_map[row]):
                    if game_map[row][col] in '|-':
                        return True
        return False
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.current_strategy = None
        self.current_pattern_idx = 0
        self.pattern_step = 0
        self.visited_positions.clear()
        self.action_history.clear()
        self.pattern_history.clear()
        self.turn_count = 0
        self.last_score = 0
        self.last_hp = None
        self.last_depth = 1
        self.stuck_counter = 0
        self._strategy_start_positions = 0