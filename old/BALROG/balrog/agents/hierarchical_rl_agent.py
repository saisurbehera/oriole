"""
Hierarchical RL Agent - LLM for high-level decisions, heuristics for execution
==============================================================================

This agent uses a hierarchical approach:
1. LLM selects high-level skills/goals
2. Heuristics execute the selected skills
3. Experience is collected for learning
4. Reward shaping guides skill selection

The key insight: Let the LLM choose WHAT to do, use heuristics for HOW to do it.
"""

import json
import random
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import re

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse


class Skill(Enum):
    """High-level skills the agent can execute"""
    EXPLORE_NEW_AREA = "explore_new"
    FIND_STAIRS_DOWN = "find_stairs_down" 
    DESCEND_STAIRS = "descend"
    COMBAT_MELEE = "fight_melee"
    FLEE_COMBAT = "flee"
    PICKUP_ITEMS = "pickup_items"
    SEARCH_DOORS = "search_doors"
    OPEN_DOOR = "open_door"
    EAT_FOOD = "eat"
    NAVIGATE_TO_ITEM = "go_to_item"
    FOLLOW_CORRIDOR = "follow_corridor"
    EXPLORE_ROOM = "explore_room"
    REST_HEAL = "rest"
    
    
@dataclass
class SkillExperience:
    """Track experience with each skill"""
    skill: Skill
    context: Dict[str, Any]
    reward: float
    success: bool
    steps: int
    

@dataclass 
class GameContext:
    """Simplified game context for LLM"""
    hp_percent: float
    hunger: str
    depth: int
    has_stairs: bool
    monsters_nearby: bool
    items_nearby: bool
    doors_nearby: bool
    in_corridor: bool
    in_room: bool
    explored_percent: float
    stuck: bool
    turn: int
    

class HierarchicalRLAgent(BaseAgent):
    """Agent using hierarchical RL with LLM for skill selection"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Skill execution state
        self.current_skill = None
        self.skill_steps = 0
        self.skill_start_context = None
        self.max_skill_steps = 50
        
        # Memory
        self.visited_positions = set()
        self.action_history = deque(maxlen=100)
        self.skill_history = deque(maxlen=20)
        self.position_history = deque(maxlen=50)
        
        # Learning
        self.skill_experience = defaultdict(list)  # skill -> list of experiences
        self.skill_success_rate = defaultdict(lambda: 0.5)  # skill -> success rate
        
        # State tracking
        self.last_score = 0
        self.last_hp = None
        self.last_depth = 1
        self.turn_count = 0
        self.stuck_counter = 0
        
        # Skill execution
        self.skill_executor = SkillExecutor()
        
    def act(self, obs, prev_action=None):
        """Main action selection using hierarchical approach"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        self.turn_count += 1
        
        # Parse game state
        context = self._extract_game_context(obs)
        game_map = self._parse_map(obs)
        player_pos = self._find_player(game_map)
        
        # Update position tracking
        if player_pos:
            self.visited_positions.add((context.depth, player_pos))
            self.position_history.append(player_pos)
            
        # Check if current skill completed or failed
        if self._should_select_new_skill(context):
            # Record experience from previous skill
            if self.current_skill:
                self._record_skill_experience(context)
                
            # Select new skill using LLM
            self.current_skill = self._select_skill_with_llm(context)
            self.skill_steps = 0
            self.skill_start_context = context
            self.skill_history.append(self.current_skill)
            
        # Execute current skill with heuristics
        action = self.skill_executor.execute_skill(
            self.current_skill, 
            context, 
            game_map, 
            player_pos,
            self.visited_positions
        )
        
        self.skill_steps += 1
        self.action_history.append(action)
        
        # Update stuck detection
        if len(set(self.position_history)) < 3:  # Not moving much
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        return LLMResponse(
            model_id="hierarchical_rl",
            completion=action,
            stop_reason="complete",
            input_tokens=50,  # Approximate
            output_tokens=5,
            reasoning=f"Skill: {self.current_skill.value if self.current_skill else 'None'}"
        )
        
    def _extract_game_context(self, obs) -> GameContext:
        """Extract simplified context for LLM decision making"""
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        
        # Parse HP
        hp_match = re.search(r'HP:(\d+)\((\d+)\)', long_text)
        hp_percent = 1.0
        if hp_match:
            current = int(hp_match.group(1))
            max_hp = int(hp_match.group(2))
            hp_percent = current / max_hp if max_hp > 0 else 1.0
            
        # Parse depth
        depth_match = re.search(r'Dlvl:(\d+)', long_text)
        depth = int(depth_match.group(1)) if depth_match else 1
        
        # Parse hunger
        hunger = "normal"
        if "hungry" in long_text.lower():
            hunger = "hungry"
        elif "weak" in long_text.lower():
            hunger = "weak"
        elif "fainting" in long_text.lower():
            hunger = "fainting"
            
        # Check game elements
        lang_obs = text.get("language_observation", "").lower()
        has_stairs = "stairs" in lang_obs
        monsters_nearby = any(m in lang_obs for m in ["goblin", "kobold", "jackal", "hostile"])
        items_nearby = any(i in lang_obs for i in ["gold", "potion", "scroll", "weapon"])
        doors_nearby = "door" in lang_obs or "doorway" in lang_obs
        
        # Check location type
        in_corridor = "corridor" in lang_obs or "passage" in lang_obs
        in_room = "room" in lang_obs or not in_corridor
        
        # Exploration estimate
        explored_percent = min(len(self.visited_positions) / 200.0, 1.0)
        
        return GameContext(
            hp_percent=hp_percent,
            hunger=hunger,
            depth=depth,
            has_stairs=has_stairs,
            monsters_nearby=monsters_nearby,
            items_nearby=items_nearby,
            doors_nearby=doors_nearby,
            in_corridor=in_corridor,
            in_room=in_room,
            explored_percent=explored_percent,
            stuck=self.stuck_counter > 10,
            turn=self.turn_count
        )
        
    def _should_select_new_skill(self, context: GameContext) -> bool:
        """Decide if we need to select a new skill"""
        # No current skill
        if not self.current_skill:
            return True
            
        # Skill ran too long
        if self.skill_steps >= self.max_skill_steps:
            return True
            
        # Emergency situations
        if context.hp_percent < 0.3 and self.current_skill != Skill.FLEE_COMBAT:
            return True
            
        if context.hunger == "fainting" and self.current_skill != Skill.EAT_FOOD:
            return True
            
        # Stuck
        if context.stuck:
            return True
            
        # Skill completed (heuristic)
        if self.current_skill == Skill.DESCEND_STAIRS and context.depth > self.last_depth:
            return True
            
        if self.current_skill == Skill.PICKUP_ITEMS and not context.items_nearby:
            return True
            
        return False
        
    def _select_skill_with_llm(self, context: GameContext) -> Skill:
        """Use LLM to select the next skill"""
        # Build expert-like prompt
        prompt = self._build_skill_selection_prompt(context)
        
        messages = [{
            "role": "user",
            "content": prompt
        }]
        
        try:
            response = self.client.generate(messages)
            skill_name = response.completion.strip().upper()
            
            # Parse skill from response
            for skill in Skill:
                if skill.name == skill_name or skill.value in response.completion.lower():
                    return skill
                    
            # Fallback to most appropriate skill
            return self._get_fallback_skill(context)
            
        except Exception as e:
            # Fallback on error
            return self._get_fallback_skill(context)
            
    def _build_skill_selection_prompt(self, context: GameContext) -> str:
        """Build expert-like prompt for skill selection"""
        # Get skill success rates
        skill_stats = []
        for skill in Skill:
            rate = self.skill_success_rate[skill]
            skill_stats.append(f"{skill.name}: {rate:.0%}")
            
        prompt = f"""NetHack Expert Decision:

Current State:
- HP: {int(context.hp_percent*100)}%
- Hunger: {context.hunger}
- Depth: {context.depth}
- Turn: {context.turn}

Observations:
- Stairs nearby: {context.has_stairs}
- Monsters: {context.monsters_nearby}
- Items: {context.items_nearby}  
- Doors: {context.doors_nearby}
- Location: {'corridor' if context.in_corridor else 'room'}
- Explored: {int(context.explored_percent*100)}%
- Stuck: {context.stuck}

Available Skills:
{chr(10).join(f'- {skill.name}' for skill in Skill)}

Recent Skills: {[s.name for s in list(self.skill_history)[-3:]]}

Choose ONE skill name:"""
        
        return prompt
        
    def _get_fallback_skill(self, context: GameContext) -> Skill:
        """Get appropriate skill based on context without LLM"""
        # Emergency priorities
        if context.hp_percent < 0.3:
            return Skill.FLEE_COMBAT if context.monsters_nearby else Skill.REST_HEAL
            
        if context.hunger == "fainting":
            return Skill.EAT_FOOD
            
        # Normal priorities
        if context.has_stairs and context.explored_percent > 0.5:
            return Skill.FIND_STAIRS_DOWN
            
        if context.monsters_nearby and context.hp_percent > 0.7:
            return Skill.COMBAT_MELEE
            
        if context.items_nearby:
            return Skill.PICKUP_ITEMS
            
        if context.doors_nearby:
            return Skill.OPEN_DOOR
            
        # Default exploration
        if context.in_corridor:
            return Skill.FOLLOW_CORRIDOR
        else:
            return Skill.EXPLORE_NEW_AREA
            
    def _record_skill_experience(self, end_context: GameContext):
        """Record experience from skill execution"""
        if not self.current_skill or not self.skill_start_context:
            return
            
        # Calculate reward
        reward = 0.0
        
        # Progress rewards
        if end_context.depth > self.skill_start_context.depth:
            reward += 10.0
            
        score_diff = self.last_score - getattr(self, '_skill_start_score', 0)
        if score_diff > 0:
            reward += score_diff / 100.0
            
        # Exploration reward
        explore_diff = end_context.explored_percent - self.skill_start_context.explored_percent
        reward += explore_diff * 5.0
        
        # Penalties
        hp_loss = self.skill_start_context.hp_percent - end_context.hp_percent
        reward -= hp_loss * 10.0
        
        if end_context.stuck:
            reward -= 2.0
            
        # Determine success
        success = reward > 0
        
        # Record experience
        exp = SkillExperience(
            skill=self.current_skill,
            context=vars(self.skill_start_context),
            reward=reward,
            success=success,
            steps=self.skill_steps
        )
        
        self.skill_experience[self.current_skill].append(exp)
        
        # Update success rate (simple moving average)
        experiences = self.skill_experience[self.current_skill][-10:]
        success_rate = sum(e.success for e in experiences) / len(experiences)
        self.skill_success_rate[self.current_skill] = success_rate
        
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
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.current_skill = None
        self.skill_steps = 0
        self.skill_start_context = None
        self.visited_positions.clear()
        self.action_history.clear()
        self.skill_history.clear()
        self.position_history.clear()
        self.last_score = 0
        self.last_hp = None
        self.last_depth = 1
        self.turn_count = 0
        self.stuck_counter = 0
        

class SkillExecutor:
    """Executes skills using heuristics"""
    
    def execute_skill(self, skill: Skill, context: GameContext, 
                     game_map: List[str], player_pos: Optional[Tuple[int, int]],
                     visited_positions: set) -> str:
        """Execute a skill and return the action"""
        
        if not skill or not player_pos:
            return "wait"
            
        # Dispatch to skill-specific execution
        if skill == Skill.EXPLORE_NEW_AREA:
            return self._explore_new_area(game_map, player_pos, visited_positions, context)
        elif skill == Skill.FIND_STAIRS_DOWN:
            return self._find_stairs(game_map, player_pos)
        elif skill == Skill.DESCEND_STAIRS:
            return self._descend_stairs(game_map, player_pos)
        elif skill == Skill.COMBAT_MELEE:
            return self._combat_melee(game_map, player_pos)
        elif skill == Skill.FLEE_COMBAT:
            return self._flee_combat(game_map, player_pos)
        elif skill == Skill.PICKUP_ITEMS:
            return self._pickup_items(game_map, player_pos)
        elif skill == Skill.SEARCH_DOORS:
            return "search"
        elif skill == Skill.OPEN_DOOR:
            return self._open_door(game_map, player_pos)
        elif skill == Skill.EAT_FOOD:
            return "eat"
        elif skill == Skill.NAVIGATE_TO_ITEM:
            return self._navigate_to_item(game_map, player_pos)
        elif skill == Skill.FOLLOW_CORRIDOR:
            return self._follow_corridor(game_map, player_pos)
        elif skill == Skill.EXPLORE_ROOM:
            return self._explore_room(game_map, player_pos, visited_positions)
        elif skill == Skill.REST_HEAL:
            return "wait"
        else:
            return self._explore_new_area(game_map, player_pos, visited_positions, context)
            
    def _explore_new_area(self, game_map, player_pos, visited_positions, context):
        """Explore unvisited areas"""
        valid_moves = self._get_valid_moves(game_map, player_pos)
        
        # Prefer unvisited
        unvisited = []
        for direction, new_pos in valid_moves:
            if (context.depth, new_pos) not in visited_positions:
                unvisited.append(direction)
                
        if unvisited:
            return random.choice(unvisited)
            
        # Move toward open spaces
        if valid_moves:
            return random.choice(valid_moves)[0]
            
        return "search"
        
    def _find_stairs(self, game_map, player_pos):
        """Navigate to stairs"""
        stairs_pos = None
        for row, line in enumerate(game_map):
            if '>' in line:
                stairs_pos = (row, line.index('>'))
                break
            elif '<' in line:
                stairs_pos = (row, line.index('<'))
                break
                
        if stairs_pos:
            return self._navigate_to(player_pos, stairs_pos, game_map)
        
        # Explore to find stairs
        return self._explore_new_area(game_map, player_pos, set(), GameContext(0,0,0,0,0,0,0,0,0,0,0,0))
        
    def _descend_stairs(self, game_map, player_pos):
        """Use stairs if on them"""
        if player_pos:
            char = self._get_char_at(game_map, player_pos)
            if char == '>':
                return '>'
            elif char == '<':
                return '<'
                
        # Navigate to stairs first
        return self._find_stairs(game_map, player_pos)
        
    def _combat_melee(self, game_map, player_pos):
        """Simple combat - move toward monster"""
        # Look for monster symbols
        directions = self._get_adjacent_positions(player_pos)
        
        for direction, pos in directions:
            char = self._get_char_at(game_map, pos)
            # Common monster symbols
            if char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:;&':
                return direction
                
        # Move randomly if no adjacent monster
        valid = self._get_valid_moves(game_map, player_pos)
        if valid:
            return random.choice(valid)[0]
        return "wait"
        
    def _flee_combat(self, game_map, player_pos):
        """Run away from monsters"""
        # Simple: move to any open space
        valid = self._get_valid_moves(game_map, player_pos)
        if valid:
            # Prefer moves that increase distance from center
            return random.choice(valid)[0]
        return "search"
        
    def _pickup_items(self, game_map, player_pos):
        """Pickup items on ground"""
        # Check if on item
        items_here = self._check_items_at_position(game_map, player_pos)
        if items_here:
            return ","  # pickup command
            
        # Look for nearby items
        for direction, pos in self._get_adjacent_positions(player_pos):
            if self._check_items_at_position(game_map, pos):
                return direction
                
        # Explore to find items
        return self._explore_new_area(game_map, player_pos, set(), GameContext(0,0,0,0,0,0,0,0,0,0,0,0))
        
    def _open_door(self, game_map, player_pos):
        """Handle doors"""
        # Look for adjacent doors
        for direction, pos in self._get_adjacent_positions(player_pos):
            char = self._get_char_at(game_map, pos)
            if char == '+':  # Closed door
                # Alternate kick and open
                return random.choice(["kick", "open"])
                
        return "search"
        
    def _navigate_to_item(self, game_map, player_pos):
        """Navigate to visible items"""
        # Find closest item
        item_positions = []
        for row, line in enumerate(game_map):
            for col, char in enumerate(line):
                if char in '$!?/="()[]*%':  # Common item symbols
                    item_positions.append((row, col))
                    
        if item_positions:
            # Find closest
            closest = min(item_positions, 
                         key=lambda p: abs(p[0]-player_pos[0]) + abs(p[1]-player_pos[1]))
            return self._navigate_to(player_pos, closest, game_map)
            
        return self._explore_new_area(game_map, player_pos, set(), GameContext(0,0,0,0,0,0,0,0,0,0,0,0))
        
    def _follow_corridor(self, game_map, player_pos):
        """Follow corridor tiles"""
        valid = self._get_valid_moves(game_map, player_pos)
        
        # Prefer corridor tiles
        corridor_moves = []
        for direction, pos in valid:
            if self._get_char_at(game_map, pos) == '#':
                corridor_moves.append(direction)
                
        if corridor_moves:
            return random.choice(corridor_moves)
            
        if valid:
            return valid[0][0]
            
        return "wait"
        
    def _explore_room(self, game_map, player_pos, visited_positions):
        """Systematic room exploration"""
        valid = self._get_valid_moves(game_map, player_pos)
        
        # Prefer unvisited in room
        unvisited = []
        for direction, pos in valid:
            if pos not in visited_positions and self._get_char_at(game_map, pos) == '.':
                unvisited.append(direction)
                
        if unvisited:
            return random.choice(unvisited)
            
        if valid:
            return random.choice(valid)[0]
            
        return "search"
        
    def _navigate_to(self, from_pos, to_pos, game_map):
        """Simple pathfinding"""
        d_row = to_pos[0] - from_pos[0]
        d_col = to_pos[1] - from_pos[1]
        
        # Try direct approach
        candidates = []
        
        if d_row < 0:
            candidates.append(("north", from_pos[0] - 1, from_pos[1]))
        elif d_row > 0:
            candidates.append(("south", from_pos[0] + 1, from_pos[1]))
            
        if d_col < 0:
            candidates.append(("west", from_pos[0], from_pos[1] - 1))
        elif d_col > 0:
            candidates.append(("east", from_pos[0], from_pos[1] + 1))
            
        for direction, new_row, new_col in candidates:
            if self._is_walkable(game_map, new_row, new_col):
                return direction
                
        # Try any valid move
        valid = self._get_valid_moves(game_map, from_pos)
        if valid:
            return valid[0][0]
            
        return "wait"
        
    def _get_valid_moves(self, game_map, pos):
        """Get all valid moves from position"""
        if not pos:
            return []
            
        moves = []
        for direction, new_pos in self._get_adjacent_positions(pos):
            if self._is_walkable(game_map, new_pos[0], new_pos[1]):
                moves.append((direction, new_pos))
                
        return moves
        
    def _get_adjacent_positions(self, pos):
        """Get adjacent positions with directions"""
        row, col = pos
        return [
            ("north", (row-1, col)),
            ("south", (row+1, col)),
            ("east", (row, col+1)),
            ("west", (row, col-1)),
            ("northeast", (row-1, col+1)),
            ("northwest", (row-1, col-1)),
            ("southeast", (row+1, col+1)),
            ("southwest", (row+1, col-1))
        ]
        
    def _is_walkable(self, game_map, row, col):
        """Check if position is walkable"""
        if 0 <= row < len(game_map) and 0 <= col < len(game_map[row]):
            char = game_map[row][col]
            return char in '.#<>+ '
        return False
        
    def _get_char_at(self, game_map, pos):
        """Get character at position"""
        row, col = pos
        if 0 <= row < len(game_map) and 0 <= col < len(game_map[row]):
            return game_map[row][col]
        return ' '
        
    def _check_items_at_position(self, game_map, pos):
        """Check if items at position"""
        char = self._get_char_at(game_map, pos)
        return char in '$!?/="()[]*%'