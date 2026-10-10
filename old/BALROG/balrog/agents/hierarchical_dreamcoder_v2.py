"""
Hierarchical DreamCoder Agent V2 for BALROG
===========================================

This agent combines:
1. DreamCoder's learning capabilities
2. Hierarchical planning with region detection
3. LLM-driven decision making at multiple levels
4. Practical pathfinding integration
"""

import json
import re
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class PlanLevel(Enum):
    STRATEGIC = 4     # "Clear the dungeon"
    TACTICAL = 3      # "Explore eastern rooms"  
    OPERATIONAL = 2   # "Go to room 3"
    IMMEDIATE = 1     # "north"

@dataclass
class Plan:
    """A single plan at any level"""
    level: PlanLevel
    goal: str
    status: str = "active"  # active, completed, failed
    steps_taken: int = 0
    max_steps: int = 100
    context: Dict = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

@dataclass
class Region:
    """A detected region in the game"""
    id: str
    type: str  # room, corridor, doorway
    bounds: Tuple[int, int, int, int]  # min_y, min_x, max_y, max_x
    explored: float = 0.0
    has_stairs: bool = False
    has_monsters: bool = False
    has_items: bool = False
    connected_to: Set[str] = None
    
    def __post_init__(self):
        if self.connected_to is None:
            self.connected_to = set()

# ============================================================================
# WORLD MODEL WITH REGION DETECTION
# ============================================================================

class HierarchicalWorldModel:
    """Enhanced world model with region detection"""
    
    def __init__(self):
        self.regions: Dict[str, Region] = {}
        self.current_region_id: Optional[str] = None
        self.player_pos: Optional[Tuple[int, int]] = None
        self.stairs_pos: Optional[Tuple[int, int]] = None
        self.monsters_present = False
        self.doors_present = False
        self.items_visible = False
        self.player_status = {
            "hungry": False,
            "hp_low": False,
            "has_weapon": False,
            "has_food": False
        }
        
    def update_from_obs(self, obs: Dict):
        """Update world model including region detection"""
        text = obs.get("text", {})
        long_term = text.get("long_term_context", "").lower()
        short_term = text.get("short_term_context", "").lower()
        
        # Basic feature detection
        self.stairs_pos = self._find_symbol_pos(long_term, ">") or self._find_symbol_pos(long_term, "<")
        self.monsters_present = any(w in long_term for w in ["monster", "attacks", "hits", "d"])  # 'd' is dog
        self.doors_present = "+" in long_term or "door" in long_term
        self.items_visible = any(c in long_term for c in [")", "[", "?", "!", "*", ":", "{"])
        
        # Player status
        self.player_status["hungry"] = "hungry" in short_term
        self.player_status["hp_low"] = self._check_hp_low(short_term)
        
        # Region detection
        self._detect_regions(long_term)
        
    def _find_symbol_pos(self, text: str, symbol: str) -> Optional[Tuple[int, int]]:
        """Find position of a symbol in the text map"""
        lines = text.strip().split('\n')
        for y, line in enumerate(lines):
            if symbol in line:
                x = line.index(symbol)
                return (y, x)
        return None
        
    def _check_hp_low(self, text: str) -> bool:
        """Check if HP is low"""
        match = re.search(r'hp:\s*(\d+)\((\d+)\)', text)
        if match:
            current = int(match.group(1))
            maximum = int(match.group(2))
            return current < maximum * 0.3
        return False
        
    def _detect_regions(self, ascii_map: str):
        """Detect rooms and corridors from ASCII map"""
        lines = ascii_map.strip().split('\n')
        if not lines:
            return
            
        # Simple region detection based on patterns
        # Rooms are areas enclosed by walls (-|)
        # Corridors are # symbols
        
        # Find player position
        for y, line in enumerate(lines):
            if '@' in line:
                self.player_pos = (y, line.index('@'))
                break
                
        # Detect current region type
        if self.player_pos:
            y, x = self.player_pos
            if y < len(lines) and x < len(lines[y]):
                current_char = lines[y][x] if lines[y][x] != '@' else '.'
                
                if current_char == '.':
                    # In a room
                    self.current_region_id = "room_current"
                    if "room_current" not in self.regions:
                        self.regions["room_current"] = Region(
                            id="room_current",
                            type="room",
                            bounds=(y-5, x-5, y+5, x+5),  # Estimate
                            explored=0.5
                        )
                elif current_char == '#':
                    # In a corridor
                    self.current_region_id = "corridor_current"
                    if "corridor_current" not in self.regions:
                        self.regions["corridor_current"] = Region(
                            id="corridor_current",
                            type="corridor",
                            bounds=(y-2, x-2, y+2, x+2),  # Estimate
                            explored=0.3
                        )
                        
        # Update current region features
        if self.current_region_id and self.current_region_id in self.regions:
            region = self.regions[self.current_region_id]
            region.has_stairs = self.stairs_pos is not None
            region.has_monsters = self.monsters_present
            region.has_items = self.items_visible
    
    def get_summary(self) -> Dict:
        """Get current world state summary"""
        return {
            "current_region": self.current_region_id,
            "regions_found": len(self.regions),
            "stairs_found": self.stairs_pos is not None,
            "threats": self.monsters_present,
            "obstacles": self.doors_present,
            "opportunities": self.items_visible,
            "status": self.player_status
        }

# ============================================================================
# HIERARCHICAL PLANNER WITH LLM INTEGRATION
# ============================================================================

class HierarchicalPlanner:
    """Enhanced planner with LLM integration"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.plan_stack: List[Plan] = []
        self.completed_plans: List[Plan] = []
        self.failed_plans: List[Plan] = []
        
    def get_current_plan(self) -> Optional[Plan]:
        """Get the lowest level active plan"""
        return self.plan_stack[-1] if self.plan_stack else None
        
    def generate_llm_prompt(self, world_state: Dict, current_plan: Optional[Plan]) -> str:
        """Generate hierarchical prompt for LLM"""
        prompt_parts = []
        
        # Current situation
        prompt_parts.append("=== NETHACK SITUATION ===")
        prompt_parts.append(f"Location: {world_state.get('current_region', 'unknown')}")
        prompt_parts.append(f"Regions explored: {world_state.get('regions_found', 0)}")
        prompt_parts.append(f"Stairs found: {world_state['stairs_found']}")
        prompt_parts.append(f"Threats present: {world_state['threats']}")
        prompt_parts.append(f"Items visible: {world_state['opportunities']}")
        
        # Player status
        status = world_state.get('status', {})
        if status.get('hp_low'):
            prompt_parts.append("WARNING: HP is low!")
        if status.get('hungry'):
            prompt_parts.append("WARNING: You are hungry!")
            
        # Current plans
        prompt_parts.append("\n=== ACTIVE PLANS ===")
        if self.plan_stack:
            for i, plan in enumerate(self.plan_stack):
                indent = "  " * i
                prompt_parts.append(f"{indent}[{plan.level.name}] {plan.goal} (steps: {plan.steps_taken})")
        else:
            prompt_parts.append("No active plans.")
            
        # Decision needed
        prompt_parts.append("\n=== DECISION NEEDED ===")
        
        if not current_plan:
            prompt_parts.append("Choose strategy:")
            prompt_parts.append("explore_and_descend")
            prompt_parts.append("rush_stairs") 
            prompt_parts.append("survive_first")
            prompt_parts.append("\nChoose one:")
            
        elif current_plan.level == PlanLevel.STRATEGIC:
            if world_state['stairs_found']:
                prompt_parts.append("Stairs found! Choose: navigate_to_stairs")
            else:
                prompt_parts.append("Choose tactic:")
                prompt_parts.append("explore_current_region")
                prompt_parts.append("search_for_stairs")
                prompt_parts.append("\nChoose one:")
            
        elif current_plan.level == PlanLevel.TACTICAL:
            if world_state['stairs_found']:
                prompt_parts.append("Choose: move_to_target")
            else:
                prompt_parts.append("Choose action:")
                prompt_parts.append("move_systematically")
                prompt_parts.append("search_area")
                prompt_parts.append("\nChoose one:")
            
        else:  # OPERATIONAL
            prompt_parts.append("Choose move:")
            prompt_parts.append("north, south, east, west, search")
            prompt_parts.append("\nChoose one:")
            
        return "\n".join(prompt_parts)
        
    def get_llm_decision(self, world_state: Dict) -> str:
        """Get hierarchical decision from LLM"""
        current_plan = self.get_current_plan()
        
        # Generate prompt
        prompt = self.generate_llm_prompt(world_state, current_plan)
        
        # Get LLM response
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate(messages)
        decision = response.completion.strip().lower()
        
        # Process decision based on current level
        if not current_plan:
            # Strategic level
            self.add_plan(PlanLevel.STRATEGIC, decision)
            return self.get_llm_decision(world_state)  # Recurse for tactical
            
        elif current_plan.level == PlanLevel.STRATEGIC:
            # Tactical level
            self.add_plan(PlanLevel.TACTICAL, decision)
            return self.get_llm_decision(world_state)  # Recurse for operational
            
        elif current_plan.level == PlanLevel.TACTICAL:
            # Operational level
            self.add_plan(PlanLevel.OPERATIONAL, decision)
            return self.get_llm_decision(world_state)  # Recurse for immediate
            
        else:
            # Immediate action
            return decision
            
    def add_plan(self, level: PlanLevel, goal: str):
        """Add a new plan to the stack"""
        # Remove any plans at same or lower level
        while self.plan_stack and self.plan_stack[-1].level.value <= level.value:
            old_plan = self.plan_stack.pop()
            old_plan.status = "interrupted"
            self.failed_plans.append(old_plan)
            
        plan = Plan(level=level, goal=goal)
        self.plan_stack.append(plan)
        
    def update_current_plan(self):
        """Update current plan progress"""
        if self.plan_stack:
            self.plan_stack[-1].steps_taken += 1
            
            # Check if plan should be abandoned
            current = self.plan_stack[-1]
            if current.steps_taken > current.max_steps:
                current.status = "failed"
                self.failed_plans.append(self.plan_stack.pop())

# ============================================================================
# SIMPLE PATHFINDING
# ============================================================================

class SimplePathfinder:
    """Simple pathfinding for immediate actions"""
    
    def __init__(self):
        self.last_positions = deque(maxlen=10)
        self.stuck_counter = 0
        
    def get_path_to_stairs(self, current_pos: Optional[Tuple[int, int]], 
                          stairs_pos: Optional[Tuple[int, int]]) -> str:
        """Get direction to move toward stairs"""
        if not current_pos or not stairs_pos:
            return "search"
            
        dy = stairs_pos[0] - current_pos[0]
        dx = stairs_pos[1] - current_pos[1]
        
        # Prefer cardinal directions
        if abs(dy) > abs(dx):
            return "north" if dy < 0 else "south"
        elif dx != 0:
            return "west" if dx < 0 else "east"
        else:
            return "wait"  # We're at the stairs
            
    def get_exploration_move(self, step_count: int) -> str:
        """Get systematic exploration movement"""
        # More exploratory pattern with less searching
        moves = ["east", "north", "west", "south", "northeast", "southeast", 
                 "southwest", "northwest", "east", "north", "search"]
        return moves[step_count % len(moves)]
        
    def detect_stuck(self, current_pos: Optional[Tuple[int, int]]) -> bool:
        """Detect if we're stuck in the same area"""
        if current_pos:
            self.last_positions.append(current_pos)
            
        if len(self.last_positions) >= 10:
            # Check if we're oscillating
            unique_positions = len(set(self.last_positions))
            if unique_positions <= 3:
                self.stuck_counter += 1
                return True
                
        self.stuck_counter = 0
        return False

# ============================================================================
# MAIN HIERARCHICAL DREAMCODER AGENT V2
# ============================================================================

class HierarchicalDreamCoderAgentV2(BaseAgent):
    """Enhanced hierarchical agent with all improvements"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Core components
        self.world_model = HierarchicalWorldModel()
        self.planner = HierarchicalPlanner(self.client)
        self.pathfinder = SimplePathfinder()
        
        # Memory and learning
        self.action_history = deque(maxlen=50)
        self.successful_patterns = defaultdict(list)
        self.learned_macros = {}
        
        # Performance tracking
        self.steps_since_progress = 0
        self.last_score = 0
        self.last_depth = 0
        
    def act(self, obs, prev_action=None):
        """Main action selection with hierarchical reasoning"""
        # Update systems
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        self.world_model.update_from_obs(obs)
        
        # Get world state
        world_state = self.world_model.get_summary()
        
        # Update planner
        self.planner.update_current_plan()
        
        # Check if we're making progress
        score = obs.get("score", 0)
        depth = obs.get("depth", 0)
        
        if score > self.last_score or depth > self.last_depth:
            self.steps_since_progress = 0
            self.last_score = score
            self.last_depth = depth
        else:
            self.steps_since_progress += 1
            
        # Get action
        if self.steps_since_progress > 15:  # Reduced threshold
            # We're stuck, try something different
            action = self._get_unstuck_action(world_state)
        else:
            # Normal hierarchical reasoning
            action = self._hierarchical_reasoning(world_state)
            
        # Avoid repetition
        action = self._avoid_repetition(action)
        
        # Record for learning
        self.action_history.append(action)
        
        return LLMResponse(
            model_id="hierarchical_dreamcoder_v2",
            completion=action,
            stop_reason="complete",
            input_tokens=0,
            output_tokens=0,
            reasoning=self._get_reasoning_summary()
        )
        
    def _hierarchical_reasoning(self, world_state: Dict) -> str:
        """Core hierarchical reasoning with LLM"""
        current_plan = self.planner.get_current_plan()
        
        # Special cases first
        if world_state['status'].get('hp_low'):
            return self._survival_action(world_state)
            
        if world_state['threats'] and not current_plan:
            return self._combat_action(world_state)
            
        # Check if we have stairs and should descend
        if (world_state['stairs_found'] and 
            self.world_model.player_pos and 
            self.world_model.stairs_pos and
            self.world_model.player_pos == self.world_model.stairs_pos):
            return "down"
            
        # Get action from planner
        action = self.planner.get_llm_decision(world_state)
        
        # Convert high-level actions to concrete moves
        if action in ["move_to_target", "navigate_to_stairs"]:
            if world_state['stairs_found']:
                return self.pathfinder.get_path_to_stairs(
                    self.world_model.player_pos,
                    self.world_model.stairs_pos
                )
            else:
                return "search"
                
        elif action in ["search_area", "search_pattern", "check_corners"]:
            return "search"
            
        elif action in ["move_systematically", "explore_current_region"]:
            return self.pathfinder.get_exploration_move(
                self.planner.get_current_plan().steps_taken
            )
            
        elif action == "open_path":
            if world_state['obstacles']:
                return "kick"
            else:
                return "search"
                
        # Return action if it's already concrete
        valid_actions = ["north", "south", "east", "west", "northeast", 
                        "southeast", "southwest", "northwest", "wait", 
                        "search", "open", "kick", "pickup", "down"]
        
        if action in valid_actions:
            return action
        else:
            # Fallback
            return "search"
            
    def _survival_action(self, world_state: Dict) -> str:
        """Emergency survival actions"""
        # Simple survival heuristics
        if world_state['threats']:
            # Run away
            return "southwest"  # Diagonal movement is often good for escape
        else:
            # Rest
            return "wait"
            
    def _combat_action(self, world_state: Dict) -> str:
        """Handle combat situations"""
        # Simple combat strategy
        if self.world_model.current_region_id and "corridor" in self.world_model.current_region_id:
            # In corridor, wait for monster
            return "wait"
        else:
            # In room, kick
            return "kick"
            
    def _get_unstuck_action(self, world_state: Dict) -> str:
        """Try to get unstuck when making no progress"""
        # Clear plans and try new strategy
        self.planner.plan_stack.clear()
        self.steps_since_progress = 0
        
        # Try different movement actions to explore
        import random
        movement_actions = ["north", "south", "east", "west", "northeast", "southeast", "southwest", "northwest"]
        return random.choice(movement_actions)
            
    def _avoid_repetition(self, action: str) -> str:
        """Avoid repeating the same action too many times"""
        if len(self.action_history) >= 5:
            recent_actions = list(self.action_history)[-5:]
            if all(a == action for a in recent_actions):
                # Too repetitive, try something else
                alternatives = ["north", "south", "east", "west", "search", "wait"]
                alternatives.remove(action) if action in alternatives else None
                import random
                return random.choice(alternatives)
        return action
        
    def _get_reasoning_summary(self) -> str:
        """Get summary of current reasoning"""
        parts = []
        
        # Current plans
        if self.planner.plan_stack:
            parts.append("Plans:")
            for plan in self.planner.plan_stack:
                parts.append(f"  [{plan.level.name}] {plan.goal}")
                
        # World state
        state = self.world_model.get_summary()
        parts.append(f"Location: {state['current_region']}")
        parts.append(f"Stairs: {'Found' if state['stairs_found'] else 'Not found'}")
        
        return " | ".join(parts)
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.world_model = HierarchicalWorldModel()
        self.planner = HierarchicalPlanner(self.client)
        self.pathfinder = SimplePathfinder()
        self.action_history.clear()
        self.steps_since_progress = 0
        self.last_score = 0
        self.last_depth = 0

# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def create_hierarchical_dreamcoder_v2(client_factory, prompt_builder):
    """Factory function for BALROG integration"""
    return HierarchicalDreamCoderAgentV2(client_factory, prompt_builder)