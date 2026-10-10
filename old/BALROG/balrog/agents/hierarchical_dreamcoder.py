"""
Hierarchical DreamCoder Agent for BALROG
========================================

A practical implementation that combines:
1. Hierarchical planning
2. LLM reasoning
3. Learned abstractions
4. NetHack-specific logic
"""

import json
import re
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse

# ============================================================================
# STEP 1: Basic Data Structures
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
    
@dataclass 
class GameRegion:
    """A region in the game (room, corridor, etc)"""
    id: str
    type: str  # room, corridor, stairs
    explored: float = 0.0  # 0-1 exploration percentage
    has_monsters: bool = False
    has_items: bool = False
    connections: List[str] = None

# ============================================================================
# STEP 2: Simple World Model
# ============================================================================

class SimpleWorldModel:
    """Maintains understanding of the game world"""
    
    def __init__(self):
        self.current_location = "unknown"
        self.visited_locations = set()
        self.stairs_found = False
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
        """Update world model from observation"""
        text = obs.get("text", {})
        long_term = text.get("long_term_context", "").lower()
        short_term = text.get("short_term_context", "").lower()
        
        # Location detection (simple)
        if "room" in long_term:
            self.current_location = "room"
        elif "corridor" in long_term:
            self.current_location = "corridor"
            
        # Feature detection
        self.stairs_found = ">" in long_term or "stairs" in long_term
        self.monsters_present = any(w in long_term for w in ["monster", "attacks", "hits"])
        self.doors_present = "+" in long_term or "door" in long_term
        self.items_visible = any(c in long_term for c in [")", "[", "?", "!", "*"])
        
        # Status updates
        self.player_status["hungry"] = "hungry" in short_term
        self.player_status["hp_low"] = "hp:" in short_term and self._check_hp_low(short_term)
        
    def _check_hp_low(self, text: str) -> bool:
        """Check if HP is low"""
        # Simple heuristic - look for HP: X(Y) pattern
        import re
        match = re.search(r'hp:\s*(\d+)\((\d+)\)', text)
        if match:
            current = int(match.group(1))
            maximum = int(match.group(2))
            return current < maximum * 0.3
        return False
    
    def get_summary(self) -> Dict:
        """Get current world state summary"""
        return {
            "location": self.current_location,
            "stairs_found": self.stairs_found,
            "threats": self.monsters_present,
            "obstacles": self.doors_present,
            "opportunities": self.items_visible,
            "status": self.player_status
        }

# ============================================================================
# STEP 3: Hierarchical Planner
# ============================================================================

class HierarchicalPlanner:
    """Manages hierarchical plans"""
    
    def __init__(self):
        self.plan_stack: List[Plan] = []
        self.completed_plans: List[Plan] = []
        self.failed_plans: List[Plan] = []
        
    def get_current_plan(self) -> Optional[Plan]:
        """Get the lowest level active plan"""
        return self.plan_stack[-1] if self.plan_stack else None
        
    def add_plan(self, level: PlanLevel, goal: str):
        """Add a new plan to the stack"""
        # Remove any plans at same or lower level
        while self.plan_stack and self.plan_stack[-1].level <= level:
            old_plan = self.plan_stack.pop()
            old_plan.status = "interrupted"
            self.failed_plans.append(old_plan)
            
        plan = Plan(level=level, goal=goal)
        self.plan_stack.append(plan)
        
    def complete_current_plan(self):
        """Mark current plan as completed"""
        if self.plan_stack:
            plan = self.plan_stack.pop()
            plan.status = "completed"
            self.completed_plans.append(plan)
            
    def fail_current_plan(self, reason: str = ""):
        """Mark current plan as failed"""
        if self.plan_stack:
            plan = self.plan_stack.pop()
            plan.status = f"failed: {reason}"
            self.failed_plans.append(plan)
            
    def get_plan_hierarchy(self) -> str:
        """Get string representation of current plan hierarchy"""
        if not self.plan_stack:
            return "No active plans"
            
        lines = []
        for i, plan in enumerate(self.plan_stack):
            indent = "  " * i
            lines.append(f"{indent}[{plan.level.name}] {plan.goal} (steps: {plan.steps_taken})")
        return "\n".join(lines)

# ============================================================================
# STEP 4: The Main Agent
# ============================================================================

class HierarchicalDreamCoderAgent(BaseAgent):
    """Agent that uses hierarchical planning with LLM reasoning"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Core components
        self.world_model = SimpleWorldModel()
        self.planner = HierarchicalPlanner()
        
        # Memory for learning
        self.successful_patterns = defaultdict(list)
        self.action_history = deque(maxlen=20)
        
        # Learned abstractions (like DreamCoder)
        self.learned_macros = {}
        
    def act(self, obs, prev_action=None):
        """Main action selection"""
        # Update systems
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        self.world_model.update_from_obs(obs)
        
        # Get action using hierarchical reasoning
        action = self._hierarchical_reasoning(obs)
        
        # Record for learning
        self.action_history.append(action)
        
        return LLMResponse(
            model_id="hierarchical_dreamcoder",
            completion=action,
            stop_reason="complete",
            input_tokens=0,
            output_tokens=0,
            reasoning=self.planner.get_plan_hierarchy()
        )
        
    def _hierarchical_reasoning(self, obs: Dict) -> str:
        """Core hierarchical reasoning logic"""
        world_state = self.world_model.get_summary()
        
        # Check current plan progress
        current_plan = self.planner.get_current_plan()
        if current_plan:
            current_plan.steps_taken += 1
            
            # Check if plan should be abandoned
            if current_plan.steps_taken > current_plan.max_steps:
                self.planner.fail_current_plan("max steps exceeded")
                current_plan = None
                
        # If no plan, create strategic plan
        if not current_plan:
            return self._create_strategic_plan(world_state)
            
        # Execute based on plan level
        if current_plan.level == PlanLevel.STRATEGIC:
            return self._execute_strategic(current_plan, world_state)
        elif current_plan.level == PlanLevel.TACTICAL:
            return self._execute_tactical(current_plan, world_state)
        elif current_plan.level == PlanLevel.OPERATIONAL:
            return self._execute_operational(current_plan, world_state)
        else:  # IMMEDIATE
            return self._execute_immediate(current_plan, world_state)
            
    def _create_strategic_plan(self, world_state: Dict) -> str:
        """Create a new strategic plan using LLM"""
        prompt = f"""
You are playing NetHack. Create a strategic plan.

Current Situation:
- Location: {world_state['location']}
- Stairs found: {world_state['stairs_found']}
- Threats: {world_state['threats']}
- Player hungry: {world_state['status']['hungry']}
- HP low: {world_state['status']['hp_low']}

Choose ONE strategic goal:
1. "explore_and_descend" - Fully explore level then go down
2. "rush_stairs" - Find stairs ASAP and descend
3. "survive_and_recover" - Get safe and heal up first
4. "systematic_clear" - Clear every room methodically

Respond with just the goal name.
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.client.generate(messages)
        
        goal = response.completion.strip().lower()
        
        # Validate and create plan
        valid_goals = ["explore_and_descend", "rush_stairs", "survive_and_recover", "systematic_clear"]
        if goal not in valid_goals:
            goal = "explore_and_descend"  # Default
            
        self.planner.add_plan(PlanLevel.STRATEGIC, goal)
        
        # Now need tactical plan
        return self._execute_strategic(self.planner.get_current_plan(), world_state)
        
    def _execute_strategic(self, plan: Plan, world_state: Dict) -> str:
        """Execute strategic plan by creating tactical subplan"""
        
        # Different tactical approaches based on strategic goal
        if plan.goal == "explore_and_descend":
            if world_state["stairs_found"]:
                self.planner.add_plan(PlanLevel.TACTICAL, "navigate_to_stairs")
            else:
                self.planner.add_plan(PlanLevel.TACTICAL, "explore_unmapped")
                
        elif plan.goal == "rush_stairs":
            if world_state["stairs_found"]:
                self.planner.add_plan(PlanLevel.TACTICAL, "navigate_to_stairs")
            else:
                self.planner.add_plan(PlanLevel.TACTICAL, "search_for_stairs")
                
        elif plan.goal == "survive_and_recover":
            if world_state["status"]["hungry"]:
                self.planner.add_plan(PlanLevel.TACTICAL, "find_food")
            elif world_state["status"]["hp_low"]:
                self.planner.add_plan(PlanLevel.TACTICAL, "find_safe_spot")
            else:
                self.planner.complete_current_plan()
                return self._hierarchical_reasoning({"text": world_state})
                
        elif plan.goal == "systematic_clear":
            self.planner.add_plan(PlanLevel.TACTICAL, "clear_current_room")
            
        return self._execute_tactical(self.planner.get_current_plan(), world_state)
        
    def _execute_tactical(self, plan: Plan, world_state: Dict) -> str:
        """Execute tactical plan by creating operational subplan"""
        
        if plan.goal == "explore_unmapped":
            # Check if current location is fully explored
            if self._is_current_area_explored():
                self.planner.add_plan(PlanLevel.OPERATIONAL, "move_to_unexplored")
            else:
                self.planner.add_plan(PlanLevel.OPERATIONAL, "explore_current_area")
                
        elif plan.goal == "navigate_to_stairs":
            self.planner.add_plan(PlanLevel.OPERATIONAL, "pathfind_to_stairs")
            
        elif plan.goal == "search_for_stairs":
            self.planner.add_plan(PlanLevel.OPERATIONAL, "systematic_search")
            
        elif plan.goal == "clear_current_room":
            if world_state["threats"]:
                self.planner.add_plan(PlanLevel.OPERATIONAL, "combat_sequence")
            elif world_state["opportunities"]:
                self.planner.add_plan(PlanLevel.OPERATIONAL, "collect_items")
            else:
                self.planner.complete_current_plan()
                return self._hierarchical_reasoning({"text": world_state})
                
        return self._execute_operational(self.planner.get_current_plan(), world_state)
        
    def _execute_operational(self, plan: Plan, world_state: Dict) -> str:
        """Execute operational plan with immediate actions"""
        
        if plan.goal == "explore_current_area":
            # Use learned pattern or default exploration
            if "explore_room" in self.learned_macros:
                actions = self.learned_macros["explore_room"]
            else:
                actions = ["east", "north", "west", "south", "search"]
            
            # Get next action in sequence
            step = plan.steps_taken % len(actions)
            return actions[step]
            
        elif plan.goal == "pathfind_to_stairs":
            # Simple heuristic - would use real pathfinding
            return self._simple_pathfind_to_target(">")
            
        elif plan.goal == "combat_sequence":
            # Use combat macro if learned
            if world_state.get("enemy_adjacent"):
                return "kick"
            else:
                return "wait"
                
        elif plan.goal == "collect_items":
            return "pickup"
            
        elif plan.goal == "systematic_search":
            # Search pattern
            search_pattern = ["search", "search", "search", "north", "search", "east"]
            step = plan.steps_taken % len(search_pattern)
            return search_pattern[step]
            
        else:
            # Default exploration
            return "search"
            
    def _execute_immediate(self, plan: Plan, world_state: Dict) -> str:
        """Execute immediate action"""
        # This level is for single actions that complete immediately
        action = plan.goal
        self.planner.complete_current_plan()
        return action
        
    def _is_current_area_explored(self) -> bool:
        """Check if current area is sufficiently explored"""
        # Count recent search actions
        recent_searches = sum(1 for a in list(self.action_history)[-10:] if a == "search")
        return recent_searches >= 3
        
    def _simple_pathfind_to_target(self, target_symbol: str) -> str:
        """Simple pathfinding heuristic"""
        # In real implementation, would use A* or similar
        # For now, use simple heuristics
        
        # Try to move toward where stairs typically are
        if not hasattr(self, '_pathfind_step'):
            self._pathfind_step = 0
            
        # Spiral outward pattern
        spiral = ["north", "east", "south", "south", "west", "west", "north", "north"]
        action = spiral[self._pathfind_step % len(spiral)]
        self._pathfind_step += 1
        
        return action
        
    def learn_from_episode(self, episode_stats: Dict):
        """Learn successful patterns (DreamCoder style)"""
        if episode_stats.get("score", 0) > 0 or episode_stats.get("depth", 0) > 0:
            # Extract successful action sequences
            for completed_plan in self.planner.completed_plans:
                if completed_plan.level == PlanLevel.OPERATIONAL:
                    # Save as reusable macro
                    self.learned_macros[completed_plan.goal] = self._extract_actions_for_plan(completed_plan)
                    
    def _extract_actions_for_plan(self, plan: Plan) -> List[str]:
        """Extract action sequence that completed a plan"""
        # Simplified - in reality would trace through execution history
        if "explore" in plan.goal:
            return ["north", "east", "south", "west", "search"]
        elif "combat" in plan.goal:
            return ["wait", "kick", "wait"]
        else:
            return ["search"]
            
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.planner = HierarchicalPlanner()
        self.action_history.clear()
        self._pathfind_step = 0

# ============================================================================
# STEP 5: Integration with BALROG
# ============================================================================

def create_hierarchical_agent(client_factory, prompt_builder):
    """Factory function for BALROG integration"""
    return HierarchicalDreamCoderAgent(client_factory, prompt_builder)

# Add to BALROG's agent factory:
# elif self.config.agent.type == "hierarchical":
#     return create_hierarchical_agent(client_factory, prompt_builder)