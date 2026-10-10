"""
LLM-Driven Hierarchical Execution for NetHack
============================================

System to help LLMs understand and execute hierarchical plans
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class PlanLevel(Enum):
    STRATEGIC = "strategic"      # "Clear the dungeon"
    TACTICAL = "tactical"        # "Explore eastern rooms"
    OPERATIONAL = "operational"  # "Navigate to Room 3"
    IMMEDIATE = "immediate"      # "Move north"

@dataclass
class HierarchicalPlan:
    """A plan at a specific level of abstraction"""
    level: PlanLevel
    goal: str
    context: Dict
    subplans: List['HierarchicalPlan'] = None
    status: str = "pending"  # pending, active, completed, failed
    
    def to_prompt(self) -> str:
        """Convert plan to LLM-understandable format"""
        return f"[{self.level.value}] Goal: {self.goal} (Status: {self.status})"

class LLMHierarchicalExecutor:
    """Helps LLM execute hierarchical plans"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.plan_stack: List[HierarchicalPlan] = []
        self.world_model = WorldModel()
        self.execution_history = []
        
    def generate_hierarchical_prompt(self, obs: Dict) -> str:
        """Generate a prompt that helps LLM understand the hierarchical situation"""
        
        # Update world model
        self.world_model.update(obs)
        
        prompt_parts = []
        
        # 1. Current Situation Summary
        prompt_parts.append("=== CURRENT SITUATION ===")
        prompt_parts.append(self.world_model.get_summary())
        
        # 2. Active Plan Hierarchy
        prompt_parts.append("\n=== ACTIVE PLANS ===")
        if self.plan_stack:
            for i, plan in enumerate(self.plan_stack):
                indent = "  " * i
                prompt_parts.append(f"{indent}{plan.to_prompt()}")
        else:
            prompt_parts.append("No active plans. Need to create strategic plan.")
        
        # 3. Decision Context
        prompt_parts.append("\n=== DECISION CONTEXT ===")
        prompt_parts.append(self._get_decision_context())
        
        # 4. Available Actions at Current Level
        prompt_parts.append("\n=== AVAILABLE ACTIONS ===")
        prompt_parts.append(self._get_available_actions())
        
        # 5. Reasoning Template
        prompt_parts.append("\n=== YOUR TASK ===")
        prompt_parts.append(self._get_reasoning_template())
        
        return "\n".join(prompt_parts)
    
    def _get_decision_context(self) -> str:
        """Provide context for current decision"""
        if not self.plan_stack:
            return """You need to create a STRATEGIC plan. Consider:
- What is the overall goal? (reach deeper levels, get stronger, find specific items)
- What major phases are needed? (explore level, clear monsters, find stairs, descend)
- What obstacles might prevent success? (locked doors, strong monsters, maze-like layout)"""
        
        current_plan = self.plan_stack[-1]
        
        if current_plan.level == PlanLevel.STRATEGIC:
            return """You have a strategic plan. Now create TACTICAL subplans:
- How should you explore? (systematic room-by-room, follow corridors, spiral out)
- What order to tackle objectives? (clear visible threats first, explore edges, check center)
- How to handle resources? (save items for emergencies, use consumables liberally)"""
            
        elif current_plan.level == PlanLevel.TACTICAL:
            return """You have a tactical plan. Now create OPERATIONAL steps:
- Which specific room/area to go to next?
- What path to take? (through cleared areas, avoid monsters, shortest route)
- What preparations needed? (equip weapons, eat food, check inventory)"""
            
        elif current_plan.level == PlanLevel.OPERATIONAL:
            return """You have an operational plan. Choose IMMEDIATE action:
- What specific movement or action to take?
- Is the path clear? (check for obstacles, monsters, traps)
- Should you reconsider? (new threat appeared, better option available)"""
            
        else:  # IMMEDIATE
            return "Execute the immediate action or replan if blocked."
    
    def _get_available_actions(self) -> str:
        """Get actions available at current planning level"""
        if not self.plan_stack:
            return """Strategic Options:
1. "PLAN: Clear dungeon systematically" - Explore fully, then descend
2. "PLAN: Rush to stairs" - Find stairs ASAP, minimal exploration  
3. "PLAN: Power up first" - Find equipment/experience before descending
4. "PLAN: Explore and map" - Focus on understanding layout"""
        
        current_plan = self.plan_stack[-1]
        current_level = current_plan.level
        
        if current_level == PlanLevel.STRATEGIC:
            options = self._get_tactical_options()
        elif current_level == PlanLevel.TACTICAL:
            options = self._get_operational_options()
        elif current_level == PlanLevel.OPERATIONAL:
            options = self._get_immediate_options()
        else:
            options = self._get_action_options()
            
        return options
    
    def _get_tactical_options(self) -> str:
        """Get tactical-level options based on world model"""
        options = []
        
        if self.world_model.unexplored_regions:
            options.append("SUBPLAN: Explore unmapped areas")
        if self.world_model.visible_items:
            options.append("SUBPLAN: Collect visible items")
        if self.world_model.monsters_present:
            options.append("SUBPLAN: Clear threats")
        if self.world_model.doors_blocking:
            options.append("SUBPLAN: Open pathway")
        if self.world_model.stairs_known:
            options.append("SUBPLAN: Navigate to stairs")
        else:
            options.append("SUBPLAN: Search for stairs")
            
        return "\n".join(options)
    
    def _get_operational_options(self) -> str:
        """Get operational-level options"""
        region_map = self.world_model.get_region_map()
        
        options = []
        for region_id, region_info in region_map.items():
            if not region_info['explored']:
                options.append(f"GOTO: {region_id} - {region_info['description']}")
        
        if self.world_model.current_region:
            options.append(f"EXPLORE: Current {self.world_model.current_region['type']}")
            
        options.append("SEARCH: Look for hidden features")
        options.append("BACKTRACK: Return to previous area")
        
        return "\n".join(options)
    
    def _get_immediate_options(self) -> str:
        """Get immediate action options"""
        return """Movement: north, south, east, west, northeast, southeast, southwest, northwest
Interaction: open, close, kick, search, wait
Combat: attack <direction>
Items: pickup, drop, eat, wear, wield
Special: pray, down (stairs), up (stairs)"""
    
    def _get_reasoning_template(self) -> str:
        """Template for LLM reasoning"""
        return """Think step by step:

1. What is my current highest-level goal?
2. What progress have I made toward it?
3. What obstacles are preventing progress?
4. At what level should I plan/act now?
5. What is the best action at that level?

Respond with ONE of:
- PLAN: <strategic goal> (if no active plan)
- SUBPLAN: <tactical goal> (if executing strategic plan)
- GOTO: <location> (if executing tactical plan)
- ACTION: <specific action> (if executing operational plan)
- REPLAN: <reason> (if current plan is failing)

Your response:"""

    def parse_llm_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response into action type and content"""
        response = response.strip()
        
        if response.startswith("PLAN:"):
            return ("plan", response[5:].strip())
        elif response.startswith("SUBPLAN:"):
            return ("subplan", response[8:].strip())
        elif response.startswith("GOTO:"):
            return ("goto", response[5:].strip())
        elif response.startswith("ACTION:"):
            return ("action", response[7:].strip())
        elif response.startswith("REPLAN:"):
            return ("replan", response[7:].strip())
        else:
            # Try to extract action directly
            return ("action", response.lower())
    
    def execute_response(self, response_type: str, content: str, obs: Dict) -> str:
        """Execute the LLM's response and return concrete action"""
        
        if response_type == "plan":
            # Create new strategic plan
            new_plan = HierarchicalPlan(
                level=PlanLevel.STRATEGIC,
                goal=content,
                context={"start_state": self.world_model.get_summary()},
                status="active"
            )
            self.plan_stack = [new_plan]
            # Recurse to get tactical plan
            return self.get_next_action(obs)
            
        elif response_type == "subplan":
            # Add tactical subplan
            tactical_plan = HierarchicalPlan(
                level=PlanLevel.TACTICAL,
                goal=content,
                context={"parent": self.plan_stack[-1].goal},
                status="active"
            )
            self.plan_stack.append(tactical_plan)
            # Recurse to get operational plan
            return self.get_next_action(obs)
            
        elif response_type == "goto":
            # Add operational plan
            operational_plan = HierarchicalPlan(
                level=PlanLevel.OPERATIONAL,
                goal=f"Navigate to {content}",
                context={"target": content},
                status="active"
            )
            self.plan_stack.append(operational_plan)
            # Use pathfinding to get immediate action
            return self._pathfind_to(content)
            
        elif response_type == "action":
            # Direct action - validate and return
            return self._validate_action(content)
            
        elif response_type == "replan":
            # Pop failed plan and try again
            if self.plan_stack:
                failed_plan = self.plan_stack.pop()
                failed_plan.status = "failed"
                self.execution_history.append({
                    "plan": failed_plan,
                    "failure_reason": content
                })
            return self.get_next_action(obs)
            
        else:
            return "wait"
    
    def get_next_action(self, obs: Dict) -> str:
        """Main entry point - get next action from LLM"""
        # Generate hierarchical prompt
        prompt = self.generate_hierarchical_prompt(obs)
        
        # Get LLM response
        messages = [{"role": "user", "content": prompt}]
        llm_response = self.llm_client.generate(messages)
        
        # Parse response
        response_type, content = self.parse_llm_response(llm_response.completion)
        
        # Execute response to get concrete action
        action = self.execute_response(response_type, content, obs)
        
        # Record execution
        self.execution_history.append({
            "observation": obs,
            "prompt": prompt,
            "llm_response": llm_response.completion,
            "parsed": (response_type, content),
            "action": action
        })
        
        return action
    
    def _pathfind_to(self, target: str) -> str:
        """Use pathfinding to navigate to target"""
        # This would use the HPA* implementation
        # For now, return a simple heuristic
        if "room" in target.lower():
            return "north"  # Placeholder
        elif "corridor" in target.lower():
            return "east"   # Placeholder
        else:
            return "search"
            
    def _validate_action(self, action: str) -> str:
        """Validate and normalize action"""
        valid_actions = [
            "north", "south", "east", "west",
            "northeast", "southeast", "southwest", "northwest",
            "wait", "search", "open", "close", "kick",
            "pickup", "drop", "eat", "wear", "wield",
            "up", "down", "pray"
        ]
        
        action = action.lower().strip()
        if action in valid_actions:
            return action
            
        # Try to extract valid action from response
        for valid in valid_actions:
            if valid in action:
                return valid
                
        return "wait"  # Safe default

class WorldModel:
    """Maintains understanding of game state"""
    
    def __init__(self):
        self.regions = {}
        self.current_region = None
        self.unexplored_regions = []
        self.visible_items = []
        self.monsters_present = False
        self.doors_blocking = False
        self.stairs_known = False
        
    def update(self, obs: Dict):
        """Update world model from observation"""
        # Extract information from observation
        text = obs.get("text", {})
        long_term = text.get("long_term_context", "")
        
        # Update various aspects
        self.monsters_present = any(word in long_term.lower() 
                                  for word in ["monster", "attacks", "hostile"])
        self.doors_blocking = "closed door" in long_term.lower()
        self.stairs_known = ">" in long_term or "stairs" in long_term.lower()
        
        # Update regions (simplified)
        if "room" in long_term.lower():
            self.current_region = {"type": "room", "id": "current_room"}
        elif "corridor" in long_term.lower():
            self.current_region = {"type": "corridor", "id": "current_corridor"}
            
    def get_summary(self) -> str:
        """Get human-readable summary"""
        summary = []
        
        if self.current_region:
            summary.append(f"Location: {self.current_region['type']}")
        if self.monsters_present:
            summary.append("Threats: Monsters present")
        if self.doors_blocking:
            summary.append("Obstacles: Closed doors")
        if self.stairs_known:
            summary.append("Progress: Stairs located")
        else:
            summary.append("Progress: Stairs not found")
            
        return "\n".join(summary)
    
    def get_region_map(self) -> Dict:
        """Get simplified region map"""
        # Placeholder - would use actual region detection
        return {
            "room_north": {"explored": False, "description": "Northern room"},
            "room_east": {"explored": False, "description": "Eastern room"},
            "corridor_main": {"explored": True, "description": "Main corridor"}
        }

# ============================================================================
# Example LLM Prompts and Responses
# ============================================================================

EXAMPLE_PROMPTS = """
=== EXAMPLE 1: Starting a new level ===

PROMPT:
=== CURRENT SITUATION ===
Location: room
Progress: Stairs not found

=== ACTIVE PLANS ===
No active plans. Need to create strategic plan.

=== DECISION CONTEXT ===
You need to create a STRATEGIC plan. Consider:
- What is the overall goal? (reach deeper levels, get stronger, find specific items)
- What major phases are needed? (explore level, clear monsters, find stairs, descend)
- What obstacles might prevent success? (locked doors, strong monsters, maze-like layout)

=== YOUR TASK ===
Think step by step...

LLM RESPONSE:
Looking at the situation, I need to:
1. Current goal: Progress deeper into the dungeon
2. No progress yet on this level
3. Unknown layout and stairs location
4. Need strategic plan

PLAN: Clear dungeon systematically

---

=== EXAMPLE 2: Executing strategic plan ===

PROMPT:
=== CURRENT SITUATION ===
Location: room
Progress: Stairs not found

=== ACTIVE PLANS ===
[strategic] Goal: Clear dungeon systematically (Status: active)

=== DECISION CONTEXT ===
You have a strategic plan. Now create TACTICAL subplans...

=== AVAILABLE ACTIONS ===
SUBPLAN: Explore unmapped areas
SUBPLAN: Search for stairs

LLM RESPONSE:
Following my strategic plan to clear systematically:
1. Current goal: Clear dungeon systematically
2. Just starting, no areas explored yet
3. Need to map out the level first
4. Should explore before focused searching

SUBPLAN: Explore unmapped areas

---

=== EXAMPLE 3: Low-level navigation ===

PROMPT:
=== CURRENT SITUATION ===
Location: room
Threats: Monsters present
Progress: Stairs not found

=== ACTIVE PLANS ===
[strategic] Goal: Clear dungeon systematically (Status: active)
  [tactical] Goal: Explore unmapped areas (Status: active)
    [operational] Goal: Navigate to room_north (Status: active)

=== AVAILABLE ACTIONS ===
Movement: north, south, east, west...
Combat: attack <direction>

LLM RESPONSE:
I need to reach the northern room, but there are monsters:
1. Goal: Navigate to room_north
2. Monster blocking direct path
3. Should clear threat first for safety
4. Need immediate combat action

ACTION: attack north
"""

# ============================================================================
# Integration with DreamCoder
# ============================================================================

class HierarchicalDreamCoderAgent:
    """DreamCoder agent that uses hierarchical LLM reasoning"""
    
    def __init__(self, client_factory, prompt_builder):
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.executor = LLMHierarchicalExecutor(self.client)
        self.learned_patterns = {}  # DreamCoder's learned abstractions
        
    def act(self, obs, prev_action=None):
        """Get action using hierarchical reasoning"""
        # Update prompt builder
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        # Get hierarchical action
        action = self.executor.get_next_action(obs)
        
        # Learn from execution
        self._update_learned_patterns()
        
        return action
    
    def _update_learned_patterns(self):
        """Update DreamCoder patterns from successful plan executions"""
        # Check completed plans in execution history
        for entry in self.executor.execution_history[-10:]:
            if "plan" in entry and entry["plan"].status == "completed":
                # Extract successful pattern
                pattern = self._extract_pattern(entry["plan"])
                if pattern:
                    pattern_key = tuple(pattern)
                    if pattern_key not in self.learned_patterns:
                        self.learned_patterns[pattern_key] = {
                            "count": 0,
                            "success_rate": 0.0,
                            "context": entry["plan"].context
                        }
                    self.learned_patterns[pattern_key]["count"] += 1

print("LLM Hierarchical Execution System Created!")
print("\nKey Features:")
print("1. Structured prompts for hierarchical reasoning")
print("2. Plan stack management") 
print("3. Context-aware action selection")
print("4. Integration with pathfinding")
print("5. Learning from successful executions")