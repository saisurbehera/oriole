"""
Fundamental Design for NetHack Sub-Programs
==========================================

Sub-programs should be more than just action sequences. They should be:
1. Goal-oriented behaviors with termination conditions
2. Context-aware with preconditions and postconditions
3. Hierarchical - able to call other sub-programs
4. Parameterized - able to adapt to specific situations
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Any
from enum import Enum

class SubProgramStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class SubProgramResult:
    """Result of executing a sub-program"""
    status: SubProgramStatus
    actions_taken: List[str]
    final_state: Dict[str, Any]
    failure_reason: Optional[str] = None

class SubProgram:
    """Base class for all sub-programs"""
    
    def __init__(self, name: str):
        self.name = name
        self.steps_executed = 0
        self.max_steps = 50  # Prevent infinite loops
        
    def can_execute(self, state: Dict) -> bool:
        """Check if preconditions are met"""
        raise NotImplementedError
        
    def is_complete(self, state: Dict) -> bool:
        """Check if goal is achieved"""
        raise NotImplementedError
        
    def get_next_action(self, state: Dict) -> str:
        """Get next action to execute"""
        raise NotImplementedError
        
    def update_state(self, state: Dict, action: str, result: Dict) -> Dict:
        """Update internal state based on action result"""
        return state
        
    def execute(self, initial_state: Dict, env) -> SubProgramResult:
        """Execute the sub-program to completion"""
        state = initial_state.copy()
        actions = []
        
        while self.steps_executed < self.max_steps:
            # Check completion
            if self.is_complete(state):
                return SubProgramResult(SubProgramStatus.SUCCESS, actions, state)
                
            # Get next action
            action = self.get_next_action(state)
            if not action:
                return SubProgramResult(SubProgramStatus.BLOCKED, actions, state, "No valid action")
                
            # Execute action
            obs, reward, done, info = env.step(action)
            actions.append(action)
            self.steps_executed += 1
            
            # Update state
            state = self.update_state(state, action, {"obs": obs, "reward": reward})
            
            # Check for failure
            if done and not self.is_complete(state):
                return SubProgramResult(SubProgramStatus.FAILED, actions, state, "Episode ended")
                
        return SubProgramResult(SubProgramStatus.FAILED, actions, state, "Max steps exceeded")

# ============================================================================
# FUNDAMENTAL SUB-PROGRAMS FOR NETHACK
# ============================================================================

class NavigateTo(SubProgram):
    """Navigate to a specific target (stairs, door, item, etc)"""
    
    def __init__(self, target_symbol: str, target_description: str):
        super().__init__(f"navigate_to_{target_description}")
        self.target_symbol = target_symbol
        self.target_description = target_description
        self.path = None
        self.stuck_counter = 0
        
    def can_execute(self, state: Dict) -> bool:
        # Can execute if target is visible
        return self.target_symbol in state.get("visible_symbols", "")
        
    def is_complete(self, state: Dict) -> bool:
        # Complete when adjacent to target
        return state.get("adjacent_to_target", False)
        
    def get_next_action(self, state: Dict) -> str:
        # Use pathfinding or heuristic
        if not self.path:
            self.path = self._compute_path(state)
            
        if self.path:
            return self.path.pop(0)
        else:
            # Explore if can't find path
            return self._explore_action(state)
            
    def _compute_path(self, state: Dict) -> List[str]:
        """Compute path to target using A* or similar"""
        # Simplified - in reality would use proper pathfinding
        target_dir = state.get("target_direction", {}).get(self.target_symbol)
        if target_dir:
            return [target_dir]
        return []
        
    def _explore_action(self, state: Dict) -> str:
        """Explore to find target"""
        # Rotate through directions
        directions = ["north", "east", "south", "west"]
        last_dir = state.get("last_direction", "north")
        idx = directions.index(last_dir)
        return directions[(idx + 1) % 4]

class OpenDoor(SubProgram):
    """Open a door - try multiple strategies"""
    
    def __init__(self):
        super().__init__("open_door")
        self.strategies = ["open", "kick", "kick", "apply"]  # Try different methods
        self.current_strategy = 0
        
    def can_execute(self, state: Dict) -> bool:
        return state.get("door_present", False)
        
    def is_complete(self, state: Dict) -> bool:
        return not state.get("door_present", False) or state.get("door_open", False)
        
    def get_next_action(self, state: Dict) -> str:
        if self.current_strategy < len(self.strategies):
            action = self.strategies[self.current_strategy]
            self.current_strategy += 1
            return action
        return None

class ClearRoom(SubProgram):
    """Clear a room - explore, fight monsters, collect items"""
    
    def __init__(self):
        super().__init__("clear_room")
        self.sub_programs = []
        self.current_sub = None
        
    def can_execute(self, state: Dict) -> bool:
        return state.get("location_type") == "room"
        
    def is_complete(self, state: Dict) -> bool:
        # Room is clear when explored and no monsters
        return (state.get("room_explored", False) and 
                not state.get("monsters_present", False))
        
    def get_next_action(self, state: Dict) -> str:
        # Prioritize sub-tasks
        if state.get("monsters_present"):
            return self._combat_action(state)
        elif state.get("items_visible"):
            return self._collect_items(state)
        else:
            return self._explore_room(state)
            
    def _combat_action(self, state: Dict) -> str:
        """Handle combat - use positioning and tactics"""
        if state.get("monster_adjacent"):
            return "kick"  # or wait for it to attack first
        else:
            # Move to better position
            return "wait"
            
    def _collect_items(self, state: Dict) -> str:
        """Move to and pickup items"""
        item_dir = state.get("nearest_item_direction")
        if item_dir:
            return item_dir
        return "pickup"
        
    def _explore_room(self, state: Dict) -> str:
        """Systematically explore room"""
        # Spiral outward pattern
        pattern = ["east", "east", "north", "north", "west", "west", "west", 
                   "south", "south", "south", "east", "east", "east"]
        step = state.get("exploration_step", 0) % len(pattern)
        return pattern[step]

class FindStairs(SubProgram):
    """Find stairs - search systematically"""
    
    def __init__(self):
        super().__init__("find_stairs")
        self.search_pattern = self._generate_search_pattern()
        self.search_index = 0
        
    def can_execute(self, state: Dict) -> bool:
        # Execute when no stairs visible and area mostly explored
        return (not state.get("stairs_visible", False) and 
                state.get("explored_percent", 0) > 0.7)
        
    def is_complete(self, state: Dict) -> bool:
        return state.get("stairs_visible", False)
        
    def get_next_action(self, state: Dict) -> str:
        # Systematic search pattern with movement
        if self.search_index < len(self.search_pattern):
            action = self.search_pattern[self.search_index]
            self.search_index += 1
            return action
        else:
            # Reset pattern
            self.search_index = 0
            return "search"
            
    def _generate_search_pattern(self) -> List[str]:
        """Generate systematic search pattern"""
        # Search in corners and along walls where stairs often hide
        return [
            "search", "search", "search",  # Search current area
            "northeast", "search", "search",  # Check corners
            "southeast", "search", "search",
            "southwest", "search", "search",
            "northwest", "search", "search",
            "north", "search", "east", "search",  # Check walls
            "south", "search", "west", "search"
        ]

class ManageInventory(SubProgram):
    """Manage inventory - equip items, eat food, etc."""
    
    def __init__(self):
        super().__init__("manage_inventory")
        self.actions_queue = []
        
    def can_execute(self, state: Dict) -> bool:
        # Execute when have unequipped items or need food
        return (state.get("has_unequipped_armor", False) or
                state.get("has_unequipped_weapon", False) or
                (state.get("hunger_level", "normal") in ["hungry", "weak"]))
        
    def is_complete(self, state: Dict) -> bool:
        return len(self.actions_queue) == 0
        
    def get_next_action(self, state: Dict) -> str:
        if not self.actions_queue:
            self._plan_actions(state)
            
        if self.actions_queue:
            return self.actions_queue.pop(0)
        return None
        
    def _plan_actions(self, state: Dict):
        """Plan inventory management actions"""
        inventory = state.get("inventory", {})
        
        # Eat if hungry
        if state.get("hunger_level") in ["hungry", "weak"]:
            food_slot = self._find_item_slot(inventory, "food")
            if food_slot:
                self.actions_queue.extend(["eat", food_slot])
                
        # Wear armor
        if state.get("has_unequipped_armor"):
            armor_slot = self._find_item_slot(inventory, "armor")
            if armor_slot:
                self.actions_queue.extend(["wear", armor_slot])
                
        # Wield weapon
        if state.get("has_unequipped_weapon"):
            weapon_slot = self._find_item_slot(inventory, "weapon")
            if weapon_slot:
                self.actions_queue.extend(["wield", weapon_slot])
                
    def _find_item_slot(self, inventory: Dict, item_type: str) -> Optional[str]:
        """Find inventory slot for item type"""
        for slot, item in inventory.items():
            if item_type in item.lower():
                return slot
        return None

class SurviveCombat(SubProgram):
    """Survive combat using tactics"""
    
    def __init__(self):
        super().__init__("survive_combat")
        self.retreat_threshold = 0.3  # HP percentage to retreat
        
    def can_execute(self, state: Dict) -> bool:
        return state.get("in_combat", False)
        
    def is_complete(self, state: Dict) -> bool:
        return not state.get("in_combat", False)
        
    def get_next_action(self, state: Dict) -> str:
        hp_percent = state.get("hp_percent", 1.0)
        
        if hp_percent < self.retreat_threshold:
            return self._retreat_action(state)
        else:
            return self._combat_action(state)
            
    def _retreat_action(self, state: Dict) -> str:
        """Retreat from combat"""
        # Move away from monsters
        monster_dir = state.get("nearest_monster_direction")
        if monster_dir:
            opposite = {
                "north": "south", "south": "north",
                "east": "west", "west": "east"
            }
            return opposite.get(monster_dir, "wait")
        return "search"  # Look for escape route
        
    def _combat_action(self, state: Dict) -> str:
        """Engage in combat"""
        if state.get("monster_adjacent"):
            # Use terrain advantage
            if state.get("in_corridor"):
                return "wait"  # Let them come to you
            else:
                return "kick"  # Attack
        else:
            # Position for combat
            monster_dir = state.get("nearest_monster_direction")
            return monster_dir if monster_dir else "wait"

# ============================================================================
# HIERARCHICAL PROGRAM EXECUTOR
# ============================================================================

class HierarchicalExecutor:
    """Executes sub-programs hierarchically"""
    
    def __init__(self):
        self.program_stack = []  # Stack of active programs
        self.completed_programs = []
        
        # Library of available sub-programs
        self.library = {
            "navigate_stairs": NavigateTo(">", "stairs"),
            "open_door": OpenDoor(),
            "clear_room": ClearRoom(),
            "find_stairs": FindStairs(),
            "manage_inventory": ManageInventory(),
            "survive_combat": SurviveCombat(),
        }
        
    def select_program(self, state: Dict) -> Optional[SubProgram]:
        """Select the best program for current state"""
        # Priority order
        priorities = [
            ("survive_combat", lambda s: s.get("in_combat", False)),
            ("navigate_stairs", lambda s: s.get("stairs_visible", False)),
            ("open_door", lambda s: s.get("door_blocking", False)),
            ("manage_inventory", lambda s: s.get("needs_inventory_management", False)),
            ("clear_room", lambda s: s.get("in_room", False) and not s.get("room_cleared", False)),
            ("find_stairs", lambda s: s.get("explored_percent", 0) > 0.7),
        ]
        
        for prog_name, condition in priorities:
            if condition(state) and prog_name in self.library:
                program = self.library[prog_name]
                if program.can_execute(state):
                    return program
                    
        return None
        
    def execute_step(self, state: Dict) -> str:
        """Execute one step of the current program"""
        # Get current program or select new one
        if not self.program_stack:
            program = self.select_program(state)
            if program:
                self.program_stack.append(program)
            else:
                return "search"  # Default exploration
                
        current_program = self.program_stack[-1]
        
        # Check if current program is complete
        if current_program.is_complete(state):
            self.completed_programs.append(current_program)
            self.program_stack.pop()
            return self.execute_step(state)  # Recurse to select next
            
        # Get next action from current program
        action = current_program.get_next_action(state)
        if not action:
            # Program is stuck, try something else
            self.program_stack.pop()
            return "wait"
            
        return action

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def create_nethack_agent_with_subprograms():
    """Create an agent that uses hierarchical sub-programs"""
    
    class SubProgramAgent:
        def __init__(self):
            self.executor = HierarchicalExecutor()
            self.state_extractor = StateExtractor()
            
        def act(self, obs):
            # Extract high-level state from observation
            state = self.state_extractor.extract(obs)
            
            # Get action from hierarchical executor
            action = self.executor.execute_step(state)
            
            return action
            
    return SubProgramAgent()

class StateExtractor:
    """Extract high-level state from NetHack observations"""
    
    def extract(self, obs):
        state = {}
        
        # Parse observation
        text = obs.get("text", {})
        long_term = text.get("long_term_context", "")
        short_term = text.get("short_term_context", "")
        
        # Extract key information
        state["stairs_visible"] = ">" in long_term
        state["door_present"] = "+" in long_term
        state["door_blocking"] = "+" in long_term and "closed" in long_term
        state["in_combat"] = any(word in long_term for word in ["attacks", "hits", "misses"])
        state["monsters_present"] = any(word in long_term for word in ["monster", "enemy"])
        state["location_type"] = "room" if "room" in long_term else "corridor"
        state["in_room"] = state["location_type"] == "room"
        
        # Inventory and status
        state["hunger_level"] = "hungry" if "hungry" in short_term else "normal"
        state["hp_percent"] = self._extract_hp_percent(short_term)
        
        # Visible items and symbols
        state["visible_symbols"] = long_term
        state["items_visible"] = any(sym in long_term for sym in [")", "[", "?", "!"])
        
        # Exploration
        explored = long_term.count(".") + long_term.count("#")
        total = explored + long_term.count(" ")
        state["explored_percent"] = explored / max(total, 1)
        
        return state
        
    def _extract_hp_percent(self, stats_text):
        """Extract HP percentage from stats"""
        # Parse HP(current/max) format
        import re
        match = re.search(r'HP:(\d+)\((\d+)\)', stats_text)
        if match:
            current = int(match.group(1))
            max_hp = int(match.group(2))
            return current / max(max_hp, 1)
        return 1.0

print("NetHack Sub-Program Architecture Designed!")
print("\nKey Concepts:")
print("1. Goal-oriented behaviors with termination conditions")
print("2. Hierarchical execution with program stacks")
print("3. State abstraction from raw observations")
print("4. Priority-based program selection")
print("5. Fallback strategies when programs fail")