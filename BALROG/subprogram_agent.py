"""Sub-program based agent with repetition penalties for NetHack/MiniHack"""

from collections import deque, Counter
from typing import Dict, List, Tuple, Optional
import re

class SubProgram:
    """Base class for sub-programs that execute multiple actions"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def get_actions(self, context: Dict) -> List[str]:
        """Return list of actions to execute"""
        raise NotImplementedError
        
    def is_applicable(self, obs: Dict) -> bool:
        """Check if this sub-program is applicable given current observation"""
        raise NotImplementedError


class ExploreSubProgram(SubProgram):
    """Systematically explore unmapped areas"""
    
    def __init__(self):
        super().__init__("explore", "Systematically explore unmapped areas")
        self.directions = ["north", "east", "south", "west"]
        self.current_dir = 0
        
    def get_actions(self, context: Dict) -> List[str]:
        # Rotate through directions to avoid getting stuck
        actions = []
        for _ in range(4):
            actions.append(self.directions[self.current_dir])
            self.current_dir = (self.current_dir + 1) % 4
        return actions
        
    def is_applicable(self, obs: Dict) -> bool:
        # Check if there are unexplored areas in the map
        return "unexplored" in obs.get("text", {}).get("long_term_context", "").lower()


class CombatSubProgram(SubProgram):
    """Handle combat situations"""
    
    def __init__(self):
        super().__init__("combat", "Fight nearby monsters")
        
    def get_actions(self, context: Dict) -> List[str]:
        # Attack in direction of monster
        monster_dir = context.get("monster_direction", "north")
        return [f"fight {monster_dir}", "wait", f"fight {monster_dir}"]
        
    def is_applicable(self, obs: Dict) -> bool:
        text = obs.get("text", {}).get("long_term_context", "").lower()
        return any(word in text for word in ["monster", "enemy", "hostile", "attacks"])


class InventorySubProgram(SubProgram):
    """Manage inventory - eat food, wear armor, etc."""
    
    def __init__(self):
        super().__init__("inventory", "Manage inventory items")
        
    def get_actions(self, context: Dict) -> List[str]:
        actions = []
        inventory = context.get("inventory", "")
        
        # Eat if hungry
        if "hungry" in context.get("status", ""):
            if re.search(r'[a-z]\s*-\s*\w*food', inventory):
                food_letter = re.search(r'([a-z])\s*-\s*\w*food', inventory).group(1)
                actions.extend(["eat", food_letter])
                
        # Wear armor if found
        if re.search(r'[a-z]\s*-\s*\w*armor', inventory):
            armor_letter = re.search(r'([a-z])\s*-\s*\w*armor', inventory).group(1)
            actions.extend(["wear", armor_letter])
            
        return actions if actions else ["wait"]
        
    def is_applicable(self, obs: Dict) -> bool:
        inventory = obs.get("text", {}).get("short_term_context", "")
        status = obs.get("text", {}).get("long_term_context", "")
        return ("hungry" in status.lower() or 
                "armor" in inventory.lower() or 
                "food" in inventory.lower())


class StairNavigationSubProgram(SubProgram):
    """Navigate to and use stairs"""
    
    def __init__(self):
        super().__init__("stairs", "Navigate to stairs")
        
    def get_actions(self, context: Dict) -> List[str]:
        # Move toward stairs if visible
        stair_dir = context.get("stair_direction")
        if stair_dir:
            return [stair_dir, stair_dir, "down"]
        return ["search", "search", "search"]  # Search for hidden stairs
        
    def is_applicable(self, obs: Dict) -> bool:
        text = obs.get("text", {}).get("long_term_context", "")
        return "stair" in text.lower() or ">" in text


class SubProgramAgent:
    """Agent that uses sub-programs instead of individual actions"""
    
    def __init__(self, base_agent, repetition_penalty_window: int = 20):
        self.base_agent = base_agent
        self.sub_programs = [
            CombatSubProgram(),
            StairNavigationSubProgram(),
            InventorySubProgram(),
            ExploreSubProgram(),  # Default fallback
        ]
        
        # For repetition penalty
        self.action_history = deque(maxlen=repetition_penalty_window)
        self.repetition_penalty_factor = 0.5
        
    def act(self, obs: Dict, prev_action: Optional[str] = None) -> str:
        # Extract context from observation
        context = self._extract_context(obs)
        
        # Select applicable sub-program
        selected_program = None
        for program in self.sub_programs:
            if program.is_applicable(obs):
                selected_program = program
                break
                
        if not selected_program:
            # Fallback to base agent
            return self.base_agent.act(obs, prev_action)
            
        # Get actions from sub-program
        candidate_actions = selected_program.get_actions(context)
        
        # Apply repetition penalty
        action_scores = {}
        action_counts = Counter(self.action_history)
        
        for action in candidate_actions:
            # Base score (1.0 for first action, decreasing for later ones)
            base_score = 1.0 / (candidate_actions.index(action) + 1)
            
            # Apply repetition penalty
            repetition_count = action_counts.get(action, 0)
            penalty = self.repetition_penalty_factor ** repetition_count
            
            action_scores[action] = base_score * penalty
            
        # Select action with highest score
        best_action = max(action_scores, key=action_scores.get)
        
        # Update history
        self.action_history.append(best_action)
        
        # Create modified prompt for LLM that restricts to this action
        self._modify_prompt_for_action(best_action)
        
        return best_action
        
    def _extract_context(self, obs: Dict) -> Dict:
        """Extract relevant context from observation"""
        context = {}
        
        text = obs.get("text", {})
        long_term = text.get("long_term_context", "")
        short_term = text.get("short_term_context", "")
        
        # Extract inventory
        if "inventory:" in short_term.lower():
            context["inventory"] = short_term
            
        # Extract status (hungry, weak, etc.)
        if "statistics:" in short_term.lower():
            context["status"] = short_term
            
        # Detect monsters and their direction
        monster_pattern = r'(monster|enemy|hostile).*(north|south|east|west)'
        match = re.search(monster_pattern, long_term.lower())
        if match:
            context["monster_direction"] = match.group(2)
            
        # Detect stairs and direction
        stair_pattern = r'(stair|>).*(north|south|east|west)'
        match = re.search(stair_pattern, long_term.lower())
        if match:
            context["stair_direction"] = match.group(2)
            
        return context
        
    def _modify_prompt_for_action(self, action: str):
        """Modify the prompt builder to suggest specific action"""
        # This would integrate with the existing prompt builder
        # to guide the LLM toward the selected action
        pass


# Example integration with existing BALROG agent
class SubProgramNaiveAgent:
    """Naive agent enhanced with sub-programs and repetition penalties"""
    
    def __init__(self, client_factory, prompt_builder):
        # Create base naive agent
        from balrog.agents.naive import NaiveAgent
        base_agent = NaiveAgent(client_factory, prompt_builder)
        
        # Wrap with sub-program logic
        self.sub_program_agent = SubProgramAgent(base_agent)
        
        # Keep references for BALROG compatibility
        self.client = base_agent.client
        self.prompt_builder = prompt_builder
        
    def act(self, obs, prev_action=None):
        return self.sub_program_agent.act(obs, prev_action)
        
    def reset(self):
        self.sub_program_agent.action_history.clear()


# Simplified action space for core actions only
CORE_NETHACK_ACTIONS = {
    # Movement
    "north", "south", "east", "west",
    "northeast", "southeast", "southwest", "northwest",
    
    # Interaction
    "pickup", "drop", "eat", "quaff", "read", "wear", "wield",
    "open", "close", "search", "look", "wait",
    
    # Combat
    "fight",  # Simplified - direction added by sub-program
    
    # Navigation  
    "up", "down",
    
    # Special
    "pray", "apply", "zap"
}

# This reduces action space from ~200+ to ~25 core actions