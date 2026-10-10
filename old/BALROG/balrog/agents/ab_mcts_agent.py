"""
AB-MCTS agent using TreeQuest library with LLM guidance
"""

import sys
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add treequest to path
treequest_path = '/home/sai/Desktop/oriole/BALROG/treequest/src'
if treequest_path not in sys.path:
    sys.path.insert(0, treequest_path)

# Import modules directly without going through __init__.py
import importlib.util
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import required modules directly
ab_mcts_a_module = import_module_from_path("ab_mcts_a", "/home/sai/Desktop/oriole/BALROG/treequest/src/treequest/algos/ab_mcts_a/algo.py")
tree_module = import_module_from_path("tree", "/home/sai/Desktop/oriole/BALROG/treequest/src/treequest/algos/tree.py")
ranker_module = import_module_from_path("ranker", "/home/sai/Desktop/oriole/BALROG/treequest/src/treequest/ranker.py")

ABMCTSA = ab_mcts_a_module.ABMCTSA
Tree = tree_module.Tree
top_k = ranker_module.top_k
from .base import BaseAgent
from balrog.client import LLMResponse


@dataclass
class NetHackState:
    """State for NetHack MCTS nodes"""
    observation: Dict[str, Any]
    reasoning: str
    action_plan: str
    depth: int
    score: float
    hp: int
    on_stairs: bool = False
    
    def __str__(self):
        return f"D{self.depth} HP{self.hp} Score{self.score} {'[STAIRS]' if self.on_stairs else ''}: {self.action_plan}"


class ABMCTSAgent(BaseAgent):
    """AB-MCTS agent using TreeQuest with LLM guidance"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Initialize TreeQuest AB-MCTS
        self.algo = ABMCTSA()
        self.search_tree = None
        
        # Track episode state
        self.current_observation = None
        self.turn_count = 0
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.search_tree = self.algo.init_tree()
        self.current_observation = None
        self.turn_count = 0
        
    def act(self, observation: Dict[str, Any], prev_action: str = None, return_reasoning: bool = False) -> LLMResponse:
        """Select action using AB-MCTS with LLM guidance"""
        self.turn_count += 1
        self.current_observation = observation
        
        # Initialize tree if first turn
        if self.search_tree is None:
            self.search_tree = self.algo.init_tree()
        
        # Define generation functions for different action types
        generate_fns = {
            'explore': self._generate_exploration_action,
            'navigate': self._generate_navigation_action,
            'combat': self._generate_combat_action
        }
        
        # Run AB-MCTS search with budget
        search_budget = 5  # Number of MCTS iterations
        for _ in range(search_budget):
            self.search_tree = self.algo.step(self.search_tree, generate_fns)
        
        # Get best action from search tree
        best_results = top_k(self.search_tree, self.algo, k=1)
        if best_results:
            best_state, best_score = best_results[0]
            if isinstance(best_state, NetHackState):
                # Extract action from the plan
                action = self._extract_action_from_plan(best_state.action_plan)
                reasoning = f"AB-MCTS (score: {best_score:.3f}): {best_state.reasoning}"
            else:
                action = "wait"
                reasoning = "AB-MCTS fallback: waiting"
        else:
            action = "wait" 
            reasoning = "AB-MCTS: no results, waiting"
        
        return LLMResponse(
            model_id="ab_mcts",
            completion=action,
            stop_reason="complete",
            input_tokens=100,
            output_tokens=20,
            reasoning=reasoning
        )
    
    def _generate_exploration_action(self, parent_state: Optional[NetHackState]) -> Tuple[NetHackState, float]:
        """Generate exploration-focused action using LLM"""
        return self._generate_llm_action(parent_state, "exploration")
    
    def _generate_navigation_action(self, parent_state: Optional[NetHackState]) -> Tuple[NetHackState, float]:
        """Generate navigation-focused action using LLM"""
        return self._generate_llm_action(parent_state, "navigation")
    
    def _generate_combat_action(self, parent_state: Optional[NetHackState]) -> Tuple[NetHackState, float]:
        """Generate combat-focused action using LLM"""
        return self._generate_llm_action(parent_state, "combat")
    
    def _generate_llm_action(self, parent_state: Optional[NetHackState], focus: str) -> Tuple[NetHackState, float]:
        """Generate action using LLM with specific focus"""
        try:
            # Parse current game state
            game_state = self._parse_game_state(self.current_observation)
            
            # Build context prompt
            context = self._build_context_prompt(parent_state, focus, game_state)
            
            # Query LLM
            response = self.client.query(context, max_tokens=150, temperature=0.7)
            
            # Parse LLM response
            llm_text = response.completion.strip()
            
            # Extract action and reasoning
            action_plan, reasoning = self._parse_llm_response(llm_text)
            
            # Calculate score based on game state and focus
            score = self._calculate_state_score(game_state, action_plan, focus)
            
            # Create new state
            new_state = NetHackState(
                observation=self.current_observation,
                reasoning=reasoning,
                action_plan=action_plan,
                depth=game_state.get('depth', 1),
                score=game_state.get('score', 0),
                hp=game_state.get('hp_current', 16),
                on_stairs=game_state.get('on_stairs', False)
            )
            
            return new_state, score
            
        except Exception as e:
            # Fallback state
            fallback_state = NetHackState(
                observation=self.current_observation,
                reasoning=f"Error in LLM generation: {e}",
                action_plan="wait",
                depth=1,
                score=0,
                hp=16
            )
            return fallback_state, 0.1
    
    def _build_context_prompt(self, parent_state: Optional[NetHackState], focus: str, game_state: Dict) -> str:
        """Build context prompt for LLM"""
        prompt_parts = [
            f"You are playing NetHack. Focus: {focus}",
            f"Current depth: {game_state.get('depth', 1)}",
            f"Current HP: {game_state.get('hp_current', 16)}",
            f"Current score: {game_state.get('score', 0)}",
        ]
        
        if parent_state:
            prompt_parts.append(f"Previous action: {parent_state.action_plan}")
            prompt_parts.append(f"Previous reasoning: {parent_state.reasoning}")
        
        # Add game state info
        if game_state.get('on_stairs'):
            prompt_parts.append("🎯 You are ON STAIRS! Use 'down' to descend to next level.")
        elif game_state.get('stairs_pos'):
            prompt_parts.append(f"Stairs visible at position {game_state['stairs_pos']}")
        
        if game_state.get('monsters_nearby'):
            prompt_parts.append("⚔️ Monsters nearby - consider combat or avoidance")
        
        if focus == "exploration":
            prompt_parts.append("Focus on exploring to find stairs down to level 2.")
        elif focus == "navigation":
            prompt_parts.append("Focus on efficient movement toward your goal.")
        elif focus == "combat":
            prompt_parts.append("Focus on combat tactics and survival.")
        
        prompt_parts.extend([
            "",
            "Respond with JSON:",
            '{"action": "specific_action", "reasoning": "why this action"}',
            "",
            "Available actions: north, south, east, west, northeast, northwest, southeast, southwest, down, wait, search, open, kick, fight"
        ])
        
        return "\\n".join(prompt_parts)
    
    def _parse_llm_response(self, llm_text: str) -> Tuple[str, str]:
        """Parse LLM response to extract action and reasoning"""
        try:
            # Try to parse as JSON
            if '{' in llm_text and '}' in llm_text:
                json_start = llm_text.find('{')
                json_end = llm_text.rfind('}') + 1
                json_text = llm_text[json_start:json_end]
                parsed = json.loads(json_text)
                
                action = parsed.get('action', 'wait')
                reasoning = parsed.get('reasoning', 'No reasoning provided')
                return action, reasoning
        except:
            pass
        
        # Fallback parsing
        lines = llm_text.split('\\n')
        action = "wait"
        reasoning = "Failed to parse LLM response"
        
        for line in lines:
            if any(act in line.lower() for act in ['north', 'south', 'east', 'west', 'down', 'wait', 'search']):
                action = line.strip().split()[0].lower()
                reasoning = line.strip()
                break
                
        return action, reasoning
    
    def _calculate_state_score(self, game_state: Dict, action_plan: str, focus: str) -> float:
        """Calculate score for the state based on game situation"""
        score = 0.5  # Base score
        
        # Reward based on game progress
        if game_state.get('depth', 1) > 1:
            score += 0.4  # Major bonus for reaching deeper levels
        
        if game_state.get('on_stairs') and 'down' in action_plan:
            score += 0.3  # Bonus for using stairs correctly
        elif game_state.get('stairs_pos') and any(move in action_plan for move in ['north', 'south', 'east', 'west']):
            score += 0.2  # Bonus for moving when stairs are visible
        
        # Focus-specific scoring
        if focus == "exploration":
            if any(move in action_plan for move in ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']):
                score += 0.1
        elif focus == "navigation":
            if game_state.get('stairs_pos') and any(move in action_plan for move in ['north', 'south', 'east', 'west']):
                score += 0.2
        elif focus == "combat":
            if game_state.get('monsters_nearby') and action_plan in ['fight', 'kick']:
                score += 0.15
        
        # Penalty for waiting too much
        if action_plan == 'wait':
            score -= 0.1
            
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _extract_action_from_plan(self, action_plan: str) -> str:
        """Extract single action from action plan"""
        # Look for valid NetHack actions
        valid_actions = [
            'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
            'down', 'up', 'wait', 'search', 'open', 'kick', 'fight', 'look', 'inventory'
        ]
        
        plan_lower = action_plan.lower()
        for action in valid_actions:
            if action in plan_lower:
                return action
                
        return 'wait'  # Fallback
    
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Parse the game state from observation (simplified version)"""
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        
        state = {
            "depth": 1,
            "hp_current": 16,
            "hp_max": 16,
            "score": 0,
            "on_stairs": False,
            "stairs_pos": None,
            "monsters_nearby": False
        }
        
        # Parse depth
        import re
        depth_match = re.search(r'Dlvl:(\\d+)', long_text)
        if depth_match:
            state["depth"] = int(depth_match.group(1))
        
        # Parse HP
        hp_match = re.search(r'HP:(\\d+)\\((\\d+)\\)', long_text)
        if hp_match:
            state["hp_current"] = int(hp_match.group(1))
            state["hp_max"] = int(hp_match.group(2))
        
        # Parse score
        score_match = re.search(r'S:(\\d+)', long_text)
        if score_match:
            state["score"] = int(score_match.group(1))
        
        # Simple checks for stairs and monsters
        state["on_stairs"] = "staircase down" in long_text.lower()
        state["stairs_pos"] = (0, 0) if "<" in long_text else None
        state["monsters_nearby"] = any(monster in long_text.lower() for monster in ["kobold", "goblin", "rat", "fox", "dog", "cat"])
        
        return state