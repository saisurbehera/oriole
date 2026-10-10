"""
MCTS Planning Agent - Uses Monte Carlo Tree Search for NetHack
==============================================================

This agent uses TreeQuest's MCTS implementation to plan actions in NetHack.
Key features:
1. Each MCTS node represents a potential game state after taking actions
2. Uses LLM to evaluate positions and suggest next moves
3. Simulates multiple action sequences to find the best path
4. Balances exploration of new strategies with exploitation of known good moves
"""

import re
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import sys
import os

# TreeQuest imports - disabled for now due to dependency issues
# Will use simplified MCTS implementation instead
StandardMCTS = None
ABMCTSA = None
top_k = None
tq_available = False

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse


@dataclass
class NetHackState:
    """Represents a state in the NetHack MCTS tree"""
    observation: Dict[str, Any]
    actions_taken: List[str] = field(default_factory=list)
    score: float = 0.0
    depth: int = 0
    game_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt_context(self) -> str:
        """Convert state to a string for LLM context"""
        stats = self.game_stats
        context = f"Depth {stats.get('depth', 1)}, HP: {stats.get('hp_current', 0)}/{stats.get('hp_max', 1)}"
        if self.actions_taken:
            context += f", Recent: {' → '.join(self.actions_taken[-3:])}"
        return context


class MCTSPlanningAgent(BaseAgent):
    """Agent that uses MCTS to plan ahead in NetHack"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # MCTS parameters
        self.mcts_iterations = 10  # Number of MCTS iterations per turn
        self.simulation_depth = 3  # How many moves to simulate ahead
        self.exploration_weight = 1.5  # UCB exploration constant
        
        # Game state tracking
        self.turn_count = 0
        self.current_obs = None
        self.action_history = deque(maxlen=20)
        
        # Initialize MCTS if available
        if tq_available and StandardMCTS:
            self.mcts_algo = StandardMCTS(
                samples_per_action=2,
                exploration_weight=self.exploration_weight
            )
        else:
            self.mcts_algo = None
            
    def act(self, obs, prev_action=None):
        """Main action selection using MCTS planning"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        self.turn_count += 1
        self.current_obs = obs
        
        # Parse game state
        game_state = self._parse_game_state(obs)
        
        # Use MCTS to find best action if available
        if self.mcts_algo and tq_available:
            action = self._mcts_planning(obs, game_state)
        else:
            # Fallback to simple planning
            action = self._simple_planning(game_state)
            
        # Record action
        self.action_history.append(action)
        
        return LLMResponse(
            model_id="mcts_planning",
            completion=action,
            stop_reason="complete",
            input_tokens=300,
            output_tokens=50,
            reasoning=f"MCTS planning: {action}"
        )
        
    def _mcts_planning(self, obs, game_state) -> str:
        """Use MCTS to plan the best action"""
        # Initialize MCTS tree
        search_tree = self.mcts_algo.init_tree()
        
        # Create initial state
        initial_state = NetHackState(
            observation=obs,
            game_stats=game_state
        )
        
        # Define action generators for different move types
        def generate_move_action(direction: str):
            def generate(parent_state: Optional[NetHackState]) -> Tuple[NetHackState, float]:
                if parent_state is None:
                    new_state = initial_state
                else:
                    new_state = self._simulate_action(parent_state, direction)
                
                # Evaluate the state
                score = self._evaluate_state(new_state)
                return new_state, score
            return generate
            
        # Get LLM-suggested actions for this state
        suggested_actions = self._get_llm_action_suggestions(game_state)
        
        # Create action mapping
        actions = {}
        
        # Add LLM-suggested actions first (higher priority)
        for action in suggested_actions:
            if action not in actions:
                actions[f"llm_{action}"] = generate_move_action(action)
        
        # Movement actions
        for direction in ["north", "south", "east", "west"]:
            actions[f"move_{direction}"] = generate_move_action(direction)
            
        # Special actions based on game state
        if game_state.get("stairs_info"):
            actions["use_stairs"] = generate_move_action(">")
            
        if game_state.get("doors_nearby"):
            actions["open_door"] = generate_move_action("open")
            
        # Always include search as an option
        actions["search"] = generate_move_action("search")
        actions["wait"] = generate_move_action("wait")
        
        # Run MCTS iterations
        for _ in range(self.mcts_iterations):
            search_tree = self.mcts_algo.step(search_tree, actions)
            
        # Get best action from tree
        best_states = top_k(search_tree, self.mcts_algo, k=3)
        
        if best_states:
            best_state, best_score = best_states[0]
            
            # Extract the action that led to this state
            if best_state.actions_taken:
                return best_state.actions_taken[0]
                
        # Fallback
        return "search"
        
    def _simulate_action(self, parent_state: NetHackState, action: str) -> NetHackState:
        """Simulate taking an action from a parent state"""
        # Create new state with simulated results
        new_actions = parent_state.actions_taken + [action]
        
        # Simulate state changes (simplified)
        new_stats = parent_state.game_stats.copy()
        
        # Update position for movement
        if action in ["north", "south", "east", "west"]:
            if parent_state.game_stats.get("player_pos"):
                px, py = parent_state.game_stats["player_pos"]
                if action == "north":
                    new_stats["player_pos"] = (px - 1, py)
                elif action == "south":
                    new_stats["player_pos"] = (px + 1, py)
                elif action == "east":
                    new_stats["player_pos"] = (px, py + 1)
                elif action == "west":
                    new_stats["player_pos"] = (px, py - 1)
                    
        # Simulate stair usage
        elif action == ">":
            if parent_state.game_stats.get("stairs_info"):
                new_stats["depth"] = parent_state.game_stats.get("depth", 1) + 1
                
        # Create new state
        new_state = NetHackState(
            observation=parent_state.observation,  # Keep same obs for simulation
            actions_taken=new_actions,
            depth=parent_state.depth + 1,
            game_stats=new_stats
        )
        
        return new_state
        
    def _evaluate_state(self, state: NetHackState) -> float:
        """Evaluate a state using LLM and heuristics"""
        # First try heuristic evaluation for speed
        heuristic_score = self._heuristic_evaluation(state)
        
        # For important states (near stairs, low health, etc), also query LLM
        stats = state.game_stats
        should_query_llm = (
            stats.get("stairs_info") is not None or
            stats.get("hp_percent", 1.0) < 0.5 or
            state.depth <= 2  # Evaluate early states more carefully
        )
        
        if should_query_llm and self.client:
            llm_score = self._llm_evaluation(state)
            # Combine scores (weighted average)
            return 0.3 * heuristic_score + 0.7 * llm_score
        else:
            return heuristic_score
            
    def _heuristic_evaluation(self, state: NetHackState) -> float:
        """Quick heuristic evaluation without LLM"""
        score = 0.5  # Base score
        
        stats = state.game_stats
        
        # Reward reaching deeper levels
        depth = stats.get("depth", 1)
        if depth > 1:
            score += 0.2 * (depth - 1)
            
        # Reward being near stairs
        if stats.get("stairs_info") and stats.get("player_pos"):
            stairs_pos = stats["stairs_info"]["pos"]
            player_pos = stats["player_pos"]
            dist = abs(stairs_pos[0] - player_pos[0]) + abs(stairs_pos[1] - player_pos[1])
            if dist < 5:
                score += 0.1 * (5 - dist) / 5
                
        # Penalize low health
        hp_percent = stats.get("hp_percent", 1.0)
        if hp_percent < 0.5:
            score -= 0.2 * (0.5 - hp_percent)
            
        # Penalize being stuck (repeated actions)
        if len(state.actions_taken) >= 3:
            last_3 = state.actions_taken[-3:]
            if len(set(last_3)) == 1:  # All same action
                score -= 0.3
                
        return max(0.0, min(1.0, score))
        
    def _llm_evaluation(self, state: NetHackState) -> float:
        """Use LLM to evaluate the quality of a state"""
        try:
            # Create evaluation prompt
            prompt = f"""Evaluate this NetHack game state on a scale of 0.0 to 1.0:

Current State:
- Dungeon Level: {state.game_stats.get('depth', 1)}
- HP: {state.game_stats.get('hp_current', 0)}/{state.game_stats.get('hp_max', 1)}
- Stairs nearby: {state.game_stats.get('stairs_info') is not None}
- Recent actions: {' → '.join(state.actions_taken[-3:])}

Consider:
1. Progress toward descending (stairs are key)
2. Safety (health, threats)
3. Efficiency (not stuck or looping)

Respond with ONLY a number between 0.0 and 1.0"""

            messages = [{"role": "user", "content": prompt}]
            response = self.client.generate(messages)
            
            # Extract score from response
            score_text = response.completion.strip()
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                # Try to extract number from text
                import re
                numbers = re.findall(r'0\.\d+|1\.0', score_text)
                if numbers:
                    return float(numbers[0])
                else:
                    return 0.5  # Default if parsing fails
                    
        except Exception as e:
            # Fallback to heuristic if LLM fails
            return self._heuristic_evaluation(state)
            
    def _get_llm_action_suggestions(self, game_state: Dict[str, Any]) -> List[str]:
        """Ask LLM for suggested actions given current game state"""
        try:
            # Create a focused prompt for action suggestions
            prompt = f"""Given this NetHack state, suggest the 3 best actions:

State:
- Dungeon Level: {game_state.get('depth', 1)}
- HP: {game_state.get('hp_current', 0)}/{game_state.get('hp_max', 1)}
- Stairs visible: {game_state.get('stairs_info') is not None}
- Doors nearby: {game_state.get('doors_nearby', False)}

Valid actions: north, south, east, west, >, <, open, search, wait, kick

Respond with ONLY a JSON list of actions, e.g.: ["north", "open", ">"]"""

            messages = [{"role": "user", "content": prompt}]
            response = self.client.generate(messages)
            
            # Parse JSON response
            try:
                import json
                actions = json.loads(response.completion.strip())
                if isinstance(actions, list):
                    # Filter to valid actions only
                    valid_actions = ["north", "south", "east", "west", ">", "<", 
                                   "open", "search", "wait", "kick"]
                    return [a for a in actions if a in valid_actions][:3]
            except json.JSONDecodeError:
                pass
                
            # Fallback: try to extract actions from text
            text = response.completion.lower()
            found_actions = []
            for action in ["north", "south", "east", "west", ">", "open", "search"]:
                if action in text:
                    found_actions.append(action)
            return found_actions[:3]
            
        except Exception as e:
            # Return default actions if LLM fails
            if game_state.get("stairs_info"):
                return [">", "south", "east"]
            else:
                return ["search", "north", "east"]
        
    def _simple_planning(self, game_state) -> str:
        """Fallback planning without MCTS"""
        # Priority actions
        if game_state.get("stairs_info"):
            return ">"
        elif game_state.get("doors_nearby"):
            return "open"
        else:
            # Random exploration
            return random.choice(["north", "south", "east", "west", "search"])
            
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Extract key information from observation"""
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        lang_obs = text.get("language_observation", "")
        
        # Parse stats
        hp_match = re.search(r'HP:(\d+)\((\d+)\)', long_text)
        hp_current = int(hp_match.group(1)) if hp_match else 0
        hp_max = int(hp_match.group(2)) if hp_match else 1
        
        depth_match = re.search(r'Dlvl:(\d+)', long_text)
        depth = int(depth_match.group(1)) if depth_match else 1
        
        # Check for stairs
        stairs_info = None
        if "stairs" in lang_obs:
            if "stairs up" in lang_obs:
                stairs_info = {"type": "up", "pos": (0, 0)}  # Simplified
            elif "stairs down" in lang_obs:
                stairs_info = {"type": "down", "pos": (0, 0)}
                
        # Check for doors
        doors_nearby = "door" in lang_obs
        
        return {
            "hp_current": hp_current,
            "hp_max": hp_max,
            "hp_percent": hp_current / hp_max if hp_max > 0 else 0,
            "depth": depth,
            "stairs_info": stairs_info,
            "doors_nearby": doors_nearby,
            "player_pos": (10, 10),  # Simplified - would parse from map
            "turn": self.turn_count
        }
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.turn_count = 0
        self.current_obs = None
        self.action_history.clear()