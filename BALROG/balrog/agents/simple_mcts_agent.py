"""
Simple MCTS Agent - Basic Monte Carlo Tree Search for NetHack
=============================================================

This agent implements a simplified MCTS without external dependencies.
It builds a tree of possible action sequences and uses simulations
to find the best path forward.
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse


@dataclass
class MCTSNode:
    """A node in the MCTS tree"""
    state: Dict[str, Any]
    action: Optional[str] = None
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_score: float = 0.0
    
    @property
    def average_score(self) -> float:
        """Average score of this node"""
        return self.total_score / self.visits if self.visits > 0 else 0.0
    
    @property
    def ucb_score(self, c: float = 1.414) -> float:
        """Upper Confidence Bound score for node selection"""
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return self.average_score
        
        exploitation = self.average_score
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
        
    def is_fully_expanded(self, possible_actions: List[str]) -> bool:
        """Check if all possible actions have been tried from this node"""
        return len(self.children) >= len(possible_actions)
        
    def get_untried_actions(self, possible_actions: List[str]) -> List[str]:
        """Get actions that haven't been tried yet"""
        tried_actions = {child.action for child in self.children}
        return [a for a in possible_actions if a not in tried_actions]


class SimpleMCTSAgent(BaseAgent):
    """Agent that uses simplified MCTS for planning"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # MCTS parameters
        self.iterations_per_move = 20
        self.simulation_depth = 5
        self.exploration_constant = 1.414
        
        # Game tracking
        self.turn_count = 0
        self.action_history = deque(maxlen=50)
        self.position_history = deque(maxlen=10)
        
    def act(self, obs, prev_action=None):
        """Select action using MCTS"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        self.turn_count += 1
        
        # Parse game state
        game_state = self._parse_game_state(obs)
        
        # Update position tracking
        if game_state.get("player_pos"):
            self.position_history.append(game_state["player_pos"])
        
        # Run MCTS to find best action
        action = self._run_mcts(game_state)
        
        # Record action
        self.action_history.append(action)
        
        return LLMResponse(
            model_id="simple_mcts",
            completion=action,
            stop_reason="complete",
            input_tokens=100,
            output_tokens=10,
            reasoning=f"MCTS selected: {action}"
        )
        
    def _run_mcts(self, initial_state: Dict[str, Any]) -> str:
        """Run MCTS algorithm to select best action"""
        # Create root node
        root = MCTSNode(state=initial_state)
        
        # Get possible actions for this state
        possible_actions = self._get_possible_actions(initial_state)
        
        # Run MCTS iterations
        for _ in range(self.iterations_per_move):
            # Selection - traverse tree to leaf
            node = self._select(root, possible_actions)
            
            # Expansion - add new child if not fully expanded
            if not node.is_fully_expanded(possible_actions) and node.visits > 0:
                node = self._expand(node, possible_actions)
            
            # Simulation - play out random game
            score = self._simulate(node)
            
            # Backpropagation - update scores up the tree
            self._backpropagate(node, score)
        
        # Choose best action based on visit count
        if root.children:
            # Get top 3 candidates
            sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
            top_children = sorted_children[:3]
            
            # If close competition, use LLM to break tie
            if len(top_children) >= 2 and top_children[0].visits < top_children[1].visits * 1.5:
                # Ask LLM to choose between top options
                return self._llm_tiebreaker(initial_state, [c.action for c in top_children])
            else:
                return top_children[0].action
        else:
            # Fallback if no children
            return random.choice(possible_actions)
            
    def _select(self, node: MCTSNode, possible_actions: List[str]) -> MCTSNode:
        """Select promising node to explore"""
        while node.children and node.is_fully_expanded(possible_actions):
            # Choose child with highest UCB score
            node = max(node.children, key=lambda c: c.ucb_score)
        return node
        
    def _expand(self, node: MCTSNode, possible_actions: List[str]) -> MCTSNode:
        """Expand node by adding a new child"""
        untried_actions = node.get_untried_actions(possible_actions)
        if untried_actions:
            # Choose random untried action
            action = random.choice(untried_actions)
            
            # Create new state after taking action
            new_state = self._apply_action(node.state, action)
            
            # Create and add child node
            child = MCTSNode(
                state=new_state,
                action=action,
                parent=node
            )
            node.children.append(child)
            return child
        return node
        
    def _simulate(self, node: MCTSNode) -> float:
        """Simulate game from this node to estimate value"""
        state = node.state.copy()
        depth = 0
        action_sequence = []
        
        # Smart simulation using LLM guidance for key decisions
        while depth < self.simulation_depth:
            # Check for terminal conditions
            if state.get("depth", 1) > 1:
                # Reached next level - very good!
                return 1.0
                
            # Get possible actions
            possible_actions = self._get_possible_actions(state)
            if not possible_actions:
                break
            
            # For first few steps or critical states, use LLM
            if depth < 2 or state.get("stairs_info"):
                action = self._get_llm_action_choice(state, possible_actions)
            else:
                # Otherwise use simple heuristics
                action = self._get_heuristic_action(state, possible_actions)
                
            state = self._apply_action(state, action)
            action_sequence.append(action)
            depth += 1
        
        # Evaluate final state
        return self._evaluate_state(state)
        
    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        """Update scores back up the tree"""
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent
            
    def _get_possible_actions(self, state: Dict[str, Any]) -> List[str]:
        """Get valid actions for current state"""
        actions = []
        
        # Always possible movements
        actions.extend(["north", "south", "east", "west"])
        
        # Stairs action
        if state.get("stairs_info"):
            actions.append(">")
            
        # Door action
        if state.get("doors_nearby"):
            actions.append("open")
            
        # Other common actions
        actions.extend(["search", "wait"])
        
        # Check if we're stuck
        if len(set(self.position_history)) <= 2 and len(self.position_history) >= 5:
            # Add more aggressive actions
            actions.extend(["kick", "search", "search"])  # Increase search probability
            
        return actions
        
    def _apply_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Apply action to state (simplified simulation)"""
        new_state = state.copy()
        
        # Simulate movement
        if action in ["north", "south", "east", "west"] and state.get("player_pos"):
            px, py = state["player_pos"]
            if action == "north":
                new_state["player_pos"] = (px - 1, py)
            elif action == "south":
                new_state["player_pos"] = (px + 1, py)
            elif action == "east":
                new_state["player_pos"] = (px, py + 1)
            elif action == "west":
                new_state["player_pos"] = (px, py - 1)
                
        # Simulate stairs
        elif action == ">" and state.get("stairs_info"):
            new_state["depth"] = state.get("depth", 1) + 1
            
        return new_state
        
    def _evaluate_state(self, state: Dict[str, Any]) -> float:
        """Evaluate state quality (0-1)"""
        score = 0.5  # Base score
        
        # Reward depth
        depth = state.get("depth", 1)
        if depth > 1:
            score += 0.3
            
        # Reward being near stairs
        if state.get("stairs_info"):
            score += 0.2
            
        # Penalize low health
        hp_percent = state.get("hp_percent", 1.0)
        if hp_percent < 0.5:
            score -= 0.2
            
        return max(0.0, min(1.0, score))
        
    def _get_llm_action_choice(self, state: Dict[str, Any], actions: List[str]) -> str:
        """Use LLM to choose best action from available options"""
        try:
            # Create focused prompt
            prompt = f"""NetHack position analysis:
Level: {state.get('depth', 1)}
HP: {state.get('hp_percent', 1.0)*100:.0f}%
Stairs nearby: {state.get('stairs_info') is not None}

Available actions: {', '.join(actions[:6])}

Which action gets us closer to stairs/next level? Reply with just the action."""

            messages = [{"role": "user", "content": prompt}]
            response = self.client.generate(messages)
            
            # Extract action from response
            response_text = response.completion.strip().lower()
            
            # Try to find a valid action in response
            for action in actions:
                if action in response_text:
                    return action
                    
            # Fallback to heuristic
            return self._get_heuristic_action(state, actions)
            
        except Exception:
            # On error, use heuristic
            return self._get_heuristic_action(state, actions)
            
    def _get_heuristic_action(self, state: Dict[str, Any], actions: List[str]) -> str:
        """Simple heuristic action selection"""
        # Priority order
        priority_actions = [">", "search", "open", "kick"]
        
        for priority in priority_actions:
            if priority in actions:
                return priority
                
        # Prefer unexplored directions
        if "north" in actions and random.random() < 0.3:
            return "north"
        elif "east" in actions and random.random() < 0.3:
            return "east"
            
        return random.choice(actions)
        
    def _llm_tiebreaker(self, state: Dict[str, Any], actions: List[str]) -> str:
        """Use LLM to break ties between similarly good actions"""
        try:
            prompt = f"""NetHack MCTS decision:
Current level: {state.get('depth', 1)}
Stairs visible: {state.get('stairs_info') is not None}

Top actions from tree search: {', '.join(actions)}

Pick the BEST action to find stairs. Reply with just the action."""

            messages = [{"role": "user", "content": prompt}]
            response = self.client.generate(messages)
            
            response_text = response.completion.strip().lower()
            for action in actions:
                if action in response_text:
                    return action
                    
            return actions[0]  # Default to most visited
            
        except Exception:
            return actions[0]
        
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Parse observation into game state"""
        import re
        
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        lang_obs = text.get("language_observation", "")
        
        # Parse HP
        hp_match = re.search(r'HP:(\d+)\((\d+)\)', long_text)
        hp_current = int(hp_match.group(1)) if hp_match else 16
        hp_max = int(hp_match.group(2)) if hp_match else 16
        
        # Parse depth
        depth_match = re.search(r'Dlvl:(\d+)', long_text)
        depth = int(depth_match.group(1)) if depth_match else 1
        
        # Check for stairs
        stairs_info = None
        if "stairs" in lang_obs.lower():
            stairs_info = {"type": "down", "nearby": True}
            
        # Check for doors
        doors_nearby = "door" in lang_obs.lower()
        
        # Try to find player position (simplified)
        player_pos = (10, 10)  # Default position
        
        return {
            "hp_current": hp_current,
            "hp_max": hp_max,
            "hp_percent": hp_current / hp_max if hp_max > 0 else 1.0,
            "depth": depth,
            "stairs_info": stairs_info,
            "doors_nearby": doors_nearby,
            "player_pos": player_pos,
            "turn": self.turn_count
        }
        
    def reset(self):
        """Reset agent for new episode"""
        self.prompt_builder.reset()
        self.turn_count = 0
        self.action_history.clear()
        self.position_history.clear()