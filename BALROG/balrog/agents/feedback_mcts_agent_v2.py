"""
Improved Feedback MCTS Agent with better stair navigation
"""

import math
import random
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json

from .base import BaseAgent
from balrog.client import LLMResponse


@dataclass
class ActionStats:
    """Statistics for an action in a given context"""
    total_reward: float = 0.0
    visit_count: int = 0
    
    @property
    def average_reward(self) -> float:
        return self.total_reward / self.visit_count if self.visit_count > 0 else 0.0
    
    def ucb_score(self, total_visits: int, c: float = 1.414) -> float:
        """Upper Confidence Bound for action selection"""
        if self.visit_count == 0:
            return float('inf')
        if total_visits == 0:
            return float('inf')
        
        exploitation = self.average_reward
        exploration = c * math.sqrt(math.log(total_visits + 1) / self.visit_count)
        return exploitation + exploration


class FeedbackMCTSAgentV2(BaseAgent):
    """Improved MCTS agent with better stair navigation"""
    
    def __init__(self, client_factory, prompt_builder, pretrained_values=None):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Action value estimates based on context
        if pretrained_values:
            self.action_values = pretrained_values
        else:
            self.action_values = defaultdict(lambda: defaultdict(ActionStats))
        
        # Game state tracking
        self.last_obs = None
        self.last_action = None
        self.last_score = 0
        self.last_hp = 16
        self.last_depth = 1
        self.last_hunger = "normal"
        
        # Context tracking
        self.turn_count = 0
        self.position_history = deque(maxlen=10)
        self.recent_rewards = deque(maxlen=20)
        
        # Stair navigation state
        self.stairs_position = None
        self.player_position = None
        self.navigating_to_stairs = False
        
        # MCTS parameters
        self.exploration_constant = 1.414
        
    def reset(self):
        """Reset agent state while preserving learned values"""
        self.last_obs = None
        self.last_action = None
        self.last_score = 0
        self.last_hp = 16
        self.last_depth = 1
        self.last_hunger = "normal"
        self.turn_count = 0
        self.position_history.clear()
        self.recent_rewards.clear()
        self.stairs_position = None
        self.player_position = None
        self.navigating_to_stairs = False
        # DO NOT reset action_values - we want to keep learning!
        
    def act(self, observation: Dict[str, Any], prev_action: str = None, return_reasoning: bool = False) -> LLMResponse:
        """Select action using learned values and feedback"""
        self.turn_count += 1
        
        # Update based on last action's result
        if self.last_obs is not None and self.last_action is not None:
            reward = self._calculate_reward(self.last_obs, observation)
            context = self._get_context(self.last_obs)
            self._update_action_value(context, self.last_action, reward)
            self.recent_rewards.append(reward)
        
        # Parse current state
        state = self._parse_game_state(observation)
        self.update_positions(state)
        
        # Get current context
        current_context = self._get_context(observation)
        
        # Choose action
        if self.navigating_to_stairs and self.stairs_position:
            # Special navigation mode
            action = self._navigate_to_position(self.stairs_position)
        else:
            # Normal UCB selection
            action = self._select_action_ucb(current_context, observation)
        
        # Store for next update
        self.last_obs = observation
        self.last_action = action
        
        return LLMResponse(
            model_id="feedback_mcts_v2",
            completion=action,
            stop_reason="complete",
            input_tokens=50,
            output_tokens=10,
            reasoning=self._get_reasoning(current_context, action)
        )
    
    def update_positions(self, state: Dict[str, Any]):
        """Update player and stairs positions from game state"""
        map_section = state.get("map_section", "")
        lines = map_section.strip().split('\n')[:21]
        
        player_y, player_x = None, None
        stairs_y, stairs_x = None, None
        
        for y, line in enumerate(lines):
            if '@' in line:
                player_x = line.index('@')
                player_y = y
            if '>' in line:
                stairs_x = line.index('>')
                stairs_y = y
                
        self.player_position = (player_x, player_y) if player_x is not None else None
        self.stairs_position = (stairs_x, stairs_y) if stairs_x is not None else None
        
        # Check if we found stairs
        if self.stairs_position and not self.navigating_to_stairs:
            self.navigating_to_stairs = True
            
        # Check if we're on the stairs
        if (self.player_position and self.stairs_position and 
            self.player_position == self.stairs_position):
            self.navigating_to_stairs = False  # We're there!
    
    def _navigate_to_position(self, target: tuple) -> str:
        """Navigate to a specific position"""
        if not self.player_position:
            return "wait"
            
        px, py = self.player_position
        tx, ty = target
        
        # Calculate direction
        dx = tx - px
        dy = ty - py
        
        # Choose movement
        if abs(dx) > abs(dy):
            # Move horizontally first
            if dx > 0:
                return "east"
            elif dx < 0:
                return "west"
        elif dy != 0:
            # Move vertically
            if dy > 0:
                return "south"
            elif dy < 0:
                return "north"
        elif dx != 0:
            # Same row, move horizontally
            if dx > 0:
                return "east"
            else:
                return "west"
        else:
            # We're at the target!
            return ">"
    
    def _calculate_reward(self, prev_obs: Dict, curr_obs: Dict) -> float:
        """Calculate reward based on state transition"""
        reward = 0.0
        
        # Parse states
        prev_state = self._parse_game_state(prev_obs)
        curr_state = self._parse_game_state(curr_obs)
        
        # Score change (normalized)
        score_change = curr_state["score"] - self.last_score
        if score_change > 0:
            reward += score_change / 100.0  # Normalize score rewards
        
        # HP change (penalize damage)
        hp_change = curr_state["hp_current"] - self.last_hp
        if hp_change < 0:
            reward += hp_change * 0.1  # -0.1 per HP lost
        elif hp_change > 0:
            reward += hp_change * 0.05  # +0.05 per HP gained
        
        # Depth change (big reward!)
        if curr_state["depth"] > self.last_depth:
            reward += 10.0  # Major reward for going deeper
            
        # Hunger penalty
        if curr_state.get("hunger") in ["Weak", "Fainting"]:
            reward -= 0.5
            
        # Movement penalty for being stuck
        if self.player_position and self.player_position in list(self.position_history)[-3:]:
            reward -= 0.1  # Small penalty for revisiting recent positions
            
        # Finding stairs bonus
        if curr_state.get("stairs_info") and not prev_state.get("stairs_info"):
            reward += 2.0  # Reward for discovering stairs
            
        # Being on stairs bonus
        if (self.player_position and self.stairs_position and 
            self.player_position == self.stairs_position):
            reward += 1.0  # Reward for reaching stairs
            
        # Update tracking
        self.last_score = curr_state["score"]
        self.last_hp = curr_state["hp_current"]
        self.last_depth = curr_state["depth"]
        if self.player_position:
            self.position_history.append(self.player_position)
        
        return reward
    
    def _get_context(self, obs: Dict) -> str:
        """Get context string for current state"""
        state = self._parse_game_state(obs)
        features = []
        
        # Key features for context
        if state.get("stairs_info"):
            # Check if we're on the stairs
            if (self.player_position and self.stairs_position and 
                self.player_position == self.stairs_position):
                features.append("on_stairs")
            else:
                features.append("stairs_visible")
                
        if state.get("doors_nearby"):
            features.append("door_adjacent")
            
        if state.get("monster_nearby"):
            features.append("monster_nearby")
            
        if state.get("room_type") == "room":
            features.append("in_room")
        elif state.get("room_type") == "corridor":
            features.append("in_corridor")
            
        # Check if stuck
        if len(self.recent_rewards) >= 10 and sum(list(self.recent_rewards)[-10:]) < 0.1:
            features.append("stuck")
            
        # Add depth as context - but not if we have other features
        if not features or len(features) <= 1:
            features.append(f"depth_{state.get('depth', 1)}")
        
        return "|".join(sorted(features)) if features else "exploring"
    
    def _update_action_value(self, context: str, action: str, reward: float):
        """Update action value estimates"""
        self.action_values[context][action].total_reward += reward
        self.action_values[context][action].visit_count += 1
    
    def _select_action_ucb(self, context: str, obs: Dict) -> str:
        """Select action using Upper Confidence Bound"""
        # Get possible actions
        possible_actions = self._get_possible_actions(obs)
        
        # Get total visits for this context
        total_visits = sum(
            stats.visit_count 
            for stats in self.action_values[context].values()
        )
        
        # Special case: no data for this context yet
        if total_visits == 0:
            # Ask LLM for initial guidance
            return self._get_llm_suggestion(context, possible_actions, obs)
        
        # Calculate UCB scores
        action_scores = {}
        for action in possible_actions:
            stats = self.action_values[context][action]
            action_scores[action] = stats.ucb_score(total_visits, self.exploration_constant)
        
        # Select action with highest UCB score
        best_action = max(action_scores, key=action_scores.get)
        
        # Occasionally ask LLM to verify our choice (10% of time)
        if random.random() < 0.1 and self.action_values[context][best_action].visit_count > 5:
            llm_action = self._get_llm_suggestion(context, possible_actions, obs)
            if llm_action != best_action:
                # LLM disagrees - maybe try it?
                if random.random() < 0.3:  # 30% chance to follow LLM
                    return llm_action
                    
        return best_action
    
    def _get_llm_suggestion(self, context: str, actions: List[str], obs: Dict) -> str:
        """Get LLM suggestion for action"""
        prompt = f"""Current context: {context}
Available actions: {', '.join(actions)}

Based on this NetHack situation, what's the best action? Respond with just the action."""

        try:
            response = self.client.query(
                prompt, 
                max_tokens=10,
                temperature=0.2
            )
            
            # Parse action from response
            for action in actions:
                if action in response.completion.lower():
                    return action
                    
        except Exception:
            pass
            
        # Fallback to random
        return random.choice(actions)
    
    def _get_possible_actions(self, obs: Dict) -> List[str]:
        """Get valid actions for current state"""
        state = self._parse_game_state(obs)
        
        # Check if we're on stairs
        if (self.player_position and self.stairs_position and 
            self.player_position == self.stairs_position):
            # We're on stairs! Only offer > action
            return [">"]
        
        actions = ["north", "south", "east", "west", "wait", "search"]
        
        # Add diagonal movements
        actions.extend(["northeast", "northwest", "southeast", "southwest"])
        
        # Add context-specific actions
        if state.get("stairs_info") and not self.navigating_to_stairs:
            actions.insert(0, ">")  # Priority!
            
        if state.get("doors_nearby"):
            actions.extend(["open", "kick"])
            
        return actions
    
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Parse observation into game state"""
        import re
        
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        lang_obs = text.get("language_observation", "")
        
        # Parse stats
        hp_match = re.search(r'HP:(\d+)\((\d+)\)', long_text)
        hp_current = int(hp_match.group(1)) if hp_match else 16
        hp_max = int(hp_match.group(2)) if hp_match else 16
        
        depth_match = re.search(r'Dlvl:(\d+)', long_text)
        depth = int(depth_match.group(1)) if depth_match else 1
        
        score_match = re.search(r'S:(\d+)', long_text)
        score = int(score_match.group(1)) if score_match else 0
        
        # Parse map to find stairs
        map_section = long_text.split("map:")[-1].split("Agent")[0] if "map:" in long_text else ""
        
        # Check features more carefully
        stairs_info = ("stairs" in lang_obs.lower() or 
                      "staircase" in lang_obs.lower() or
                      ">" in map_section)
        
        doors_nearby = ("|" in lang_obs or "-" in lang_obs or 
                       "door" in lang_obs.lower() or
                       "closed" in lang_obs.lower())
        
        # Detect room vs corridor
        room_type = "room" if "room" in lang_obs.lower() else "unknown"
        if "corridor" in lang_obs.lower() or "passage" in lang_obs.lower():
            room_type = "corridor"
        
        # Check for monsters
        monster_nearby = any(char in lang_obs for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        # Hunger status
        hunger = "normal"
        if "Weak" in long_text:
            hunger = "Weak"
        elif "Fainting" in long_text:
            hunger = "Fainting"
        
        return {
            "hp_current": hp_current,
            "hp_max": hp_max,
            "depth": depth,
            "score": score,
            "stairs_info": stairs_info,
            "doors_nearby": doors_nearby,
            "room_type": room_type,
            "monster_nearby": monster_nearby,
            "hunger": hunger,
            "map_section": map_section
        }
    
    def _get_reasoning(self, context: str, action: str) -> str:
        """Get reasoning for the selected action"""
        stats = self.action_values[context][action]
        
        if self.navigating_to_stairs:
            return f"Navigating to stairs at {self.stairs_position}"
        
        if stats.visit_count > 0:
            return f"Context: {context}, Action: {action}, Avg reward: {stats.average_reward:.2f}, Visits: {stats.visit_count}"
        else:
            return f"Exploring new action {action} in context {context}"