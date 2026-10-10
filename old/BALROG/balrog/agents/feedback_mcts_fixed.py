"""
Fixed Feedback MCTS Agent with proper position tracking
"""

import math
import random
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import re

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


class FeedbackMCTSFixed(BaseAgent):
    """Fixed MCTS agent with proper position tracking"""
    
    def __init__(self, client_factory, prompt_builder, pretrained_values=None):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Action value estimates based on context
        if pretrained_values:
            # Convert regular dict to defaultdict structure
            self.action_values = defaultdict(lambda: defaultdict(ActionStats))
            for context, actions in pretrained_values.items():
                for action, stats in actions.items():
                    self.action_values[context][action] = stats
        else:
            self.action_values = defaultdict(lambda: defaultdict(ActionStats))
        
        # Game state tracking
        self.last_obs = None
        self.last_action = None
        self.last_score = 0
        self.last_hp = 16
        self.last_depth = 1
        self.last_hunger = "normal"
        
        # Position tracking
        self.player_pos = None
        self.stairs_pos = None
        self.navigating_to_stairs = False
        
        # Context tracking
        self.turn_count = 0
        self.position_history = deque(maxlen=10)
        self.recent_rewards = deque(maxlen=20)
        self.exploration_constant = 1.414
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        
        # Don't reset learned values! Keep them across episodes
        self.last_obs = None
        self.last_action = None
        self.last_score = 0
        self.last_hp = 16
        self.last_depth = 1
        self.last_hunger = "normal"
        self.turn_count = 0
        self.position_history.clear()
        self.recent_rewards.clear()
        self.player_pos = None
        self.stairs_pos = None
        self.navigating_to_stairs = False
        
    def act(self, observation: Dict[str, Any], prev_action: str = None, return_reasoning: bool = False) -> LLMResponse:
        """Select action using learned values and exploration"""
        self.turn_count += 1
        
        # Calculate reward from last action
        if self.last_obs is not None and self.last_action is not None:
            reward = self._calculate_reward(self.last_obs, observation)
            context = self._get_context(self.last_obs)
            self._update_action_value(context, self.last_action, reward)
            self.recent_rewards.append(reward)
        
        # Parse current state and update positions
        state = self._parse_game_state(observation)
        self._update_positions(state)
        
        # Get context for current state
        current_context = self._get_context(observation)
        
        # Choose action
        if self.navigating_to_stairs and self.stairs_pos and self.player_pos:
            # Navigate to stairs if we know where they are
            action = self._navigate_to_position(self.stairs_pos)
            if self.player_pos == self.stairs_pos:
                self.navigating_to_stairs = False
                action = ">"  # Try going down
        else:
            # Normal UCB selection
            action = self._select_action_ucb(current_context, observation)
        
        # Store for next update
        self.last_obs = observation
        self.last_action = action
        
        return LLMResponse(
            model_id="feedback_mcts_fixed",
            completion=action,
            stop_reason="complete",
            input_tokens=50,
            output_tokens=10,
            reasoning=self._get_reasoning(current_context, action)
        )
    
    def _update_positions(self, state: Dict[str, Any]):
        """Update player and stairs positions from parsed state"""
        self.player_pos = state.get("player_pos")
        self.stairs_pos = state.get("stairs_pos")
        
        if self.player_pos:
            self.position_history.append(self.player_pos)
            
        # Start navigating if we found stairs
        if self.stairs_pos and not self.navigating_to_stairs:
            self.navigating_to_stairs = True
    
    def _navigate_to_position(self, target: Tuple[int, int]) -> str:
        """Navigate to a specific position"""
        if not self.player_pos:
            return "wait"
            
        px, py = self.player_pos
        tx, ty = target
        
        dx = tx - px
        dy = ty - py
        
        # Choose movement direction
        if abs(dx) > abs(dy):
            return "east" if dx > 0 else "west"
        elif dy != 0:
            return "south" if dy > 0 else "north"
        elif dx != 0:
            return "east" if dx > 0 else "west"
        else:
            # We're at the target
            return "wait"
    
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
        if curr_state.get("player_pos") and curr_state["player_pos"] in list(self.position_history)[-3:]:
            reward -= 0.1  # Small penalty for revisiting recent positions
            
        # Finding stairs bonus
        if curr_state.get("stairs_pos") and not prev_state.get("stairs_pos"):
            reward += 2.0  # Reward for discovering stairs
            
        # Being on stairs bonus
        if (curr_state.get("player_pos") and curr_state.get("stairs_pos") and 
            curr_state["player_pos"] == curr_state["stairs_pos"]):
            reward += 1.0  # Reward for reaching stairs
            
        # Update tracking
        self.last_score = curr_state["score"]
        self.last_hp = curr_state["hp_current"]
        self.last_depth = curr_state["depth"]
        
        return reward
    
    def _get_context(self, obs: Dict) -> str:
        """Get context string for current state"""
        state = self._parse_game_state(obs)
        features = []
        
        # Key features for context
        if state.get("stairs_pos"):
            # Check if we're on the stairs
            if state.get("player_pos") == state.get("stairs_pos"):
                features.append("on_stairs")
            else:
                features.append("stairs_visible")
                
        if state.get("doors_nearby"):
            features.append("door_adjacent")
            
        if state.get("monsters_nearby"):
            features.append("monster_nearby")
            
        if state.get("in_room"):
            features.append("in_room")
            
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
        actions = ["north", "south", "east", "west", "wait", "search"]
        
        # Add context-specific actions
        if state.get("stairs_pos"):
            # Only add > if we're on the stairs
            if state.get("player_pos") == state.get("stairs_pos"):
                actions.insert(0, ">")  # Priority!
            
        if state.get("doors_nearby"):
            actions.extend(["open", "kick"])
            
        # If stuck, add more exploration
        if "stuck" in self._get_context(obs):
            actions.extend(["northwest", "northeast", "southwest", "southeast"])
            
        return actions
    
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Parse observation into game state with proper position tracking"""
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
        
        # Parse map to find positions
        map_section = long_text.split("map:")[-1].split("Agent")[0] if "map:" in long_text else ""
        lines = map_section.strip().split('\n')[:21]  # NetHack map is 21 lines
        
        player_pos = None
        stairs_pos = None
        
        for y, line in enumerate(lines):
            if '@' in line:
                player_pos = (line.index('@'), y)
            if '>' in line:
                stairs_pos = (line.index('>'), y)
        
        # Check features
        doors_nearby = ("door" in lang_obs.lower() or "+" in lang_obs)
        monsters_nearby = any(m in lang_obs.lower() for m in ["kobold", "goblin", "jackal", "rat", "newt"])
        
        # Check for walls/rooms
        in_room = "|" in map_section and "-" in map_section
        
        # Check hunger
        hunger = "normal"
        if "Weak" in long_text:
            hunger = "Weak"
        elif "Fainting" in long_text:
            hunger = "Fainting"
        
        return {
            "hp_current": hp_current,
            "hp_max": hp_max,
            "hp_percent": hp_current / hp_max if hp_max > 0 else 1.0,
            "depth": depth,
            "score": score,
            "hunger": hunger,
            "stairs_pos": stairs_pos,
            "doors_nearby": doors_nearby,
            "monsters_nearby": monsters_nearby,
            "in_room": in_room,
            "player_pos": player_pos,
            "lang_obs": lang_obs,
            "map_section": map_section[:200]  # For debugging
        }
    
    def _get_reasoning(self, context: str, action: str) -> str:
        """Get reasoning for action choice"""
        if self.navigating_to_stairs:
            return f"Navigating to stairs at {self.stairs_pos}"
            
        if context in self.action_values and action in self.action_values[context]:
            stats = self.action_values[context][action]
            if stats.visit_count > 0:
                return f"Context: {context[:30]}... | {action} (avg reward: {stats.average_reward:.2f})"
        return f"Exploring: {action} in context {context[:30]}..."