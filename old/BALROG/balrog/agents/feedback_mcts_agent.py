"""
Feedback-Based MCTS Agent - Learns from actual game rewards
===========================================================

This agent uses MCTS but learns action values from real game feedback
rather than trying to simulate future states. It tracks:
- Score changes after actions
- HP changes (negative reward for damage)
- Depth changes (big positive reward)
- Hunger/status effects
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

from balrog.agents.base import BaseAgent
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


class FeedbackMCTSAgent(BaseAgent):
    """MCTS agent that learns from actual game feedback"""
    
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
        
        # Context tracking
        self.turn_count = 0
        self.position_history = deque(maxlen=10)
        self.recent_rewards = deque(maxlen=20)
        
        # MCTS parameters
        self.exploration_constant = 2.0
        self.context_features = ["stairs_visible", "door_adjacent", "monster_near", "low_hp", "stuck"]
        
    def act(self, obs, prev_action=None):
        """Select action using learned values and exploration"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        self.turn_count += 1
        
        # Calculate reward from previous action
        if self.last_obs is not None and self.last_action is not None:
            reward = self._calculate_reward(self.last_obs, obs)
            context = self._get_context(self.last_obs)
            
            # Update action values
            self._update_action_value(context, self.last_action, reward)
            self.recent_rewards.append(reward)
        
        # Get current context
        current_context = self._get_context(obs)
        
        # Select action using UCB
        action = self._select_action_ucb(current_context, obs)
        
        # Store for next update
        self.last_obs = obs
        self.last_action = action
        
        return LLMResponse(
            model_id="feedback_mcts",
            completion=action,
            stop_reason="complete",
            input_tokens=50,
            output_tokens=10,
            reasoning=self._get_reasoning(current_context, action)
        )
    
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
        if curr_state.get("player_pos") in list(self.position_history)[-3:]:
            reward -= 0.1  # Small penalty for revisiting recent positions
            
        # Finding stairs bonus
        if curr_state.get("stairs_info") and not prev_state.get("stairs_info"):
            reward += 2.0  # Reward for discovering stairs
            
        # Update tracking
        self.last_score = curr_state["score"]
        self.last_hp = curr_state["hp_current"]
        self.last_depth = curr_state["depth"]
        if curr_state.get("player_pos"):
            self.position_history.append(curr_state["player_pos"])
            
        return reward
    
    def _get_context(self, obs: Dict) -> str:
        """Extract context features for state"""
        state = self._parse_game_state(obs)
        
        features = []
        
        # Check each feature
        if state.get("stairs_info"):
            features.append("stairs_visible")
            
        if state.get("doors_nearby"):
            features.append("door_adjacent")
            
        if state.get("monsters_nearby"):
            features.append("monster_near")
            
        if state.get("hp_percent", 1.0) < 0.5:
            features.append("low_hp")
            
        if state.get("in_room"):
            features.append("in_room")
            
        # Check if stuck
        if len(set(self.position_history)) <= 2 and len(self.position_history) >= 5:
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
        if random.random() < 0.1 and stats.visit_count > 5:
            llm_action = self._get_llm_suggestion(context, possible_actions, obs)
            if llm_action != best_action:
                # LLM disagrees - maybe try it?
                if random.random() < 0.3:  # 30% chance to follow LLM
                    return llm_action
                    
        return best_action
    
    def _get_llm_suggestion(self, context: str, actions: List[str], obs: Dict) -> str:
        """Get LLM suggestion for action"""
        try:
            state = self._parse_game_state(obs)
            
            # Build context description
            context_desc = context.replace("|", ", ").replace("_", " ")
            
            prompt = f"""NetHack situation:
Context: {context_desc}
Recent rewards: {sum(self.recent_rewards):.1f}
Available: {', '.join(actions[:8])}

What action is best? Reply with just the action."""

            messages = [{"role": "user", "content": prompt}]
            response = self.client.generate(messages)
            
            # Extract action
            response_text = response.completion.strip().lower()
            for action in actions:
                if action in response_text:
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
        if state.get("stairs_info"):
            actions.insert(0, ">")  # Priority!
            
        if state.get("doors_nearby"):
            actions.extend(["open", "kick"])
            
        # If stuck, add more exploration
        if "stuck" in self._get_context(obs):
            actions.extend(["northwest", "northeast", "southwest", "southeast"])
            
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
                      ">" in map_section or 
                      "<" in map_section)
        doors_nearby = ("door" in lang_obs.lower() or 
                       "+" in map_section)
        monsters_nearby = any(m in lang_obs.lower() for m in ["kobold", "goblin", "jackal", "rat", "newt"])
        
        # Check for walls/rooms
        in_room = "|" in map_section and "-" in map_section
        
        # Simple position tracking (would need proper parsing)
        player_pos = (self.turn_count % 20, self.turn_count % 20)  # Placeholder
        
        return {
            "hp_current": hp_current,
            "hp_max": hp_max,
            "hp_percent": hp_current / hp_max if hp_max > 0 else 1.0,
            "depth": depth,
            "score": score,
            "stairs_info": stairs_info,
            "doors_nearby": doors_nearby,
            "monsters_nearby": monsters_nearby,
            "in_room": in_room,
            "player_pos": player_pos,
            "lang_obs": lang_obs,
            "map_section": map_section[:200]  # For debugging
        }
    
    def _get_reasoning(self, context: str, action: str) -> str:
        """Get reasoning for action choice"""
        if context in self.action_values and action in self.action_values[context]:
            stats = self.action_values[context][action]
            if stats.visit_count > 0:
                return f"Context: {context[:30]}... | {action} (avg reward: {stats.average_reward:.2f})"
        return f"Exploring: {action} in context {context[:30]}..."
    
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        
        # Don't reset learned values! Keep them across episodes
        # Only reset episode-specific tracking
        self.last_obs = None
        self.last_action = None
        self.last_score = 0
        self.last_hp = 16
        self.last_depth = 1
        self.turn_count = 0
        self.position_history.clear()
        self.recent_rewards.clear()
        
    def save_values(self, filepath: str):
        """Save learned action values to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.action_values), f)
            
    def load_values(self, filepath: str):
        """Load learned action values from file"""
        import pickle
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
            # Convert back to defaultdict
            self.action_values = defaultdict(lambda: defaultdict(ActionStats))
            for context, actions in loaded.items():
                for action, stats in actions.items():
                    self.action_values[context][action] = stats