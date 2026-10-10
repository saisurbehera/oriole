"""
Working Feedback MCTS Agent that correctly parses NetHack observations
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


class FeedbackMCTSWorking(BaseAgent):
    """MCTS agent that correctly parses NetHack text map"""
    
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
        
        # Position tracking
        self.player_pos = None
        self.stairs_pos = None
        self.last_positions = deque(maxlen=10)
        
        # Context tracking
        self.turn_count = 0
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
        self.turn_count = 0
        self.player_pos = None
        self.stairs_pos = None
        self.last_positions.clear()
        self.recent_rewards.clear()
        
    def act(self, observation: Dict[str, Any], prev_action: str = None, return_reasoning: bool = False) -> LLMResponse:
        """Select action using learned values and exploration"""
        self.turn_count += 1
        
        # Calculate reward from last action
        if self.last_obs is not None and self.last_action is not None:
            reward = self._calculate_reward(self.last_obs, observation)
            context = self._get_context(self.last_obs)
            self._update_action_value(context, self.last_action, reward)
            self.recent_rewards.append(reward)
        
        # Parse current state
        state = self._parse_game_state(observation)
        
        # Get context for current state
        current_context = self._get_context(observation)
        
        # Choose action based on position
        if state["on_stairs"]:
            # We're on stairs! Go down!
            action = ">"
        elif state["stairs_pos"] and state["player_pos"]:
            # Navigate to stairs
            action = self._navigate_to_position(state["stairs_pos"], state["player_pos"])
        else:
            # Normal UCB selection for exploration
            action = self._select_action_ucb(current_context, observation)
        
        # Store for next update
        self.last_obs = observation
        self.last_action = action
        
        return LLMResponse(
            model_id="feedback_mcts_working",
            completion=action,
            stop_reason="complete",
            input_tokens=50,
            output_tokens=10,
            reasoning=self._get_reasoning(current_context, action, state)
        )
    
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Parse observation correctly using text map"""
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        
        # Parse stats
        hp_match = re.search(r'HP:(\d+)\((\d+)\)', long_text)
        hp_current = int(hp_match.group(1)) if hp_match else 16
        hp_max = int(hp_match.group(2)) if hp_match else 16
        
        depth_match = re.search(r'Dlvl:(\d+)', long_text)
        depth = int(depth_match.group(1)) if depth_match else 1
        
        score_match = re.search(r'S:(\d+)', long_text)
        score = int(score_match.group(1)) if score_match else 0
        
        # Parse positions from TEXT MAP, not glyphs!
        player_pos = None
        stairs_pos = None
        on_stairs = False
        
        if "map:" in long_text:
            map_start = long_text.index("map:") + 5  # Skip "map:\n"
            
            # Find where the status line starts (has "St:" in it)
            map_end = long_text.find("\nAgent", map_start)
            if map_end == -1:
                map_end = long_text.find("\nDlvl:", map_start)
            
            if map_end > map_start:
                map_section = long_text[map_start:map_end]
            else:
                map_section = long_text[map_start:]
            
            map_lines = map_section.strip().split('\n')[:21]  # Max 21 lines
            
            # Find positions in text map
            for y, line in enumerate(map_lines):
                if y >= 21:  # NetHack map is 21 lines
                    break
                    
                # Find player
                if '@' in line:
                    player_pos = (line.index('@'), y)
                    
                # Find stairs
                if '>' in line:
                    x = line.index('>')
                    stairs_pos = (x, y)
                    
                    # Check if player is on stairs
                    if player_pos and player_pos == (x, y):
                        on_stairs = True
        
        # Update position tracking
        if player_pos:
            self.player_pos = player_pos
            self.last_positions.append(player_pos)
        if stairs_pos:
            self.stairs_pos = stairs_pos
        
        # Parse language observation
        lang_obs = ""
        if "language observation:" in long_text:
            lang_start = long_text.index("language observation:")
            lang_section = long_text[lang_start:].split("\n\n")[0]
            lang_obs = lang_section.replace("language observation:\n", "")
        
        # Check features
        stairs_visible = "stairs" in lang_obs.lower() or stairs_pos is not None
        doors_nearby = "door" in lang_obs.lower()
        monsters_nearby = any(m in lang_obs.lower() for m in ["kobold", "goblin", "jackal", "rat", "newt"])
        in_room = "room" in lang_obs.lower()
        
        # Update tracking
        self.last_score = score
        self.last_hp = hp_current
        self.last_depth = depth
        
        return {
            "hp_current": hp_current,
            "hp_max": hp_max,
            "depth": depth,
            "score": score,
            "player_pos": player_pos,
            "stairs_pos": stairs_pos,
            "on_stairs": on_stairs,
            "stairs_visible": stairs_visible,
            "doors_nearby": doors_nearby,
            "monsters_nearby": monsters_nearby,
            "in_room": in_room,
            "lang_obs": lang_obs
        }
    
    def _navigate_to_position(self, target: Tuple[int, int], current: Tuple[int, int]) -> str:
        """Navigate to target position"""
        tx, ty = target
        cx, cy = current
        
        dx = tx - cx
        dy = ty - cy
        
        # Prefer cardinal directions
        if abs(dx) > abs(dy):
            return "east" if dx > 0 else "west"
        elif dy != 0:
            return "south" if dy > 0 else "north"
        elif dx != 0:
            return "east" if dx > 0 else "west"
        else:
            # We're at target but not detecting it properly
            return "wait"
    
    def _calculate_reward(self, prev_obs: Dict, curr_obs: Dict) -> float:
        """Calculate reward based on state transition"""
        prev_state = self._parse_game_state(prev_obs)
        curr_state = self._parse_game_state(curr_obs)
        
        reward = 0.0
        
        # Depth change (big reward!)
        if curr_state["depth"] > prev_state["depth"]:
            reward += 10.0  # SUCCESS!
            
        # Score change
        score_change = curr_state["score"] - prev_state["score"]
        if score_change > 0:
            reward += score_change / 100.0
        
        # HP change
        hp_change = curr_state["hp_current"] - prev_state["hp_current"]
        if hp_change < 0:
            reward += hp_change * 0.1
        elif hp_change > 0:
            reward += hp_change * 0.05
            
        # Finding stairs bonus
        if curr_state["stairs_visible"] and not prev_state["stairs_visible"]:
            reward += 2.0
            
        # Being on stairs bonus
        if curr_state["on_stairs"]:
            reward += 1.0
            
        # Movement penalty if stuck
        if curr_state["player_pos"] and len(self.last_positions) >= 3:
            if curr_state["player_pos"] in list(self.last_positions)[-3:]:
                reward -= 0.1
        
        return reward
    
    def _get_context(self, obs: Dict) -> str:
        """Get context string for current state"""
        state = self._parse_game_state(obs)
        features = []
        
        # Key features
        if state["on_stairs"]:
            features.append("on_stairs")
        elif state["stairs_visible"]:
            features.append("stairs_visible")
            
        if state["doors_nearby"]:
            features.append("door_adjacent")
            
        if state["monsters_nearby"]:
            features.append("monster_nearby")
            
        if state["in_room"]:
            features.append("in_room")
            
        # Check if stuck
        if len(self.recent_rewards) >= 10 and sum(list(self.recent_rewards)[-10:]) < 0.1:
            features.append("stuck")
            
        # Add depth if no other features
        if not features:
            features.append(f"depth_{state['depth']}")
        
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
        
        # Occasionally ask LLM to verify (10% of time)
        if random.random() < 0.1 and self.action_values[context][best_action].visit_count > 5:
            llm_action = self._get_llm_suggestion(context, possible_actions, obs)
            if llm_action != best_action and random.random() < 0.3:
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
        
        # If on stairs, only offer down action
        if state["on_stairs"]:
            return [">"]
        
        actions = ["north", "south", "east", "west", "wait", "search"]
        
        # Add context-specific actions
        if state["doors_nearby"]:
            actions.extend(["open", "kick"])
            
        # If stuck, add diagonals
        if "stuck" in self._get_context(obs):
            actions.extend(["northwest", "northeast", "southwest", "southeast"])
            
        return actions
    
    def _get_reasoning(self, context: str, action: str, state: Dict[str, Any]) -> str:
        """Get reasoning for action choice"""
        if state["on_stairs"]:
            return f"On stairs at {state['player_pos']}! Going down!"
        elif state["stairs_pos"] and state["player_pos"]:
            distance = abs(state["stairs_pos"][0] - state["player_pos"][0]) + abs(state["stairs_pos"][1] - state["player_pos"][1])
            return f"Moving {action} toward stairs (distance: {distance})"
        elif context in self.action_values and action in self.action_values[context]:
            stats = self.action_values[context][action]
            if stats.visit_count > 0:
                return f"{context} | {action} (avg: {stats.average_reward:.2f}, n={stats.visit_count})"
        return f"Exploring: {action} in {context}"