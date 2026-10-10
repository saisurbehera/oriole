"""
Corrected MCTS agent that uses proper glyph coordinates
"""

import math
import random
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import re
import numpy as np

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


class MCTSCorrected(BaseAgent):
    """MCTS agent with corrected position parsing"""
    
    def __init__(self, client_factory, prompt_builder, pretrained_values=None):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Action value estimates
        if pretrained_values:
            self.action_values = defaultdict(lambda: defaultdict(ActionStats))
            for context, actions in pretrained_values.items():
                for action, stats in actions.items():
                    self.action_values[context][action] = stats
        else:
            self.action_values = defaultdict(lambda: defaultdict(ActionStats))
        
        # State tracking
        self.last_obs = None
        self.last_action = None
        self.last_score = 0
        self.last_hp = 16
        self.last_depth = 1
        self.turn_count = 0
        self.recent_rewards = deque(maxlen=20)
        self.last_positions = deque(maxlen=10)
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.last_obs = None
        self.last_action = None
        self.last_score = 0
        self.last_hp = 16
        self.last_depth = 1
        self.turn_count = 0
        self.recent_rewards.clear()
        self.last_positions.clear()
        
    def act(self, observation: Dict[str, Any], prev_action: str = None, return_reasoning: bool = False) -> LLMResponse:
        """Select action using MCTS"""
        self.turn_count += 1
        
        # Update based on last action
        if self.last_obs is not None and self.last_action is not None:
            reward = self._calculate_reward(self.last_obs, observation)
            context = self._get_context(self.last_obs)
            self._update_action_value(context, self.last_action, reward)
            self.recent_rewards.append(reward)
        
        # Parse current state
        state = self._parse_game_state(observation)
        
        # Get context
        current_context = self._get_context_from_state(state)
        
        # Choose action
        if state["on_stairs"]:
            action = "down"
            reasoning = f"On stairs at {state['player_pos']}! Going down"
        elif state["stairs_pos"]:
            # Navigate to stairs
            action = self._navigate_to_stairs(state)
            dist = self._manhattan_distance(state["player_pos"], state["stairs_pos"])
            reasoning = f"Moving {action} to stairs at {state['stairs_pos']} (distance: {dist})"
        else:
            # Use MCTS for exploration
            action = self._select_action_ucb(current_context, state)
            reasoning = f"MCTS selected {action} in {current_context}"
        
        # Store for next update
        self.last_obs = observation
        self.last_action = action
        
        return LLMResponse(
            model_id="mcts_corrected",
            completion=action,
            stop_reason="complete",
            input_tokens=50,
            output_tokens=10,
            reasoning=reasoning
        )
    
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Parse the game state from observation using proper coordinates"""
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        
        # Parse stats from text
        hp_current = 16
        hp_max = 16
        depth = 1
        score = 0
        
        hp_match = re.search(r'HP:(\d+)\((\d+)\)', long_text)
        if hp_match:
            hp_current = int(hp_match.group(1))
            hp_max = int(hp_match.group(2))
            
        depth_match = re.search(r'Dlvl:(\d+)', long_text)
        if depth_match:
            depth = int(depth_match.group(1))
            
        score_match = re.search(r'S:(\d+)', long_text)
        if score_match:
            score = int(score_match.group(1))
        
        # Parse positions from GLYPHS (correct coordinates)
        player_pos = None
        stairs_pos = None
        on_stairs = False
        
        if "obs" in obs and "glyphs" in obs["obs"]:
            glyphs = obs["obs"]["glyphs"]
            
            # Find player position (should be at cursor position)
            if "tty_cursor" in obs["obs"]:
                cursor = obs["obs"]["tty_cursor"]
                # Cursor is (y, x) format
                player_pos = (cursor[1], cursor[0])
            
            # Find stairs glyph
            stairs_glyphs = np.where(glyphs == 2378)  # '>' glyph
            if len(stairs_glyphs[0]) > 0:
                # Take the first stairs found
                stairs_pos = (stairs_glyphs[1][0], stairs_glyphs[0][0])
                
                # Check if any stairs position matches player position
                for i in range(len(stairs_glyphs[0])):
                    stairs_y, stairs_x = stairs_glyphs[0][i], stairs_glyphs[1][i]
                    if player_pos and player_pos == (stairs_x, stairs_y):
                        on_stairs = True
                        stairs_pos = (stairs_x, stairs_y)
                        break
        
        # Update position history
        if player_pos:
            self.last_positions.append(player_pos)
        
        # Parse language observation
        lang_obs = ""
        if "language observation:" in long_text:
            lang_start = long_text.find("language observation:") + 21
            lang_end = long_text.find("\n\n", lang_start)
            if lang_end > 0:
                lang_obs = long_text[lang_start:lang_end]
        
        # Parse features
        stairs_visible = stairs_pos is not None
        doors_nearby = "door" in lang_obs.lower()
        monsters_nearby = any(m in lang_obs.lower() for m in ["kobold", "goblin", "jackal", "rat", "newt", "dog", "cat"])
        in_room = "room" in lang_obs.lower()
        
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
    
    def _navigate_to_stairs(self, state: Dict[str, Any]) -> str:
        """Navigate toward stairs position"""
        px, py = state["player_pos"]
        sx, sy = state["stairs_pos"]
        
        dx = sx - px
        dy = sy - py
        
        # Prefer cardinal directions
        if abs(dx) > abs(dy):
            return "east" if dx > 0 else "west"
        elif dy != 0:
            return "south" if dy > 0 else "north"
        else:
            return "east" if dx > 0 else "west"
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _calculate_reward(self, prev_obs: Dict, curr_obs: Dict) -> float:
        """Calculate reward for state transition"""
        prev_state = self._parse_game_state(prev_obs)
        curr_state = self._parse_game_state(curr_obs)
        
        reward = 0.0
        
        # Major reward for going deeper
        if curr_state["depth"] > prev_state["depth"]:
            reward += 10.0
            
        # Score change
        score_diff = curr_state["score"] - prev_state["score"]
        if score_diff > 0:
            reward += score_diff / 100.0
            
        # HP change
        hp_diff = curr_state["hp_current"] - prev_state["hp_current"]
        if hp_diff < 0:
            reward += hp_diff * 0.1
        elif hp_diff > 0:
            reward += hp_diff * 0.05
            
        # Finding stairs
        if curr_state["stairs_visible"] and not prev_state["stairs_visible"]:
            reward += 2.0
            
        # Being on stairs
        if curr_state["on_stairs"]:
            reward += 1.0
            
        # Stuck penalty
        if len(self.last_positions) >= 3:
            recent = list(self.last_positions)[-3:]
            if curr_state["player_pos"] and curr_state["player_pos"] in recent:
                reward -= 0.1
                
        return reward
    
    def _get_context_from_state(self, state: Dict[str, Any]) -> str:
        """Get context string from parsed state"""
        features = []
        
        if state["on_stairs"]:
            features.append("on_stairs")
        elif state["stairs_visible"]:
            features.append("stairs_visible")
            
        if state["doors_nearby"]:
            features.append("doors")
            
        if state["monsters_nearby"]:
            features.append("monsters")
            
        if state["in_room"]:
            features.append("room")
            
        # Check if stuck
        if len(self.recent_rewards) >= 10:
            recent_rewards = list(self.recent_rewards)[-10:]
            if sum(recent_rewards) < 0.1:
                features.append("stuck")
                
        if not features:
            features.append(f"depth{state['depth']}")
            
        return "|".join(sorted(features))
    
    def _get_context(self, obs: Dict) -> str:
        """Get context from raw observation"""
        state = self._parse_game_state(obs)
        return self._get_context_from_state(state)
    
    def _update_action_value(self, context: str, action: str, reward: float):
        """Update action value estimate"""
        self.action_values[context][action].total_reward += reward
        self.action_values[context][action].visit_count += 1
    
    def _select_action_ucb(self, context: str, state: Dict[str, Any]) -> str:
        """Select action using UCB"""
        # Get possible actions
        actions = ["north", "south", "east", "west", "wait", "search"]
        
        if state["doors_nearby"]:
            actions.extend(["open", "kick"])
            
        if "stuck" in context:
            actions.extend(["northeast", "northwest", "southeast", "southwest"])
        
        # Get total visits
        total_visits = sum(
            stats.visit_count 
            for stats in self.action_values[context].values()
        )
        
        # If no data, use random
        if total_visits == 0:
            return random.choice(actions)
        
        # Calculate UCB scores
        scores = {}
        for action in actions:
            stats = self.action_values[context][action]
            scores[action] = stats.ucb_score(total_visits)
        
        # Select best
        return max(scores, key=scores.get)