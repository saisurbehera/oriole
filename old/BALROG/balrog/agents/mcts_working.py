"""
Working MCTS agent that finds correct downward stairs
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


class MCTSWorking(BaseAgent):
    """Working MCTS agent that finds correct stairs"""
    
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
        if state["on_down_stairs"]:
            action = "down"
            reasoning = f"On DOWN stairs at {state['player_pos']}! Going down"
        elif state["down_stairs_pos"]:
            # Navigate to down stairs
            action = self._navigate_to_stairs(state)
            dist = self._manhattan_distance(state["player_pos"], state["down_stairs_pos"])
            reasoning = f"Moving {action} to down stairs at {state['down_stairs_pos']} (distance: {dist})"
        else:
            # Use MCTS for exploration
            action = self._select_action_ucb(current_context, state)
            reasoning = f"MCTS exploring with {action} in {current_context}"
        
        # Store for next update
        self.last_obs = observation
        self.last_action = action
        
        return LLMResponse(
            model_id="mcts_working",
            completion=action,
            stop_reason="complete",
            input_tokens=50,
            output_tokens=10,
            reasoning=reasoning
        )
    
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Parse the game state from observation"""
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
        
        # Parse positions from TEXT MAP (this is the key fix!)
        player_pos = None
        down_stairs_pos = None
        on_down_stairs = False
        
        if "map:" in long_text:
            # Extract map text properly
            map_start = long_text.find("map:") + 5
            # Find the status line that ends the map
            status_markers = ["\\nAgent ", "\\nDlvl:", "\\n$", "\\nSt:"]
            map_end = len(long_text)
            for marker in status_markers:
                pos = long_text.find(marker, map_start)
                if pos > 0 and pos < map_end:
                    map_end = pos
                    
            map_text = long_text[map_start:map_end]
            map_lines = map_text.strip().split('\\n')
            
            # Parse positions from map
            for y, line in enumerate(map_lines[:21]):
                # Find player
                if '@' in line:
                    player_pos = (line.index('@'), y)
                    
                # Find DOWN stairs - look for '<' character
                if '<' in line:
                    down_stairs_pos = (line.index('<'), y)
                    
            # Check if player is on down stairs
            if player_pos and down_stairs_pos and player_pos == down_stairs_pos:
                on_down_stairs = True
        
        # Update position history
        if player_pos:
            self.last_positions.append(player_pos)
        
        # Parse language observation
        lang_obs = ""
        if "language observation:" in long_text:
            lang_start = long_text.find("language observation:") + 21
            lang_end = long_text.find("\\n\\n", lang_start)
            if lang_end > 0:
                lang_obs = long_text[lang_start:lang_end]
        
        # Parse features
        down_stairs_visible = down_stairs_pos is not None
        doors_nearby = "door" in lang_obs.lower()
        monsters_nearby = any(m in lang_obs.lower() for m in ["kobold", "goblin", "jackal", "rat", "newt", "dog", "cat"])
        in_room = "room" in lang_obs.lower()
        
        return {
            "hp_current": hp_current,
            "hp_max": hp_max,
            "depth": depth,
            "score": score,
            "player_pos": player_pos,
            "down_stairs_pos": down_stairs_pos,
            "on_down_stairs": on_down_stairs,
            "down_stairs_visible": down_stairs_visible,
            "doors_nearby": doors_nearby,
            "monsters_nearby": monsters_nearby,
            "in_room": in_room,
            "lang_obs": lang_obs
        }
    
    def _navigate_to_stairs(self, state: Dict[str, Any]) -> str:
        """Navigate toward down stairs position"""
        px, py = state["player_pos"]
        sx, sy = state["down_stairs_pos"]
        
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
            
        # Finding down stairs
        if curr_state["down_stairs_visible"] and not prev_state["down_stairs_visible"]:
            reward += 2.0
            
        # Being on down stairs
        if curr_state["on_down_stairs"]:
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
        
        if state["on_down_stairs"]:
            features.append("on_down_stairs")
        elif state["down_stairs_visible"]:
            features.append("down_stairs_visible")
            
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