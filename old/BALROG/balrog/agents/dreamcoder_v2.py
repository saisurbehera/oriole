"""Improved DreamCoder agent with better action generation and learning"""

import json
import os
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse


@dataclass
class ProgramAbstraction:
    """Learned abstraction/sub-program"""
    name: str
    pattern: tuple  # Sequence of actions (tuple for hashability)
    context_conditions: tuple  # Conditions as tuple of (key, value) pairs
    success_rate: float = 0.0
    usage_count: int = 0
    
    def __hash__(self):
        """Make hashable based on name and pattern"""
        return hash((self.name, self.pattern))
    
    def matches_context(self, context: Dict) -> bool:
        """Check if current context matches this abstraction's conditions"""
        for key, pattern in self.context_conditions:
            if key not in context:
                return False
            if not re.search(pattern, str(context[key]), re.IGNORECASE):
                return False
        return True


class DreamCoderAgentV2(BaseAgent):
    """Improved DreamCoder agent with better NetHack understanding"""
    
    def __init__(self, client_factory, prompt_builder, 
                 memory_size: int = 1000,
                 repetition_penalty_window: int = 20,
                 library_save_path: Optional[str] = None):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.memory_size = memory_size
        
        # Use absolute path to avoid Hydra working directory issues
        if library_save_path and not os.path.isabs(library_save_path):
            library_save_path = os.path.abspath(library_save_path)
        self.library_save_path = library_save_path or os.path.abspath("dreamcoder_library_v2.json")
        
        # Library of learned abstractions
        self.library: List[ProgramAbstraction] = []
        
        # Memory of successful episodes
        self.episode_memory: List[Dict] = []
        
        # Current episode trace
        self.current_trace: List[Tuple[Dict, str, float]] = []
        self.episode_count = 0
        
        # For repetition penalty
        self.action_history = deque(maxlen=repetition_penalty_window)
        self.repetition_penalty_factor = 0.5
        
        # Track abstraction execution
        self.current_abstraction: Optional[ProgramAbstraction] = None
        self.abstraction_step = 0
        
        # Track failed actions
        self.failed_action_count = 0
        
        # Initialize with better primitives
        self._initialize_primitives()
        self._load_library_if_exists()
        
    def _initialize_primitives(self):
        """Initialize with NetHack-specific primitives"""
        # Basic patterns
        self.library.append(ProgramAbstraction(
            name="explore_room",
            pattern=("east", "north", "west", "south"),
            context_conditions=(("location_type", "room"), ("no_stairs", "true")),
            success_rate=0.6
        ))
        
        # Expert patterns from successful trajectories
        expert_patterns = [
            # Early game exploration - proven effective pattern
            ProgramAbstraction(
                name="expert_early_exploration",
                pattern=("search", "open", "north", "search", "east"),
                context_conditions=(("location_type", "room"), ("explored_percent", "low")),
                success_rate=0.8
            ),
            # Corridor navigation - avoids dead ends
            ProgramAbstraction(
                name="expert_corridor_navigation", 
                pattern=("north", "north", "search", "north"),
                context_conditions=(("location_type", "corridor"),),
                success_rate=0.75
            ),
            # Door breaking sequence - handles locked doors
            ProgramAbstraction(
                name="expert_door_sequence",
                pattern=("kick", "kick", "open", "north"),
                context_conditions=(("door_present", "true"),),
                success_rate=0.8
            ),
            # Combat kiting - survive fights
            ProgramAbstraction(
                name="expert_combat_retreat",
                pattern=("wait", "kick", "south", "wait"),
                context_conditions=(("enemy_adjacent", "true"),),
                success_rate=0.7
            ),
            # Intensive stair search
            ProgramAbstraction(
                name="expert_stair_search",
                pattern=("search", "search", "search", "east", "search"),
                context_conditions=(("explored_percent", "high"), ("no_stairs", "true")),
                success_rate=0.75
            ),
            # Diagonal exploration - efficient coverage
            ProgramAbstraction(
                name="expert_diagonal_explore",
                pattern=("northeast", "search", "southwest", "search"),
                context_conditions=(("location_type", "room"),),
                success_rate=0.7
            ),
            # Pickup items when found
            ProgramAbstraction(
                name="pickup_items",
                pattern=("pickup",),
                context_conditions=(("items_on_ground", "true"),),
                success_rate=0.9
            )
        ]
        
        # Add all expert patterns
        for pattern in expert_patterns:
            self.library.append(pattern)
        
    def act(self, obs, prev_action=None):
        """Generate action with better understanding of game state"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
            
        self.prompt_builder.update_observation(obs)
        
        # Extract context from observation
        context = self._extract_context(obs)
        
        # Check if we're in the middle of executing an abstraction
        if self.current_abstraction and self.abstraction_step < len(self.current_abstraction.pattern):
            action = self.current_abstraction.pattern[self.abstraction_step]
            self.abstraction_step += 1
            
            # If abstraction is complete, clear it
            if self.abstraction_step >= len(self.current_abstraction.pattern):
                self.current_abstraction = None
                self.abstraction_step = 0
        else:
            # Select new abstraction or use improved LLM
            applicable_abstractions = [
                abstr for abstr in self.library 
                if abstr.matches_context(context) and abstr.success_rate > 0.4
            ]
            
            if applicable_abstractions and self.failed_action_count < 5:
                # Apply repetition penalty
                abstraction_scores = {}
                for abstr in applicable_abstractions:
                    base_score = abstr.success_rate
                    
                    # Penalize if recently used
                    recent_actions = list(self.action_history)[-len(abstr.pattern):]
                    if len(recent_actions) >= len(abstr.pattern) and recent_actions == list(abstr.pattern):
                        penalty = self.repetition_penalty_factor
                    else:
                        penalty = 1.0
                        
                    abstraction_scores[abstr] = base_score * penalty
                
                # Select best abstraction
                self.current_abstraction = max(abstraction_scores, key=abstraction_scores.get)
                self.abstraction_step = 0
                self.current_abstraction.usage_count += 1
                
                # Get first action from abstraction
                action = self.current_abstraction.pattern[0]
                self.abstraction_step = 1
            else:
                # Use improved LLM action selection
                action = self._get_smart_llm_action(obs, context)
                
        # Update history and trace
        self.action_history.append(action)
        reward = self._estimate_immediate_reward(obs, action, context)
        self.current_trace.append((context, action, reward))
        
        # Create response object
        return LLMResponse(
            model_id="dreamcoder_v2",
            completion=action,
            stop_reason="complete",
            input_tokens=0,
            output_tokens=0,
            reasoning=f"Using abstraction: {self.current_abstraction.name}" if self.current_abstraction else f"LLM action (context: {context.get('location_type', 'unknown')})"
        )
        
    def _get_smart_llm_action(self, obs, context):
        """Smarter LLM action selection"""
        messages = self.prompt_builder.get_prompt()
        
        # Build context-aware action set
        smart_actions = []
        
        # Movement actions based on context
        if context.get("stairs_visible") == "true":
            smart_actions.extend(["down"])  # Go down stairs
        elif context.get("door_present") == "true":
            smart_actions.extend(["open", "kick"])
        elif context.get("enemy_adjacent") == "true":
            smart_actions.extend(["wait", "kick"])  # Combat
        else:
            # Exploration - avoid up unless stairs are visible
            smart_actions.extend(["north", "south", "east", "west", "search"])
            
        # Add wait as fallback
        if "wait" not in smart_actions:
            smart_actions.append("wait")
            
        # Build smarter prompt
        action_prompt = f"""
You are in a dungeon. Choose the BEST action for this situation.

Context:
- Location: {context.get('location_type', 'unknown')}
- Stairs visible: {context.get('stairs_visible', 'no')}
- Enemy nearby: {context.get('enemy_adjacent', 'no')}
- Door present: {context.get('door_present', 'no')}

Available actions: {', '.join(smart_actions)}

Important:
- Only use 'down' if you see stairs (>)
- Use 'open' or 'kick' for closed doors (+)
- Use 'wait' if a monster is nearby but not adjacent
- Use movement (north/south/east/west) to explore

Output ONLY the action name, nothing else.
        """.strip()
        
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + action_prompt
            
        try:
            response = self.client.generate(messages)
            action = response.completion.strip().lower()
            
            # Validate action
            if action not in smart_actions:
                # Find best match
                for valid_action in smart_actions:
                    if valid_action in action or action in valid_action:
                        action = valid_action
                        break
                else:
                    # Default to safe exploration
                    action = smart_actions[0] if smart_actions else "wait"
                    self.failed_action_count += 1
            else:
                self.failed_action_count = 0
                
        except Exception as e:
            print(f"[DreamCoder] LLM error: {e}, using fallback")
            action = "wait"
            
        return action
        
    def _extract_context(self, obs: Dict) -> Dict:
        """Extract detailed context from observation"""
        context = {}
        text = obs.get("text", {})
        long_term = text.get("long_term_context", "").lower()
        short_term = text.get("short_term_context", "").lower()
        
        # Location analysis
        if "room" in long_term or "chamber" in long_term:
            context["location_type"] = "room"
        elif "corridor" in long_term or "passage" in long_term:
            context["location_type"] = "corridor"
        else:
            context["location_type"] = "unknown"
            
        # Stairs detection - look for > symbol
        context["stairs_visible"] = "true" if ">" in long_term or "stair" in long_term else "false"
        context["no_stairs"] = "false" if context["stairs_visible"] == "true" else "true"
        
        # Door detection - look for + symbol or door text
        context["door_present"] = "true" if "+" in long_term or "door" in long_term else "false"
        
        # Enemy detection
        enemy_words = ["attacks", "hits", "misses", "bites", "strikes"]
        context["enemy_adjacent"] = "true" if any(w in long_term for w in enemy_words) else "false"
        
        # Monster proximity
        monster_words = ["monster", "creature", "enemy"] 
        context["enemy_nearby"] = "true" if any(w in long_term for w in monster_words) else "false"
        
        # Exploration status
        map_chars = long_term.count(".") + long_term.count("#") + long_term.count("+")
        unexplored = long_term.count(" ")
        if map_chars > 0:
            explored_ratio = map_chars / (map_chars + unexplored)
            context["explored_percent"] = "high" if explored_ratio > 0.7 else "low"
        else:
            context["explored_percent"] = "unknown"
            
        # Item detection - look for common item symbols
        item_symbols = [")", "[", "?", "!", "*", "(", "$", "%", "=", "/"]
        context["items_on_ground"] = "true" if any(sym in long_term for sym in item_symbols) else "false"
            
        return context
        
    def _estimate_immediate_reward(self, obs: Dict, action: str, context: Dict) -> float:
        """Better reward estimation"""
        reward = 0.0
        
        # Penalize heavy repetition
        recent_actions = list(self.action_history)[-10:]
        repetition_ratio = recent_actions.count(action) / max(len(recent_actions), 1)
        if repetition_ratio > 0.4:
            reward -= 0.3
            
        # Reward appropriate actions
        if context.get("stairs_visible") == "true" and action == "down":
            reward += 0.5  # Great! Going down stairs
        elif context.get("door_present") == "true" and action in ["open", "kick"]:
            reward += 0.3  # Good door handling
        elif context.get("enemy_adjacent") == "true" and action in ["wait", "kick"]:
            reward += 0.2  # Reasonable combat
        elif action in ["north", "east", "south", "west"] and context.get("explored_percent") == "low":
            reward += 0.1  # Exploration
        elif action == "up" and context.get("stairs_visible") != "true":
            reward -= 0.5  # Bad! No stairs to go up
            
        return reward
        
    def end_episode(self, episode_stats: Dict):
        """Enhanced episode end handling"""
        self.episode_count += 1
        total_reward = episode_stats.get("episode_return", 0)
        steps = episode_stats.get("episode_length", 1)
        score = episode_stats.get("score", 0) 
        depth = episode_stats.get("depth", 0)
        
        # Better success metric - any progress is good
        success_metric = score / 100.0 + depth * 0.5 + total_reward / 10.0
        
        print(f"\n[DreamCoder] Episode {self.episode_count} ended:")
        print(f"  Score: {score}, Depth: {depth}, Steps: {steps}")
        print(f"  Success metric: {success_metric:.3f}")
        print(f"  Library size: {len(self.library)}, Memory: {len(self.episode_memory)}")
        
        # Save episode if it showed any progress
        if success_metric > 0 or depth > 0 or score > 0:
            self.episode_memory.append({
                "trace": self.current_trace.copy(),
                "total_reward": total_reward,
                "success_metric": success_metric,
                "score": score,
                "depth": depth,
                "steps": steps
            })
            print(f"  -> Episode saved to memory (showed progress)")
            
            # Keep memory bounded
            if len(self.episode_memory) > self.memory_size:
                self.episode_memory.pop(0)
                
        # Update success rates
        self._update_abstraction_success_rates(success_metric)
        
        # Reset for next episode
        self.current_trace.clear()
        self.current_abstraction = None
        self.abstraction_step = 0
        self.failed_action_count = 0
        
        # Learn every 3 episodes if we have any successful ones
        successful_episodes = [e for e in self.episode_memory if e.get("success_metric", 0) > 0]
        if len(successful_episodes) >= 2 and self.episode_count % 3 == 0:
            print(f"\n[DreamCoder] Learning from {len(successful_episodes)} successful episodes")
            self.learn_abstractions()
            self._save_library()
            
    def learn_abstractions(self):
        """Learn new abstractions from successful episodes"""
        if not self.episode_memory:
            return
            
        # Focus on successful episodes
        successful_episodes = sorted(
            self.episode_memory,
            key=lambda e: e.get("success_metric", 0),
            reverse=True
        )[:20]  # Top 20 episodes
        
        if not successful_episodes:
            print("[DreamCoder] No successful episodes to learn from")
            return
            
        pattern_contexts = defaultdict(list)
        pattern_rewards = defaultdict(list)
        
        for episode in successful_episodes:
            trace = episode["trace"]
            
            # Extract patterns of different lengths
            for length in range(2, min(6, len(trace))):
                for i in range(len(trace) - length + 1):
                    actions = tuple(t[1] for t in trace[i:i+length])
                    contexts = [t[0] for t in trace[i:i+length]]
                    rewards = [t[2] for t in trace[i:i+length]]
                    
                    # Skip patterns with too much repetition
                    if len(set(actions)) < length / 2:
                        continue
                        
                    common_ctx = self._find_common_context(contexts)
                    if common_ctx:
                        pattern_contexts[actions].append(common_ctx)
                        pattern_rewards[actions].append(sum(rewards))
                        
        # Create new abstractions
        new_count = 0
        for pattern, contexts in pattern_contexts.items():
            if len(contexts) >= 2:  # Seen at least twice
                # Check if pattern already exists
                exists = any(abstr.pattern == pattern for abstr in self.library)
                
                if not exists:
                    merged_context = self._merge_contexts(contexts)
                    avg_reward = sum(pattern_rewards[pattern]) / len(pattern_rewards[pattern])
                    
                    new_abstraction = ProgramAbstraction(
                        name=f"learned_{len(self.library)}_{pattern[0]}_{pattern[-1]}",
                        pattern=pattern,
                        context_conditions=tuple(merged_context.items()),
                        success_rate=max(0.4, min(0.9, avg_reward + 0.6))
                    )
                    self.library.append(new_abstraction)
                    new_count += 1
                    
        print(f"[DreamCoder] Learned {new_count} new abstractions")
        self._compress_library()
        
    def _update_abstraction_success_rates(self, success_metric: float):
        """Update success rates based on episode performance"""
        # Update used abstractions
        for abstr in self.library:
            if abstr.usage_count > 0:
                # Find if this abstraction's pattern appears in the trace
                pattern_len = len(abstr.pattern)
                for i in range(len(self.current_trace) - pattern_len + 1):
                    trace_actions = [t[1] for t in self.current_trace[i:i+pattern_len]]
                    if list(abstr.pattern) == trace_actions:
                        # Update success rate with exponential moving average
                        alpha = 0.1
                        abstr.success_rate = (1 - alpha) * abstr.success_rate + alpha * min(1.0, success_metric)
                        break
                        
    def _find_common_context(self, contexts: List[Dict]) -> Optional[Dict]:
        """Find common elements across contexts"""
        if not contexts:
            return None
            
        # Find keys present in most contexts
        key_counts = Counter()
        for ctx in contexts:
            key_counts.update(ctx.keys())
            
        common_keys = [k for k, count in key_counts.items() 
                      if count >= len(contexts) * 0.7]
        
        if not common_keys:
            return None
            
        common_context = {}
        for key in common_keys:
            values = [ctx.get(key, "") for ctx in contexts]
            value_counts = Counter(values)
            most_common_value, count = value_counts.most_common(1)[0]
            if count >= len(contexts) * 0.7:
                common_context[key] = most_common_value
                
        return common_context if common_context else None
        
    def _merge_contexts(self, contexts: List[Dict]) -> Dict[str, str]:
        """Merge multiple contexts into pattern conditions"""
        merged = {}
        
        for key in set().union(*(ctx.keys() for ctx in contexts)):
            values = [ctx.get(key, "") for ctx in contexts if key in ctx]
            if values:
                # Create regex pattern that matches common values
                if len(set(values)) == 1:
                    merged[key] = values[0]
                else:
                    # Create alternation pattern
                    merged[key] = "|".join(set(values))
                    
        return merged
        
    def _compress_library(self):
        """Remove poor-performing abstractions"""
        # Remove abstractions with very low success rate after enough usage
        self.library = [
            a for a in self.library 
            if a.success_rate > 0.2 or a.usage_count < 3
        ]
        
    def _save_library(self):
        """Save learned library to file"""
        library_data = [
            {
                "name": abstr.name,
                "pattern": list(abstr.pattern),
                "conditions": dict(abstr.context_conditions),
                "success_rate": abstr.success_rate,
                "usage_count": abstr.usage_count
            }
            for abstr in self.library
        ]
        
        Path(self.library_save_path).parent.mkdir(exist_ok=True, parents=True)
        with open(self.library_save_path, "w") as f:
            json.dump(library_data, f, indent=2)
        print(f"[DreamCoder] Saved {len(self.library)} abstractions to {self.library_save_path}")
        
    def _load_library_if_exists(self):
        """Load previously learned library"""
        if os.path.exists(self.library_save_path):
            try:
                with open(self.library_save_path, "r") as f:
                    library_data = json.load(f)
                    
                for data in library_data:
                    exists = any(a.name == data["name"] for a in self.library)
                    if not exists:
                        self.library.append(ProgramAbstraction(
                            name=data["name"],
                            pattern=tuple(data["pattern"]),
                            context_conditions=tuple(data["conditions"].items()),
                            success_rate=data["success_rate"],
                            usage_count=data["usage_count"]
                        ))
                print(f"[DreamCoder] Loaded {len(library_data)} abstractions from {self.library_save_path}")
            except Exception as e:
                print(f"[DreamCoder] Error loading library: {e}")
                
    def reset(self):
        """Reset agent state for new episode"""
        self.prompt_builder.reset()
        self.action_history.clear()
        self.current_abstraction = None
        self.abstraction_step = 0
        self.failed_action_count = 0