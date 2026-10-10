"""DreamCoder-inspired agent that learns action abstractions over time"""

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


class DreamCoderAgent(BaseAgent):
    """Agent that learns action abstractions from experience like DreamCoder"""
    
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
        self.library_save_path = library_save_path or os.path.abspath("dreamcoder_library.json")
        
        # Library of learned abstractions
        self.library: List[ProgramAbstraction] = []
        
        # Memory of successful episodes
        self.episode_memory: List[Dict] = []
        
        # Current episode trace
        self.current_trace: List[Tuple[Dict, str, float]] = []
        self.current_reward = 0.0
        
        # For repetition penalty
        self.action_history = deque(maxlen=repetition_penalty_window)
        self.repetition_penalty_factor = 0.5
        
        # Track abstraction execution
        self.current_abstraction: Optional[ProgramAbstraction] = None
        self.abstraction_step = 0
        
        # Initialize with primitives and load saved library
        self._initialize_primitives()
        self._load_library_if_exists()
        
    def _initialize_primitives(self):
        """Initialize with basic movement and action primitives"""
        # Basic exploration pattern
        self.library.append(ProgramAbstraction(
            name="explore_systematically",
            pattern=("north", "east", "south", "west", "search"),
            context_conditions=(("location_type", "room|corridor"), ("no_enemies", "true")),
            success_rate=0.5
        ))
        
        # Combat pattern
        self.library.append(ProgramAbstraction(
            name="combat_basic",
            pattern=("wait", "wait"),  # Let monster approach, then attack
            context_conditions=(("enemy_present", "true"),),
            success_rate=0.5
        ))
        
        # Stair seeking
        self.library.append(ProgramAbstraction(
            name="seek_stairs",
            pattern=("search", "search", "search"),
            context_conditions=(("stairs_visible", "false"), ("explored_percent", "high")),
            success_rate=0.5
        ))
        
    def act(self, obs, prev_action=None):
        """Generate action using learned abstractions or base LLM"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
            
        self.prompt_builder.update_observation(obs)
        
        # Extract context from observation
        context = self._extract_context(obs)
        
        # Check if we're in the middle of executing an abstraction
        if self.current_abstraction and self.abstraction_step < len(self.current_abstraction.pattern):
            action = self.current_abstraction.pattern[self.abstraction_step]
            self.abstraction_step += 1
            
            # Substitute variables in action
            action = self._substitute_variables(action, context)
            
            # If abstraction is complete, clear it
            if self.abstraction_step >= len(self.current_abstraction.pattern):
                self.current_abstraction = None
                self.abstraction_step = 0
        else:
            # Select new abstraction or use base agent
            applicable_abstractions = [
                abstr for abstr in self.library 
                if abstr.matches_context(context) and abstr.success_rate > 0.3
            ]
            
            if applicable_abstractions:
                # Apply repetition penalty
                abstraction_scores = {}
                for abstr in applicable_abstractions:
                    base_score = abstr.success_rate
                    
                    # Penalize if recently used
                    recent_actions = list(self.action_history)[-len(abstr.pattern):]
                    if recent_actions == abstr.pattern[:len(recent_actions)]:
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
                action = self._substitute_variables(action, context)
                self.abstraction_step = 1
            else:
                # Use base LLM with restricted action space
                action = self._get_llm_action(obs, context)
                
        # Update history and trace
        self.action_history.append(action)
        reward = self._estimate_immediate_reward(obs, action, context)
        self.current_trace.append((context, action, reward))
        
        # Create response object compatible with BALROG
        return LLMResponse(
            model_id="dreamcoder",
            completion=action,
            stop_reason="complete",
            input_tokens=0,  # Will be updated by actual LLM call
            output_tokens=0,
            reasoning=f"Using abstraction: {self.current_abstraction.name}" if self.current_abstraction else "Direct action"
        )
        
    def _get_llm_action(self, obs, context):
        """Get action from base LLM with restricted action space"""
        messages = self.prompt_builder.get_prompt()
        
        # Add context about learned patterns
        context_info = "\nLearned patterns suggest avoiding: "
        recent_failures = [a.name for a in self.library if a.success_rate < 0.2]
        if recent_failures:
            context_info += ", ".join(recent_failures[:3])
        else:
            context_info += "none"
            
        # Restrict to core actions
        core_actions = self._get_core_actions()
        action_prompt = f"""
Choose ONE action from: {', '.join(core_actions)}

{context_info}

Output only the action name, nothing else.
        """.strip()
        
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + action_prompt
            
        response = self.client.generate(messages)
        
        # Extract action from response
        action = response.completion.strip().lower()
        
        # Validate action
        if action not in core_actions:
            # Find closest match
            for valid_action in core_actions:
                if valid_action in action or action in valid_action:
                    action = valid_action
                    break
            else:
                action = "wait"  # Default fallback
                
        return action
        
    def _get_core_actions(self):
        """Get reduced set of core actions"""
        return [
            "north", "south", "east", "west",
            "northeast", "southeast", "southwest", "northwest",
            "up", "down", "wait", "search", "look",
            "pickup", "eat", "quaff", "wear", "wield",
            "open", "close", "kick", "pray"
        ]
        
    def _extract_context(self, obs: Dict) -> Dict:
        """Extract structured context from observation"""
        context = {}
        text = obs.get("text", {})
        long_term = text.get("long_term_context", "").lower()
        short_term = text.get("short_term_context", "").lower()
        
        # Location analysis
        if "corridor" in long_term or "passage" in long_term:
            context["location_type"] = "corridor"
        elif "room" in long_term or "chamber" in long_term:
            context["location_type"] = "room"
        else:
            context["location_type"] = "unknown"
            
        # Enemy detection
        enemy_words = ["monster", "attacks", "hostile", "hits", "misses"]
        context["enemy_present"] = "true" if any(w in long_term for w in enemy_words) else "false"
        context["no_enemies"] = "false" if context["enemy_present"] == "true" else "true"
        
        # Stairs detection
        context["stairs_visible"] = "true" if ">" in long_term or "stair" in long_term else "false"
        
        # Exploration status (rough estimate)
        map_area = long_term.count(".") + long_term.count("#")
        unexplored = long_term.count(" ")
        if map_area > 0:
            explored_ratio = 1 - (unexplored / (map_area + unexplored))
            context["explored_percent"] = "high" if explored_ratio > 0.7 else "low"
        else:
            context["explored_percent"] = "unknown"
            
        # Status checks
        if "hungry" in short_term or "weak" in short_term:
            context["need_food"] = "true"
            
        # Inventory summary
        if "food" in short_term:
            match = re.search(r'([a-z])\s*-\s*\w*food', short_term)
            if match:
                context["food_letter"] = match.group(1)
                
        return context
        
    def _substitute_variables(self, action: str, context: Dict) -> str:
        """Substitute variables in action patterns"""
        if "{{" in action and "}}" in action:
            var_match = re.search(r'\{\{(\w+)\}\}', action)
            if var_match:
                var_name = var_match.group(1)
                if var_name in context:
                    return action.replace(f"{{{{{var_name}}}}}", context[var_name])
        return action
        
    def _estimate_immediate_reward(self, obs: Dict, action: str, context: Dict) -> float:
        """Estimate immediate reward for action"""
        reward = 0.0
        
        # Penalize repetition
        recent_actions = list(self.action_history)[-5:]
        if recent_actions.count(action) > 2:
            reward -= 0.2
            
        # Reward exploration
        if action in ["north", "east", "south", "west"] and context.get("explored_percent") == "low":
            reward += 0.1
            
        # Reward appropriate responses
        if context.get("enemy_present") == "true" and action in ["wait", "kick"]:
            reward += 0.15
        elif context.get("need_food") == "true" and action == "eat":
            reward += 0.2
        elif context.get("stairs_visible") == "true" and action == "down":
            reward += 0.3
            
        return reward
        
    def end_episode(self, episode_stats: Dict):
        """Called at episode end to save memory and trigger learning"""
        # Calculate total reward from episode stats
        total_reward = episode_stats.get("episode_return", 0)
        steps = episode_stats.get("episode_length", 1)
        
        # Normalize reward by episode length
        normalized_reward = total_reward / max(steps, 1)
        
        print(f"\n[DreamCoder] Episode ended: reward={total_reward:.2f}, steps={steps}, normalized={normalized_reward:.4f}")
        print(f"[DreamCoder] Memory size: {len(self.episode_memory)}, Library size: {len(self.library)}")
        
        # Save episode if it was reasonably successful
        if normalized_reward > -0.1:  # Not terrible
            self.episode_memory.append({
                "trace": self.current_trace.copy(),
                "total_reward": total_reward,
                "normalized_reward": normalized_reward,
                "steps": steps
            })
            
            # Keep memory bounded
            if len(self.episode_memory) > self.memory_size:
                self.episode_memory.pop(0)
                
        # Update success rates of used abstractions
        self._update_abstraction_success_rates(normalized_reward)
        
        # Reset for next episode
        self.current_trace.clear()
        self.current_abstraction = None
        self.abstraction_step = 0
        
        # Periodically learn new abstractions
        if len(self.episode_memory) > 0 and len(self.episode_memory) % 5 == 0:
            print(f"\n[DreamCoder] Triggering learning with {len(self.episode_memory)} episodes in memory")
            self.learn_abstractions()
            self._save_library()
            print(f"[DreamCoder] Library saved to: {self.library_save_path}")
            
    def _update_abstraction_success_rates(self, episode_reward: float):
        """Update success rates of abstractions used in this episode"""
        used_abstractions = defaultdict(list)
        
        # Find which abstractions were used
        for abstr in self.library:
            pattern_len = len(abstr.pattern)
            for i in range(len(self.current_trace) - pattern_len + 1):
                trace_actions = [t[1] for t in self.current_trace[i:i+pattern_len]]
                if trace_actions == abstr.pattern:
                    # Calculate local reward around this abstraction
                    local_rewards = [t[2] for t in self.current_trace[i:i+pattern_len]]
                    used_abstractions[abstr].append(sum(local_rewards))
                    
        # Update success rates with exponential moving average
        alpha = 0.1  # Learning rate
        for abstr, rewards in used_abstractions.items():
            avg_reward = sum(rewards) / len(rewards)
            # Normalize to 0-1 range
            success_score = max(0, min(1, (avg_reward + 1) / 2))
            abstr.success_rate = (1 - alpha) * abstr.success_rate + alpha * success_score
            
    def learn_abstractions(self):
        """Learn new abstractions from successful episodes"""
        if len(self.episode_memory) < 3:
            return
            
        # Find common successful patterns
        pattern_contexts = defaultdict(list)
        pattern_rewards = defaultdict(list)
        
        # Analyze recent successful episodes
        for episode in sorted(self.episode_memory[-20:], 
                            key=lambda e: e["normalized_reward"], 
                            reverse=True)[:10]:
            trace = episode["trace"]
            
            # Extract patterns of different lengths
            for length in range(2, min(8, len(trace))):
                for i in range(len(trace) - length + 1):
                    actions = tuple(t[1] for t in trace[i:i+length])
                    contexts = [t[0] for t in trace[i:i+length]]
                    rewards = [t[2] for t in trace[i:i+length]]
                    
                    # Find common context
                    common_ctx = self._find_common_context(contexts)
                    if common_ctx:
                        pattern_contexts[actions].append(common_ctx)
                        pattern_rewards[actions].append(sum(rewards))
                        
        # Create new abstractions for frequent successful patterns
        for pattern, contexts in pattern_contexts.items():
            if len(contexts) >= 3:  # Seen at least 3 times
                # Check if pattern already exists
                exists = any(
                    abstr.pattern == pattern 
                    for abstr in self.library
                )
                
                if not exists:
                    # Merge contexts to find common conditions
                    merged_context = self._merge_contexts(contexts)
                    avg_reward = sum(pattern_rewards[pattern]) / len(pattern_rewards[pattern])
                    
                    new_abstraction = ProgramAbstraction(
                        name=f"learned_{len(self.library)}_{pattern[0]}_{pattern[-1]}",
                        pattern=pattern,  # Already a tuple
                        context_conditions=tuple(merged_context.items()),
                        success_rate=max(0.3, min(0.9, avg_reward + 0.5))
                    )
                    self.library.append(new_abstraction)
                    
        # Compress library
        self._compress_library()
        
    def _find_common_context(self, contexts: List[Dict]) -> Optional[Dict]:
        """Find common elements across contexts"""
        if not contexts:
            return None
            
        # Find keys present in most contexts
        key_counts = Counter()
        for ctx in contexts:
            key_counts.update(ctx.keys())
            
        common_keys = [k for k, count in key_counts.items() 
                      if count >= len(contexts) * 0.8]
        
        if not common_keys:
            return None
            
        common_context = {}
        for key in common_keys:
            values = [ctx.get(key, "") for ctx in contexts]
            # If values are mostly the same, include it
            value_counts = Counter(values)
            most_common_value, count = value_counts.most_common(1)[0]
            if count >= len(contexts) * 0.8:
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
        """Remove redundant or poor-performing abstractions"""
        # Remove abstractions with very low success rate
        self.library = [a for a in self.library if a.success_rate > 0.1 or a.usage_count < 5]
        
        # Merge very similar abstractions
        merged = []
        used = set()
        
        for i, abstr1 in enumerate(self.library):
            if i in used:
                continue
                
            # Find similar abstractions
            similar = [abstr1]
            for j, abstr2 in enumerate(self.library[i+1:], i+1):
                if j not in used and self._abstractions_similar(abstr1, abstr2):
                    similar.append(abstr2)
                    used.add(j)
                    
            # Merge if multiple similar found
            if len(similar) > 1:
                merged_abstr = self._merge_abstractions(similar)
                merged.append(merged_abstr)
            else:
                merged.append(abstr1)
                
        self.library = merged
        
    def _abstractions_similar(self, a1: ProgramAbstraction, a2: ProgramAbstraction) -> bool:
        """Check if two abstractions are similar"""
        if abs(len(a1.pattern) - len(a2.pattern)) > 1:
            return False
            
        # Check pattern similarity
        if len(a1.pattern) == len(a2.pattern):
            matches = sum(1 for x, y in zip(a1.pattern, a2.pattern) if x == y)
            if matches / len(a1.pattern) < 0.7:
                return False
        
        # Check context similarity
        ctx_keys1 = set(a1.context_conditions.keys())
        ctx_keys2 = set(a2.context_conditions.keys())
        if len(ctx_keys1 & ctx_keys2) < len(ctx_keys1 | ctx_keys2) * 0.5:
            return False
            
        return True
        
    def _merge_abstractions(self, abstractions: List[ProgramAbstraction]) -> ProgramAbstraction:
        """Merge similar abstractions"""
        # Use the most successful pattern
        best = max(abstractions, key=lambda a: a.success_rate * (1 + a.usage_count * 0.01))
        
        # Average success rates
        avg_success = sum(a.success_rate for a in abstractions) / len(abstractions)
        total_usage = sum(a.usage_count for a in abstractions)
        
        # Merge context conditions
        merged_conditions = {}
        for abstr in abstractions:
            for key, pattern in abstr.context_conditions.items():
                if key in merged_conditions:
                    # Combine patterns with |
                    existing = set(merged_conditions[key].split("|"))
                    new = set(pattern.split("|"))
                    merged_conditions[key] = "|".join(existing | new)
                else:
                    merged_conditions[key] = pattern
                    
        return ProgramAbstraction(
            name=f"merged_{best.name}",
            pattern=best.pattern,
            context_conditions=tuple(merged_conditions.items()),
            success_rate=avg_success,
            usage_count=total_usage
        )
        
    def _save_library(self):
        """Save learned library to file"""
        library_data = [
            {
                "name": abstr.name,
                "pattern": list(abstr.pattern),  # Convert tuple to list for JSON
                "conditions": dict(abstr.context_conditions),  # Convert tuple to dict for JSON
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
                    
                # Add loaded abstractions to existing primitives
                for data in library_data:
                    # Don't duplicate primitives
                    exists = any(
                        a.name == data["name"] for a in self.library
                    )
                    if not exists:
                        self.library.append(ProgramAbstraction(
                            name=data["name"],
                            pattern=tuple(data["pattern"]),  # Convert list back to tuple
                            context_conditions=tuple(data["conditions"].items()),  # Convert dict back to tuple
                            success_rate=data["success_rate"],
                            usage_count=data["usage_count"]
                        ))
            except Exception as e:
                print(f"Error loading library: {e}")
                
    def reset(self):
        """Reset agent state for new episode"""
        self.prompt_builder.reset()
        self.action_history.clear()
        self.current_abstraction = None
        self.abstraction_step = 0