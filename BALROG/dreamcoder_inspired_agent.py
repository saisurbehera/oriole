"""DreamCoder-inspired approach for NetHack action learning"""

from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import re
import json


@dataclass
class ProgramAbstraction:
    """Learned abstraction/sub-program"""
    name: str
    pattern: List[str]  # Sequence of actions
    context_conditions: Dict[str, str]  # Conditions when this applies
    success_rate: float = 0.0
    usage_count: int = 0
    
    def matches_context(self, context: Dict) -> bool:
        """Check if current context matches this abstraction's conditions"""
        for key, pattern in self.context_conditions.items():
            if key not in context:
                return False
            if not re.search(pattern, str(context[key]), re.IGNORECASE):
                return False
        return True


class DreamCoderNetHackAgent:
    """Agent that learns action abstractions like DreamCoder"""
    
    def __init__(self, base_agent, memory_size: int = 1000):
        self.base_agent = base_agent
        self.memory_size = memory_size
        
        # Library of learned abstractions
        self.library: List[ProgramAbstraction] = []
        
        # Memory of successful episodes
        self.episode_memory: List[Dict] = []
        
        # Current episode trace
        self.current_trace: List[Tuple[Dict, str, float]] = []
        
        # Initialize with some primitive abstractions
        self._initialize_primitives()
        
    def _initialize_primitives(self):
        """Initialize with basic movement and action primitives"""
        # Basic movement patterns
        self.library.append(ProgramAbstraction(
            name="explore_room",
            pattern=["north", "east", "south", "west"],
            context_conditions={"location": "room|chamber"},
        ))
        
        self.library.append(ProgramAbstraction(
            name="follow_corridor",
            pattern=["north", "north", "search"],
            context_conditions={"location": "corridor|passage"},
        ))
        
        # Combat patterns
        self.library.append(ProgramAbstraction(
            name="hit_and_retreat",
            pattern=["fight", "south", "wait", "north", "fight"],
            context_conditions={"enemy": "monster|hostile"},
        ))
        
        # Item management
        self.library.append(ProgramAbstraction(
            name="eat_when_hungry",
            pattern=["eat", "{{food_letter}}"],
            context_conditions={"status": "hungry|weak", "inventory": "food"},
        ))
        
    def act(self, obs: Dict, prev_action: Optional[str] = None) -> str:
        """Select action using learned abstractions"""
        context = self._extract_context(obs)
        
        # Try to apply learned abstractions first
        applicable_abstractions = [
            abstr for abstr in self.library 
            if abstr.matches_context(context) and abstr.success_rate > 0.3
        ]
        
        if applicable_abstractions:
            # Sort by success rate and recency
            best_abstraction = max(
                applicable_abstractions,
                key=lambda a: a.success_rate * (1 + a.usage_count * 0.01)
            )
            
            # Execute abstraction
            action = self._execute_abstraction(best_abstraction, context)
            best_abstraction.usage_count += 1
        else:
            # Fall back to base agent
            action = self.base_agent.act(obs, prev_action)
            
        # Record for learning
        reward = self._estimate_immediate_reward(obs, action)
        self.current_trace.append((context, action, reward))
        
        return action
        
    def _execute_abstraction(self, abstraction: ProgramAbstraction, context: Dict) -> str:
        """Execute a learned abstraction"""
        pattern = abstraction.pattern.copy()
        
        # Substitute variables in pattern
        for i, action in enumerate(pattern):
            if "{{" in action and "}}" in action:
                # This is a variable to be filled from context
                var_name = action[2:-2]
                if var_name == "food_letter":
                    # Extract food letter from inventory
                    match = re.search(r'([a-z])\s*-\s*\w*food', context.get("inventory", ""))
                    if match:
                        pattern[i] = match.group(1)
                        
        # Return first action in pattern
        # (In full implementation, would track position in pattern)
        return pattern[0]
        
    def learn_abstractions(self):
        """Dream phase: Learn new abstractions from successful episodes"""
        if len(self.episode_memory) < 10:
            return
            
        # Find common successful patterns
        pattern_counts = defaultdict(int)
        pattern_rewards = defaultdict(float)
        
        for episode in self.episode_memory[-100:]:  # Recent episodes
            trace = episode["trace"]
            total_reward = episode["total_reward"]
            
            # Extract action subsequences
            for i in range(len(trace) - 3):
                # Look for patterns of length 3-5
                for length in range(3, min(6, len(trace) - i)):
                    actions = [t[1] for t in trace[i:i+length]]
                    contexts = [t[0] for t in trace[i:i+length]]
                    
                    # Find common context conditions
                    common_context = self._extract_common_context(contexts)
                    
                    pattern_key = tuple(actions)
                    pattern_counts[pattern_key] += 1
                    pattern_rewards[pattern_key] += total_reward
                    
        # Create new abstractions for frequent successful patterns
        for pattern, count in pattern_counts.items():
            if count >= 5:  # Seen at least 5 times
                avg_reward = pattern_rewards[pattern] / count
                if avg_reward > 0:
                    # Check if pattern already exists
                    if not any(abstr.pattern == list(pattern) for abstr in self.library):
                        new_abstraction = ProgramAbstraction(
                            name=f"learned_pattern_{len(self.library)}",
                            pattern=list(pattern),
                            context_conditions={},  # Would extract from common_context
                            success_rate=min(avg_reward / 10.0, 1.0),  # Normalize
                        )
                        self.library.append(new_abstraction)
                        
        # Compress library by merging similar abstractions
        self._compress_library()
        
    def _compress_library(self):
        """Merge similar abstractions to avoid library explosion"""
        # Find abstractions with similar patterns
        merged = []
        used = set()
        
        for i, abstr1 in enumerate(self.library):
            if i in used:
                continue
                
            similar_group = [abstr1]
            for j, abstr2 in enumerate(self.library[i+1:], i+1):
                if j in used:
                    continue
                    
                # Check similarity (e.g., edit distance, context overlap)
                if self._abstractions_similar(abstr1, abstr2):
                    similar_group.append(abstr2)
                    used.add(j)
                    
            # Merge similar abstractions
            if len(similar_group) > 1:
                merged_abstr = self._merge_abstractions(similar_group)
                merged.append(merged_abstr)
            else:
                merged.append(abstr1)
                
        self.library = merged
        
    def _abstractions_similar(self, a1: ProgramAbstraction, a2: ProgramAbstraction) -> bool:
        """Check if two abstractions are similar enough to merge"""
        # Simple check: same length and >70% action overlap
        if len(a1.pattern) != len(a2.pattern):
            return False
            
        matches = sum(1 for x, y in zip(a1.pattern, a2.pattern) if x == y)
        return matches / len(a1.pattern) > 0.7
        
    def _merge_abstractions(self, abstractions: List[ProgramAbstraction]) -> ProgramAbstraction:
        """Merge multiple similar abstractions into one"""
        # Take most common action at each position
        merged_pattern = []
        for i in range(len(abstractions[0].pattern)):
            actions_at_i = [a.pattern[i] for a in abstractions]
            most_common = Counter(actions_at_i).most_common(1)[0][0]
            merged_pattern.append(most_common)
            
        # Average success rates
        avg_success = sum(a.success_rate for a in abstractions) / len(abstractions)
        total_usage = sum(a.usage_count for a in abstractions)
        
        return ProgramAbstraction(
            name=f"merged_{abstractions[0].name}",
            pattern=merged_pattern,
            context_conditions=abstractions[0].context_conditions,
            success_rate=avg_success,
            usage_count=total_usage,
        )
        
    def _extract_context(self, obs: Dict) -> Dict:
        """Extract structured context from observation"""
        context = {}
        text = obs.get("text", {})
        
        # Location type
        long_term = text.get("long_term_context", "").lower()
        if "corridor" in long_term:
            context["location"] = "corridor"
        elif "room" in long_term:
            context["location"] = "room"
            
        # Enemy presence
        if any(word in long_term for word in ["monster", "attacks", "hostile"]):
            context["enemy"] = "present"
            
        # Status
        short_term = text.get("short_term_context", "").lower()
        if "hungry" in short_term:
            context["status"] = "hungry"
        elif "weak" in short_term:
            context["status"] = "weak"
            
        # Inventory summary
        context["inventory"] = short_term
        
        return context
        
    def _extract_common_context(self, contexts: List[Dict]) -> Dict:
        """Find common elements across multiple contexts"""
        if not contexts:
            return {}
            
        # Find keys present in all contexts
        common_keys = set(contexts[0].keys())
        for ctx in contexts[1:]:
            common_keys &= set(ctx.keys())
            
        common_context = {}
        for key in common_keys:
            # Find common values or patterns
            values = [ctx[key] for ctx in contexts]
            if len(set(values)) == 1:
                common_context[key] = values[0]
                
        return common_context
        
    def _estimate_immediate_reward(self, obs: Dict, action: str) -> float:
        """Estimate immediate reward for action (before env step)"""
        # Heuristic rewards
        context = self._extract_context(obs)
        
        # Penalize repetition
        if self.current_trace and self.current_trace[-1][1] == action:
            return -0.1
            
        # Reward exploration
        if action in ["north", "east", "south", "west"] and "unexplored" in str(obs):
            return 0.1
            
        # Reward combat when enemies present
        if "enemy" in context and action == "fight":
            return 0.2
            
        return 0.0
        
    def end_episode(self, total_reward: float):
        """Called at episode end to save memory and trigger learning"""
        if total_reward > 0:  # Only remember successful episodes
            self.episode_memory.append({
                "trace": self.current_trace.copy(),
                "total_reward": total_reward,
            })
            
            # Keep memory bounded
            if len(self.episode_memory) > self.memory_size:
                self.episode_memory.pop(0)
                
        # Reset for next episode
        self.current_trace.clear()
        
        # Periodically learn new abstractions
        if len(self.episode_memory) % 10 == 0:
            self.learn_abstractions()
            
    def save_library(self, path: str):
        """Save learned library to file"""
        library_data = [
            {
                "name": abstr.name,
                "pattern": abstr.pattern,
                "conditions": abstr.context_conditions,
                "success_rate": abstr.success_rate,
                "usage_count": abstr.usage_count,
            }
            for abstr in self.library
        ]
        with open(path, "w") as f:
            json.dump(library_data, f, indent=2)
            
    def load_library(self, path: str):
        """Load library from file"""
        with open(path, "r") as f:
            library_data = json.load(f)
            
        self.library = [
            ProgramAbstraction(
                name=data["name"],
                pattern=data["pattern"],
                context_conditions=data["conditions"],
                success_rate=data["success_rate"],
                usage_count=data["usage_count"],
            )
            for data in library_data
        ]