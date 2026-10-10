#!/usr/bin/env python3
"""
Demo to show how the DreamCoder learning works
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class LearnedPattern:
    """A pattern the agent has learned"""
    name: str
    actions: List[str]
    context: Dict[str, str]
    success_count: int = 0
    usage_count: int = 0
    
    @property
    def success_rate(self):
        return self.success_count / max(1, self.usage_count)

class SimpleLearningDemo:
    def __init__(self):
        self.patterns = []
        self.current_trace = []
        
    def simulate_episode(self, episode_num: int):
        """Simulate one episode of NetHack"""
        print(f"\n=== Episode {episode_num} ===")
        
        # Simulated game state
        stairs_found = False
        position = (5, 5)
        stairs_pos = (8, 2)
        
        # Track what happens
        actions_taken = []
        
        for step in range(20):
            # Current context
            context = {
                "stairs_visible": str(stairs_found),
                "position": f"{position[0]},{position[1]}"
            }
            
            # Check if we have a learned pattern for this context
            matching_patterns = [
                p for p in self.patterns 
                if p.context.get("stairs_visible") == context["stairs_visible"]
                and p.success_rate > 0.5
            ]
            
            if matching_patterns and stairs_found:
                # Use learned pattern!
                pattern = max(matching_patterns, key=lambda p: p.success_rate)
                pattern_step = min(step - 5, len(pattern.actions) - 1)  # Adjust for when stairs were found
                if pattern_step >= 0:
                    action = pattern.actions[pattern_step]
                    print(f"  Step {step}: Using learned pattern '{pattern.name}' -> {action}")
                else:
                    action = "search"
            else:
                # Explore randomly
                if not stairs_found and step % 5 == 0:
                    action = "search"
                elif position[0] < stairs_pos[0]:
                    action = "east"
                elif position[0] > stairs_pos[0]:
                    action = "west"
                elif position[1] < stairs_pos[1]:
                    action = "north"
                elif position[1] > stairs_pos[1]:
                    action = "south"
                else:
                    action = "down"
                    
                print(f"  Step {step}: Exploring -> {action}")
            
            actions_taken.append(action)
            
            # Simulate action effects
            if action == "search" and abs(position[0] - stairs_pos[0]) <= 2 and abs(position[1] - stairs_pos[1]) <= 2:
                stairs_found = True
                print(f"    -> Found stairs at {stairs_pos}!")
            elif action == "east":
                position = (position[0] + 1, position[1])
            elif action == "west":
                position = (position[0] - 1, position[1])
            elif action == "north":
                position = (position[0], position[1] - 1)
            elif action == "south":  
                position = (position[0], position[1] + 1)
            elif action == "down" and position == stairs_pos:
                print(f"    -> SUCCESS! Descended stairs!")
                
                # Learn from this success!
                self.learn_from_success(actions_taken, stairs_found)
                return True
                
        print("  Episode ended without finding stairs")
        return False
        
    def learn_from_success(self, actions: List[str], had_stairs: bool):
        """Extract patterns from successful episode"""
        print("\n  🧠 LEARNING FROM SUCCESS:")
        
        # Find the successful navigation pattern
        # Look for the last few actions that led to stairs
        stairs_nav_pattern = []
        for i in range(len(actions)-1, -1, -1):
            if actions[i] == "down":
                # Found the successful descent, backtrack to get navigation
                start = max(0, i-3)
                stairs_nav_pattern = actions[start:i+1]
                break
                
        if stairs_nav_pattern:
            new_pattern = LearnedPattern(
                name=f"stairs_descent_v{len(self.patterns)}",
                actions=stairs_nav_pattern,
                context={"stairs_visible": "true"},
                success_count=1,
                usage_count=1
            )
            
            self.patterns.append(new_pattern)
            print(f"    Learned pattern: {stairs_nav_pattern}")
            print(f"    Context: When stairs are visible")
                
def main():
    demo = SimpleLearningDemo()
    
    print("🎮 DREAMCODER LEARNING DEMONSTRATION")
    print("=" * 40)
    print("Watch how the agent learns patterns over episodes:\n")
    
    successes = 0
    for episode in range(5):
        if demo.simulate_episode(episode + 1):
            successes += 1
            
    print(f"\n📊 FINAL RESULTS:")
    print(f"Success rate: {successes}/5 episodes")
    print(f"Patterns learned: {len(demo.patterns)}")
    
    if demo.patterns:
        print("\n🧠 LEARNED PATTERNS:")
        for pattern in demo.patterns:
            print(f"  - {pattern.name}: {pattern.actions}")
            print(f"    Success rate: {pattern.success_rate:.0%}")

if __name__ == "__main__":
    main()