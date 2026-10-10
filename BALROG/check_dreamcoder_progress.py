#!/usr/bin/env python3
"""Check DreamCoder agent progress from episode files"""

import json
import glob
from collections import Counter, defaultdict

# Find all episode files
episode_files = sorted(glob.glob("NetHackChallenge-v0_run_*.json"))
print(f"Found {len(episode_files)} completed episodes\n")

# Analyze episodes
total_score = 0
total_depth = 0
action_patterns = defaultdict(int)
door_attempts = 0
exploration_actions = 0

for i, file in enumerate(episode_files[-10:]):  # Last 10 episodes
    with open(file, 'r') as f:
        data = json.load(f)
    
    score = data.get('score', 0)
    depth = data.get('depth', 0)
    actions = data.get('action_frequency', {})
    
    total_score += score
    total_depth += depth
    
    # Count action types
    for action, count in actions.items():
        action_patterns[action] += count
        if action in ['open', 'kick']:
            door_attempts += count
        elif action in ['north', 'south', 'east', 'west']:
            exploration_actions += count
    
    print(f"Episode {i}: Score={score}, Depth={depth}, Steps={data.get('num_steps', 0)}")
    if score > 0 or depth > 0:
        print(f"  *** PROGRESS! ***")

print(f"\n=== Summary of last {min(10, len(episode_files))} episodes ===")
print(f"Total score: {total_score}")
print(f"Total depth reached: {total_depth}")
print(f"Door attempts (open/kick): {door_attempts}")
print(f"Exploration moves: {exploration_actions}")

print("\nTop 10 actions:")
for action, count in sorted(action_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {action}: {count}")

# Check for patterns
print("\n=== Action Patterns ===")
if 'up' in action_patterns:
    print(f"❌ Still trying 'up' without stairs: {action_patterns['up']} times")
else:
    print("✓ Not trying 'up' anymore (good!)")

if door_attempts > 0:
    print(f"✓ Attempting to handle doors: {door_attempts} times")
    
if exploration_actions > 0:
    print(f"✓ Exploring with movement: {exploration_actions} times")

# Estimate if agent is using abstractions
total_actions = sum(action_patterns.values())
unique_actions = len(action_patterns)
print(f"\nAction diversity: {unique_actions} unique actions out of {total_actions} total")

# Check if it's stuck in loops
for episode_file in episode_files[-3:]:
    with open(episode_file, 'r') as f:
        data = json.load(f)
    actions = data.get('action_frequency', {})
    top_action = max(actions.items(), key=lambda x: x[1]) if actions else ('none', 0)
    repetition_rate = top_action[1] / sum(actions.values()) if actions else 0
    if repetition_rate > 0.5:
        print(f"\n⚠️  High repetition in recent episode: {top_action[0]} used {repetition_rate:.0%} of the time")