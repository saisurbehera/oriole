#!/usr/bin/env python3

"""
Inspect the trained MCTS values
"""

import pickle
import json

# Load trained values
with open('mcts_trained_values.pkl', 'rb') as f:
    action_values = pickle.load(f)

print("=== Trained MCTS Values ===")
print(f"Contexts learned: {len(action_values)}")
print()

# Show each context and its best actions
for context, actions in action_values.items():
    print(f"Context: {context}")
    
    # Sort actions by average reward
    sorted_actions = []
    for action, stats in actions.items():
        if stats.visit_count > 0:
            sorted_actions.append((action, stats.average_reward, stats.visit_count))
    
    sorted_actions.sort(key=lambda x: x[1], reverse=True)
    
    # Show top 5 actions
    for action, avg_reward, visits in sorted_actions[:5]:
        print(f"  {action:10s} avg_reward={avg_reward:6.2f}, visits={visits:3d}")
    print()

# Find the best overall action-context pairs
print("=== Best Action-Context Pairs ===")
best_pairs = []
for context, actions in action_values.items():
    for action, stats in actions.items():
        if stats.visit_count > 2:  # At least 3 visits
            best_pairs.append((context, action, stats.average_reward, stats.visit_count))

best_pairs.sort(key=lambda x: x[2], reverse=True)

for context, action, avg_reward, visits in best_pairs[:10]:
    print(f"{context:40s} -> {action:10s} (avg={avg_reward:.2f}, n={visits})")

# Check for stairs-related learning
print("\n=== Stairs-Related Learning ===")
stairs_contexts = [ctx for ctx in action_values.keys() if 'stairs' in ctx]
if stairs_contexts:
    for ctx in stairs_contexts:
        print(f"\nContext: {ctx}")
        if '>' in action_values[ctx]:
            stats = action_values[ctx]['>']
            print(f"  '>' action: avg_reward={stats.average_reward:.2f}, visits={stats.visit_count}")
else:
    print("No stairs-related contexts found!")