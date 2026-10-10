#!/usr/bin/env python3

"""
Test the improved MCTS agent
"""

import sys
import subprocess
import json

# Run the improved MCTS agent
cmd = [
    "python3", "eval.py",
    "agent.type=feedback_mcts_v2",
    "client.client_name=openai",
    "client.model_id=meta-llama/Llama-3.2-1B-Instruct",
    "client.base_url=http://0.0.0.0:8080/v1",
    "envs.names=nle",
    "eval.num_episodes.nle=3",
    "eval.num_workers=1",
    "eval.save_trajectories=true"
]

print("Testing improved MCTS agent...")
print(" ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True)

print("\n=== Output ===")
print(result.stdout)

if result.stderr:
    print("\n=== Errors ===")
    print(result.stderr)

# Extract results
if "Results:" in result.stdout:
    results_line = [line for line in result.stdout.split('\n') if "Results:" in line][0]
    print(f"\n{results_line}")
    
# Check if we reached level 2
if "level_progression" in result.stdout:
    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        if "level_progression" in line:
            # Print context around it
            for j in range(max(0, i-2), min(len(lines), i+3)):
                print(lines[j])