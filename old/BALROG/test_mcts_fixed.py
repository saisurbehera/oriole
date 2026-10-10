#!/usr/bin/env python3

"""
Test the fixed MCTS agent
"""

import subprocess
import json

# Run the fixed MCTS agent
cmd = [
    "python3", "eval.py",
    "agent.type=feedback_mcts_fixed",
    "client.client_name=openai",
    "client.model_id=meta-llama/Llama-3.2-1B-Instruct",
    "client.base_url=http://0.0.0.0:8080/v1",
    "envs.names=nle",
    "eval.num_episodes.nle=10",
    "eval.num_workers=1",
    "eval.save_trajectories=true",
    "eval.max_steps_per_episode=1000"
]

print("Testing fixed MCTS agent...")
print(" ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True)

print("\n=== Output ===")
# Extract key results
if "Overall Average Progression:" in result.stdout:
    for line in result.stdout.split('\n'):
        if "Overall Average Progression:" in line:
            print(line)
        elif "nle:" in line and "Episodes:" in line:
            print(line)
            
# Check for level 2 achievements
output_lines = result.stdout.split('\n')
level_2_count = 0
for line in output_lines:
    if "Dlvl:2" in line or "level 2" in line.lower():
        level_2_count += 1
        
if level_2_count > 0:
    print(f"\n🎉 Reached level 2 in some episodes! (mentions: {level_2_count})")
else:
    print("\n❌ Did not reach level 2")

# Show any errors
if result.stderr:
    print("\n=== Errors ===")
    error_lines = result.stderr.split('\n')
    for line in error_lines[-10:]:  # Last 10 error lines
        if line.strip():
            print(line)