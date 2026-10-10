#!/usr/bin/env python3

"""
Test the working MCTS agent
"""

import subprocess
import json
import time

print("=== Testing Working MCTS Agent ===")

# Run evaluation
cmd = [
    "python3", "eval.py",
    "agent.type=mcts_working",
    "client.client_name=openai", 
    "client.model_id=meta-llama/Llama-3.2-1B-Instruct",
    "client.base_url=http://0.0.0.0:8080/v1",
    "envs.names=nle",
    "eval.num_episodes.nle=10",
    "eval.num_workers=1",
    "eval.save_trajectories=false",
    "eval.max_steps_per_episode=1000"
]

print("Running:", " ".join(cmd))
start_time = time.time()

result = subprocess.run(cmd, capture_output=True, text=True)

elapsed = time.time() - start_time

# Parse results
print("\n=== Results ===")
if "Overall Average Progression:" in result.stdout:
    for line in result.stdout.split('\n'):
        if "Overall Average Progression:" in line:
            progression = line.split(':')[1].strip().split('%')[0]
            print(f"Progression: {progression}%")
            
            if float(progression) > 0:
                print("🎉 SUCCESS! Agent reached deeper levels!")
            else:
                print("❌ Still stuck on level 1")
                
print(f"Time: {elapsed:.1f}s")

# Check for errors
if result.stderr:
    print("\n=== Errors ===")
    for line in result.stderr.split('\n')[-5:]:
        if line.strip():
            print(line)