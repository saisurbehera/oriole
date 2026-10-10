#!/usr/bin/env python3

"""
Test the final MCTS implementation
"""

import subprocess
import time

print("=== Testing Final MCTS Implementation ===")

cmd = [
    "python3", "eval.py",
    "agent.type=mcts_final",
    "client.client_name=openai", 
    "client.model_id=meta-llama/Llama-3.2-1B-Instruct",
    "client.base_url=http://0.0.0.0:8080/v1",
    "envs.names=nle",
    "eval.num_episodes.nle=10",
    "eval.num_workers=1",
    "eval.save_trajectories=false",
    "eval.max_steps_per_episode=1000"
]

print("Command:", " ".join(cmd))
print("\nRunning evaluation...")

start = time.time()
result = subprocess.run(cmd, capture_output=True, text=True)
elapsed = time.time() - start

# Parse results
print("\n=== Results ===")
for line in result.stdout.split('\n'):
    if "Overall Average Progression:" in line:
        print(line)
        progression = float(line.split(':')[1].strip().split('%')[0])
        
        if progression > 0:
            print(f"\n🎉 SUCCESS! MCTS agent achieved {progression}% progression!")
            print("The agent can navigate to deeper dungeon levels!")
        else:
            print("\n❌ Still 0% progression")
            
print(f"\nTime taken: {elapsed:.1f}s")

# Show any errors
if result.stderr:
    print("\n=== Errors ===")
    print(result.stderr)