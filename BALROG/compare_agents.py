#!/usr/bin/env python3

"""
Compare planning-first vs MCTS agents
"""

import subprocess
import json
import time

agents = [
    ("planning_first", "Planning-First Agent"),
    ("feedback_mcts", "Feedback MCTS Agent"),
    ("simple_heuristic", "Simple Heuristic Agent (baseline)")
]

results = {}

for agent_type, agent_name in agents:
    print(f"\n{'='*60}")
    print(f"Testing {agent_name}")
    print('='*60)
    
    cmd = [
        "python3", "eval.py",
        f"agent.type={agent_type}",
        "client.client_name=openai",
        "client.model_id=meta-llama/Llama-3.2-1B-Instruct",
        "client.base_url=http://0.0.0.0:8080/v1",
        "envs.names=nle",
        "eval.num_episodes.nle=5",
        "eval.num_workers=1",
        "eval.save_trajectories=false",
        "eval.max_steps_per_episode=500"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    # Extract results
    if "Overall Average Progression:" in result.stdout:
        for line in result.stdout.split('\n'):
            if "Overall Average Progression:" in line:
                progression = line.split(':')[1].strip().split('%')[0]
                results[agent_name] = {
                    'progression': float(progression),
                    'time': elapsed_time,
                    'episodes': 5
                }
                print(f"✓ Progression: {progression}%")
                print(f"✓ Time: {elapsed_time:.1f}s")
                break
    else:
        results[agent_name] = {
            'progression': 0.0,
            'time': elapsed_time,
            'episodes': 5,
            'error': True
        }
        print("✗ Failed to get results")

# Print comparison
print(f"\n{'='*60}")
print("COMPARISON RESULTS")
print('='*60)
print(f"{'Agent':<30} {'Progression':<15} {'Time (s)':<10} {'Success Rate'}")
print('-'*60)

for agent_name, data in sorted(results.items(), key=lambda x: x[1]['progression'], reverse=True):
    prog = data['progression']
    time_taken = data['time']
    success = "✓" if prog > 0 else "✗"
    print(f"{agent_name:<30} {prog:>10.2f}%     {time_taken:>8.1f}     {success}")

# Save results
with open('agent_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)
    
print(f"\nResults saved to agent_comparison_results.json")