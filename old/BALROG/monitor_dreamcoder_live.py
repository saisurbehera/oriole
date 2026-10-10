#!/usr/bin/env python3
"""Live monitor for DreamCoder agent progress"""

import json
import os
import time
from datetime import datetime
import glob


def get_latest_results_dir():
    """Find the most recent dreamcoder results directory"""
    pattern = "results/*dreamcoder*"
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getctime)


def analyze_episode(json_file):
    """Extract key metrics from episode JSON"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        return {
            'score': data.get('score', 0),
            'depth': data.get('depth', 0),
            'steps': data.get('num_steps', 0),
            'return': data.get('episode_return', 0),
            'actions': data.get('action_frequency', {}),
            'task': data.get('task', 'unknown')
        }
    except:
        return None


def monitor_progress():
    """Monitor DreamCoder progress in real-time"""
    print("=== DreamCoder Live Monitor ===")
    print("Watching for new episodes...\n")
    
    seen_files = set()
    episode_count = 0
    total_score = 0
    total_depth = 0
    max_score = 0
    max_depth = 0
    
    while True:
        results_dir = get_latest_results_dir()
        if not results_dir:
            print("Waiting for results directory...")
            time.sleep(5)
            continue
        
        # Find all episode files
        json_files = glob.glob(f"{results_dir}/**/NetHackChallenge-v0_run_*.json", recursive=True)
        
        # Process new files
        new_files = set(json_files) - seen_files
        
        for json_file in sorted(new_files):
            episode_data = analyze_episode(json_file)
            if episode_data:
                episode_count += 1
                score = episode_data['score']
                depth = episode_data['depth']
                steps = episode_data['steps']
                
                total_score += score
                total_depth += depth
                max_score = max(max_score, score)
                max_depth = max(max_depth, depth)
                
                # Get top actions
                actions = episode_data['actions']
                top_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)[:3]
                action_str = ', '.join([f"{a}:{c}" for a, c in top_actions])
                
                # Print episode summary
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Episode {episode_count}:")
                print(f"  Score: {score}, Depth: {depth}, Steps: {steps}")
                print(f"  Top actions: {action_str}")
                
                if score > 0 or depth > 0:
                    print(f"  >>> PROGRESS! Score: {score}, Depth: {depth} <<<")
                
                # Print running statistics
                if episode_count % 5 == 0:
                    avg_score = total_score / episode_count
                    avg_depth = total_depth / episode_count
                    print(f"\n--- Stats after {episode_count} episodes ---")
                    print(f"  Average score: {avg_score:.2f}")
                    print(f"  Average depth: {avg_depth:.2f}")
                    print(f"  Max score: {max_score}")
                    print(f"  Max depth: {max_depth}")
                    print("-" * 40)
                
                seen_files.add(json_file)
        
        # Check for library file
        library_files = glob.glob("dreamcoder_library*.json")
        if library_files:
            for lib_file in library_files:
                try:
                    with open(lib_file, 'r') as f:
                        library = json.load(f)
                    print(f"\n[LIBRARY UPDATE] {len(library)} abstractions in {lib_file}")
                    
                    # Show newest abstractions
                    if library:
                        newest = sorted(library, key=lambda x: x.get('usage_count', 0), reverse=True)[:3]
                        for abstr in newest:
                            print(f"  - {abstr['name']}: {' -> '.join(abstr['pattern'])}")
                except:
                    pass
        
        time.sleep(2)


if __name__ == "__main__":
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")