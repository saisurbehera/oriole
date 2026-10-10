#!/usr/bin/env python3
"""Test script for DreamCoder agent with NetHack to track skill improvement over time"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict


def analyze_dreamcoder_results(results_dir: str):
    """Analyze results from DreamCoder agent runs"""
    
    # Find all result files
    result_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json') and 'run_' in file:
                result_files.append(os.path.join(root, file))
    
    # Sort by creation time to get chronological order
    result_files.sort(key=lambda x: os.path.getctime(x))
    
    # Extract metrics
    episodes = []
    for i, file_path in enumerate(result_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        episode_info = {
            'episode_num': i + 1,
            'return': data.get('episode_return', 0),
            'steps': data.get('num_steps', 0),
            'score': data.get('score', 0),
            'turns': data.get('turns', 0),
            'depth': data.get('depth', 0),
            'failed_actions': len(data.get('failed_candidates', [])),
            'task': data.get('task', 'unknown'),
        }
        
        # Calculate efficiency metrics
        if episode_info['steps'] > 0:
            episode_info['score_per_step'] = episode_info['score'] / episode_info['steps']
            episode_info['depth_per_step'] = episode_info['depth'] / episode_info['steps']
        else:
            episode_info['score_per_step'] = 0
            episode_info['depth_per_step'] = 0
            
        episodes.append(episode_info)
    
    return pd.DataFrame(episodes)


def analyze_library_growth(library_path: str = "dreamcoder_library.json"):
    """Analyze how the learned library grows over time"""
    
    if not os.path.exists(library_path):
        print(f"Library file {library_path} not found")
        return None
        
    with open(library_path, 'r') as f:
        library = json.load(f)
        
    # Analyze abstractions
    abstraction_stats = []
    for abstr in library:
        stats = {
            'name': abstr['name'],
            'pattern_length': len(abstr['pattern']),
            'success_rate': abstr['success_rate'],
            'usage_count': abstr['usage_count'],
            'pattern': ' -> '.join(abstr['pattern']),
            'conditions': abstr['conditions']
        }
        abstraction_stats.append(stats)
        
    df = pd.DataFrame(abstraction_stats)
    
    # Print summary
    print("\n=== Learned Abstractions Summary ===")
    print(f"Total abstractions: {len(df)}")
    print(f"Average pattern length: {df['pattern_length'].mean():.2f}")
    print(f"Most used abstractions:")
    print(df.nlargest(5, 'usage_count')[['name', 'pattern', 'usage_count', 'success_rate']])
    
    print(f"\nMost successful abstractions (success_rate > 0.5):")
    successful = df[df['success_rate'] > 0.5].sort_values('success_rate', ascending=False)
    print(successful[['name', 'pattern', 'success_rate', 'usage_count']])
    
    return df


def plot_learning_curves(df: pd.DataFrame, output_dir: str = "dreamcoder_analysis"):
    """Plot various learning curves"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Rolling averages for smoother curves
    window = min(10, len(df) // 4)
    
    # 1. Score over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode_num'], df['score'], alpha=0.3, label='Raw scores')
    plt.plot(df['episode_num'], df['score'].rolling(window).mean(), label=f'{window}-episode average')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('NetHack Score Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'score_over_time.png'))
    plt.close()
    
    # 2. Depth reached over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode_num'], df['depth'], alpha=0.3, label='Raw depth')
    plt.plot(df['episode_num'], df['depth'].rolling(window).mean(), label=f'{window}-episode average')
    plt.xlabel('Episode')
    plt.ylabel('Dungeon Depth')
    plt.title('Dungeon Depth Reached Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'depth_over_time.png'))
    plt.close()
    
    # 3. Efficiency metrics
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode_num'], df['score_per_step'].rolling(window).mean(), label='Score/Step')
    plt.plot(df['episode_num'], df['depth_per_step'].rolling(window).mean(), label='Depth/Step')
    plt.xlabel('Episode')
    plt.ylabel('Efficiency')
    plt.title('Agent Efficiency Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'efficiency_over_time.png'))
    plt.close()
    
    # 4. Failed actions over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode_num'], df['failed_actions'], alpha=0.3, label='Raw failed actions')
    plt.plot(df['episode_num'], df['failed_actions'].rolling(window).mean(), label=f'{window}-episode average')
    plt.xlabel('Episode')
    plt.ylabel('Failed Actions per Episode')
    plt.title('Invalid Actions Over Time (Lower is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'failed_actions_over_time.png'))
    plt.close()
    
    # 5. Summary statistics
    print("\n=== Performance Statistics ===")
    print(f"Total episodes analyzed: {len(df)}")
    print(f"\nFirst 10 episodes:")
    print(f"  Average score: {df.head(10)['score'].mean():.2f}")
    print(f"  Average depth: {df.head(10)['depth'].mean():.2f}")
    print(f"  Average failed actions: {df.head(10)['failed_actions'].mean():.2f}")
    
    if len(df) >= 20:
        print(f"\nLast 10 episodes:")
        print(f"  Average score: {df.tail(10)['score'].mean():.2f}")
        print(f"  Average depth: {df.tail(10)['depth'].mean():.2f}")
        print(f"  Average failed actions: {df.tail(10)['failed_actions'].mean():.2f}")
        
        # Calculate improvement
        early_score = df.head(10)['score'].mean()
        late_score = df.tail(10)['score'].mean()
        improvement = ((late_score - early_score) / max(early_score, 1)) * 100
        print(f"\nScore improvement: {improvement:.1f}%")


def create_run_script():
    """Create a script to run the DreamCoder agent"""
    
    script_content = """#!/bin/bash
# Run DreamCoder agent on NetHack only

# Number of episodes to run
EPISODES=50

echo "Starting DreamCoder evaluation on NetHack..."
echo "This will run $EPISODES episodes to track learning progress"

python eval.py \\
  agent.type=dreamcoder \\
  agent.max_image_history=0 \\
  agent.max_text_history=16 \\
  agent.memory_size=1000 \\
  agent.repetition_penalty_window=20 \\
  agent.library_save_path=dreamcoder_library_nethack.json \\
  eval.num_workers=1 \\
  eval.num_episodes.nle=$EPISODES \\
  eval.num_episodes.minihack=0 \\
  eval.num_episodes.babyai=0 \\
  eval.num_episodes.crafter=0 \\
  eval.num_episodes.textworld=0 \\
  eval.num_episodes.babaisai=0 \\
  envs.names=nle \\
  tasks.nle_tasks=["NetHackChallenge-v0"] \\
  client.client_name=vllm \\
  client.model_id=meta-llama/Llama-3.2-1B-Instruct \\
  client.base_url=http://0.0.0.0:8080/v1

echo "Evaluation complete! Analyzing results..."

# Run analysis
python test_dreamcoder.py --analyze
"""
    
    with open('run_dreamcoder_nethack.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('run_dreamcoder_nethack.sh', 0o755)
    print("Created run_dreamcoder_nethack.sh")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DreamCoder agent')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing results')
    parser.add_argument('--create-script', action='store_true', help='Create run script')
    parser.add_argument('--results-dir', default='outputs', help='Results directory')
    
    args = parser.parse_args()
    
    if args.create_script:
        create_run_script()
    elif args.analyze:
        # Find most recent results
        results_dirs = []
        if os.path.exists(args.results_dir):
            for item in os.listdir(args.results_dir):
                if 'dreamcoder' in item and os.path.isdir(os.path.join(args.results_dir, item)):
                    results_dirs.append(os.path.join(args.results_dir, item))
        
        if not results_dirs:
            print(f"No DreamCoder results found in {args.results_dir}")
            print("Run the agent first with: ./run_dreamcoder_nethack.sh")
        else:
            # Use most recent
            results_dir = max(results_dirs, key=os.path.getctime)
            print(f"Analyzing results from: {results_dir}")
            
            df = analyze_dreamcoder_results(results_dir)
            if not df.empty:
                plot_learning_curves(df)
                print(f"\nPlots saved to: dreamcoder_analysis/")
                
            # Analyze library
            library_files = ['dreamcoder_library_nethack.json', 'dreamcoder_library.json']
            for lib_file in library_files:
                if os.path.exists(lib_file):
                    analyze_library_growth(lib_file)
                    break
    else:
        print("Use --create-script to create run script or --analyze to analyze results")