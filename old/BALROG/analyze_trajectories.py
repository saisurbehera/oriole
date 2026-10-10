#!/usr/bin/env python3
import csv
import json
import os
from collections import Counter, defaultdict

def analyze_csv_trajectory(filepath):
    """Analyze a single CSV trajectory file."""
    actions = []
    reasoning = []
    invalid_actions = 0
    total_steps = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            # Read lines and handle potential CSV issues
            lines = f.readlines()
            if not lines or not lines[0].startswith('Step,Action,Reasoning'):
                return None
                
            # Parse manually to handle malformed CSV
            for i, line in enumerate(lines[1:]):  # Skip header
                try:
                    # Simple split since commas might appear in observations
                    parts = line.strip().split(',', 5)  # Split into max 6 parts
                    if len(parts) >= 3:
                        total_steps += 1
                        action = parts[1]
                        reason = parts[2]
                        
                        # Check if this is an invalid action message
                        if "did not contain a valid action" in line:
                            invalid_actions += 1
                        else:
                            actions.append(action)
                            reasoning.append(reason)
                except:
                    continue
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None
    
    return {
        'total_steps': total_steps,
        'invalid_actions': invalid_actions,
        'valid_actions': len(actions),
        'action_distribution': Counter(actions),
        'reasoning_patterns': Counter(reasoning)
    }

def analyze_all_trajectories(base_dir):
    """Analyze all trajectory files in a directory."""
    results = defaultdict(lambda: {
        'files_analyzed': 0,
        'total_steps': 0,
        'total_invalid': 0,
        'action_distribution': Counter(),
        'reasoning_distribution': Counter(),
        'invalid_action_examples': []
    })
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv') and 'run_' in file:
                filepath = os.path.join(root, file)
                agent_type = 'hierarchical' if 'hierarchical_rl_results' in filepath else 'expert_guided'
                
                print(f"\nAnalyzing {filepath}...")
                analysis = analyze_csv_trajectory(filepath)
                
                if analysis is None:
                    continue
                    
                results[agent_type]['files_analyzed'] += 1
                results[agent_type]['total_steps'] += analysis['total_steps']
                results[agent_type]['total_invalid'] += analysis['invalid_actions']
                results[agent_type]['action_distribution'].update(analysis['action_distribution'])
                results[agent_type]['reasoning_distribution'].update(analysis['reasoning_patterns'])
                
                # Collect examples of what causes invalid actions
                if analysis['invalid_actions'] > 0:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if "did not contain a valid action" in line and i > 0:
                                # Get the previous line which contains the invalid action
                                prev_line = lines[i-1].strip()
                                if prev_line and agent_type == 'expert_guided':
                                    results[agent_type]['invalid_action_examples'].append(prev_line[:100])
                
    return results

def main():
    print("Analyzing NetHack trajectory files...")
    
    # Analyze both directories
    hierarchical_dir = "/home/sai/Desktop/oriole/BALROG/hierarchical_rl_results"
    expert_dir = "/home/sai/Desktop/oriole/BALROG/expert_guided_rl_results_v2"
    
    all_results = {}
    
    if os.path.exists(hierarchical_dir):
        hier_results = analyze_all_trajectories(hierarchical_dir)
        all_results.update(hier_results)
    
    if os.path.exists(expert_dir):
        expert_results = analyze_all_trajectories(expert_dir)
        all_results.update(expert_results)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF LLM OUTPUTS AND FAILURES")
    print("="*80)
    
    for agent_type, data in all_results.items():
        print(f"\n{agent_type.upper()} AGENT:")
        print(f"  Files analyzed: {data['files_analyzed']}")
        print(f"  Total steps: {data['total_steps']}")
        print(f"  Invalid actions: {data['total_invalid']} ({data['total_invalid']/data['total_steps']*100:.1f}%)")
        
        print(f"\n  Top 10 Actions:")
        for action, count in data['action_distribution'].most_common(10):
            print(f"    {action}: {count}")
        
        print(f"\n  Top 10 Reasoning Patterns:")
        for reason, count in data['reasoning_distribution'].most_common(10):
            print(f"    {reason}: {count}")
        
        if data['invalid_action_examples']:
            print(f"\n  Examples of Invalid Actions:")
            for example in data['invalid_action_examples'][:5]:
                print(f"    {example}")

if __name__ == "__main__":
    main()