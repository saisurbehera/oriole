#!/usr/bin/env python3
"""Monitor DreamCoder abstractions in real-time"""

import json
import os
import time
from datetime import datetime


def monitor_library(library_path="dreamcoder_library.json", refresh_rate=5):
    """Monitor the DreamCoder library file for changes"""
    
    print(f"Monitoring {library_path} for abstraction updates...")
    print("Press Ctrl+C to stop\n")
    
    last_modified = None
    last_abstractions = []
    
    while True:
        try:
            if os.path.exists(library_path):
                current_modified = os.path.getmtime(library_path)
                
                if current_modified != last_modified:
                    last_modified = current_modified
                    
                    with open(library_path, 'r') as f:
                        library = json.load(f)
                    
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Library updated!")
                    print(f"Total abstractions: {len(library)}")
                    
                    # Show new abstractions
                    current_names = {a['name'] for a in library}
                    last_names = {a['name'] for a in last_abstractions}
                    new_names = current_names - last_names
                    
                    if new_names:
                        print("\nNEW ABSTRACTIONS LEARNED:")
                        for abstr in library:
                            if abstr['name'] in new_names:
                                print(f"\n  Name: {abstr['name']}")
                                print(f"  Pattern: {' -> '.join(abstr['pattern'])}")
                                print(f"  Success rate: {abstr['success_rate']:.2f}")
                                print(f"  Conditions: {abstr['conditions']}")
                    
                    # Show top 5 most successful
                    print("\nTOP 5 ABSTRACTIONS BY SUCCESS RATE:")
                    sorted_abstractions = sorted(library, key=lambda x: x['success_rate'], reverse=True)[:5]
                    for i, abstr in enumerate(sorted_abstractions, 1):
                        print(f"{i}. {abstr['name']} (rate: {abstr['success_rate']:.2f}, used: {abstr['usage_count']})")
                        print(f"   Pattern: {' -> '.join(abstr['pattern'])}")
                    
                    # Show most used
                    print("\nMOST USED ABSTRACTIONS:")
                    sorted_by_usage = sorted(library, key=lambda x: x['usage_count'], reverse=True)[:5]
                    for i, abstr in enumerate(sorted_by_usage, 1):
                        if abstr['usage_count'] > 0:
                            print(f"{i}. {abstr['name']} (used: {abstr['usage_count']} times)")
                    
                    last_abstractions = library
                    print("\n" + "="*60)
                    
            else:
                print(f"Waiting for {library_path} to be created...")
                
            time.sleep(refresh_rate)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error reading library: {e}")
            time.sleep(refresh_rate)


def check_initial_abstractions():
    """Check what abstractions are built-in"""
    print("\n=== INITIAL BUILT-IN ABSTRACTIONS ===")
    
    # These are the primitives from the code
    primitives = [
        {
            "name": "explore_systematically",
            "pattern": ["north", "east", "south", "west", "search"],
            "conditions": {"location_type": "room|corridor", "no_enemies": "true"},
            "description": "Systematic exploration pattern for rooms/corridors"
        },
        {
            "name": "combat_basic", 
            "pattern": ["wait", "wait"],
            "conditions": {"enemy_present": "true"},
            "description": "Basic combat - let monster approach"
        },
        {
            "name": "seek_stairs",
            "pattern": ["search", "search", "search"],
            "conditions": {"stairs_visible": "false", "explored_percent": "high"},
            "description": "Search for stairs when area is mostly explored"
        }
    ]
    
    for p in primitives:
        print(f"\n{p['name']}:")
        print(f"  Pattern: {' -> '.join(p['pattern'])}")
        print(f"  Conditions: {p['conditions']}")
        print(f"  Description: {p['description']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor DreamCoder abstractions')
    parser.add_argument('--library', default='dreamcoder_library.json', help='Library file path')
    parser.add_argument('--rate', type=int, default=5, help='Refresh rate in seconds')
    parser.add_argument('--show-initial', action='store_true', help='Show initial abstractions')
    
    args = parser.parse_args()
    
    if args.show_initial:
        check_initial_abstractions()
    
    monitor_library(args.library, args.rate)