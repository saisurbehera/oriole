#!/usr/bin/env python3

"""
Test the exact action mapping for stairs
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env

def test_stairs():
    # Create environment config
    class MockEnvConfig:
        def __init__(self):
            self.nle_kwargs = {
                "character": "@",
                "max_episode_steps": 200,
                "no_progress_timeout": 100,
                "savedir": None,
                "save_ttyrec_every": 0,
                "skip_more": True
            }
    
    class MockAgentConfig:
        def __init__(self):
            self.max_image_history = 0
    
    class MockFullConfig:
        def __init__(self):
            self.envs = MockEnvConfig()
            self.agent = MockAgentConfig()
    
    # Create environment
    full_config = MockFullConfig()
    env = make_nle_env("nle", "NetHackChallenge-v0", full_config)
    
    print("=== Test Stairs Action Mapping ===")
    
    # Check action meanings
    raw_env = env.env.env.env  # Unwrap to get to NLE env
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")
    
    # Map action names to indices
    action_map = {}
    for action_name in ['north', 'south', 'east', 'west', 'wait', 'search', '>', '<', 'open', 'kick']:
        action_idx = env._action_ids.get(action_name)
        if action_idx is not None:
            action_map[action_name] = action_idx
            print(f"'{action_name}' -> index {action_idx}")
    
    # Reset and explore
    obs = env.reset()
    
    # Manual search pattern
    search_pattern = ['search'] * 5 + ['south'] * 3 + ['east'] * 3 + ['search'] * 5
    
    for i in range(100):
        # Get current state info
        text = obs.get("text", {})
        lang_obs = text.get("language_observation", "")
        
        # Check if stairs visible
        if "staircase down" in lang_obs.lower() or "stairs down" in lang_obs.lower() or ">" in lang_obs:
            print(f"\n🔍 STAIRS FOUND at step {i}!")
            print(f"Language observation: {lang_obs}")
            
            # Get player position from long text
            long_text = text.get("long_term_context", "")
            map_section = long_text.split("map:")[-1].split("Agent")[0] if "map:" in long_text else ""
            lines = map_section.strip().split('\n')[:21]
            
            # Find player and stairs
            player_y, player_x = None, None
            stairs_y, stairs_x = None, None
            
            for y, line in enumerate(lines):
                if '@' in line:
                    player_x = line.index('@')
                    player_y = y
                if '>' in line:
                    stairs_x = line.index('>')
                    stairs_y = y
                    
            if player_y is not None and stairs_y is not None:
                print(f"Player at ({player_x}, {player_y}), stairs at ({stairs_x}, {stairs_y})")
                
                # Move to stairs if needed
                while player_x != stairs_x or player_y != stairs_y:
                    if player_x < stairs_x:
                        action = 'east'
                    elif player_x > stairs_x:
                        action = 'west'
                    elif player_y < stairs_y:
                        action = 'south'
                    elif player_y > stairs_y:
                        action = 'north'
                    
                    print(f"Moving {action} towards stairs...")
                    obs, _, done, _ = env.step(action)
                    
                    # Update position
                    long_text = obs.get("text", {}).get("long_term_context", "")
                    map_section = long_text.split("map:")[-1].split("Agent")[0] if "map:" in long_text else ""
                    lines = map_section.strip().split('\n')[:21]
                    
                    for y, line in enumerate(lines):
                        if '@' in line:
                            player_x = line.index('@')
                            player_y = y
                            
                    if done:
                        break
                
                # Now on stairs, try '>'
                print("\n✅ On stairs! Trying '>' action...")
                print(f"'>' action index: {action_map.get('>')}")
                
                obs, reward, done, info = env.step('>')
                
                # Check result
                new_text = obs.get("text", {})
                new_lang_obs = new_text.get("language_observation", "")
                new_long_text = new_text.get("long_term_context", "")
                
                print(f"\nResult:")
                print(f"  Reward: {reward}")
                print(f"  New observation: {new_lang_obs}")
                
                if "Dlvl:2" in new_long_text:
                    print("  🎉 SUCCESS! Reached level 2!")
                else:
                    print("  ❌ Still on level 1")
                    
                    # Check for messages
                    if "message" in new_text:
                        print(f"  Message: {new_text['message']}")
                
                return
        
        # Continue exploring
        if i < len(search_pattern):
            action = search_pattern[i]
        else:
            action = ['north', 'south', 'east', 'west'][i % 4]
            
        obs, _, done, _ = env.step(action)
        
        if done:
            break
    
    print("\nDid not find stairs in 100 steps")

if __name__ == "__main__":
    test_stairs()