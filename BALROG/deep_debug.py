#!/usr/bin/env python3

"""
Deep debug to understand environment behavior
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env
import re

def debug_env():
    # Create environment
    class MockConfig:
        def __init__(self):
            self.nle_kwargs = {"character": "@", "max_episode_steps": 100}
            self.agent = type('obj', (object,), {'max_image_history': 0})
            self.envs = type('obj', (object,), {'nle_kwargs': self.nle_kwargs})
    
    env = make_nle_env("nle", "NetHackChallenge-v0", MockConfig())
    
    print("=== Deep Debug Environment ==")
    
    # Reset
    obs, info = env.reset()
    
    print("\n=== Initial State ===")
    print(f"Info: {info}")
    
    # Parse depth from observation
    text = obs.get("text", {})
    long_text = text.get("long_term_context", "")
    
    depth_match = re.search(r'Dlvl:(\d+)', long_text)
    initial_depth = int(depth_match.group(1)) if depth_match else 1
    print(f"Initial depth: {initial_depth}")
    
    # Parse player position
    player_pos = None
    stairs_pos = None
    
    if "map:" in long_text:
        map_start = long_text.find("map:") + 5
        status_markers = ["\nAgent ", "\nDlvl:", "\n$", "\nSt:"]
        map_end = len(long_text)
        for marker in status_markers:
            pos = long_text.find(marker, map_start)
            if pos > 0 and pos < map_end:
                map_end = pos
                
        map_text = long_text[map_start:map_end]
        map_lines = map_text.strip().split('\n')
        
        print(f"\nMap lines:")
        for i, line in enumerate(map_lines[:10]):
            print(f"{i}: {line}")
        
        # Find positions
        for y, line in enumerate(map_lines[:21]):
            if '@' in line:
                player_pos = (line.index('@'), y)
            if '>' in line:
                stairs_pos = (line.index('>'), y)
    
    print(f"Player position: {player_pos}")
    print(f"Stairs position: {stairs_pos}")
    print(f"On stairs: {player_pos == stairs_pos}")
    
    # Try down action
    print("\n=== Trying 'down' action ===")
    obs, reward, done, truncated, info = env.step("down")
    
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")
    
    # Check new state
    text = obs.get("text", {})
    long_text = text.get("long_term_context", "")
    
    depth_match = re.search(r'Dlvl:(\d+)', long_text)
    new_depth = int(depth_match.group(1)) if depth_match else 1
    print(f"New depth: {new_depth}")
    
    # Check message
    if "message:" in long_text:
        msg_start = long_text.find("message:") + 9
        msg_end = long_text.find("\n", msg_start)
        message = long_text[msg_start:msg_end]
        print(f"Message: '{message}'")
    
    # Show the observation structure
    print("\n=== Observation Structure ===")
    print(f"Observation keys: {list(obs.keys())}")
    if "obs" in obs:
        print(f"Raw obs keys: {list(obs['obs'].keys())}")
    
    # Check if it's actually NetHack or a different game
    if "obs" in obs and "glyphs" in obs["obs"]:
        import numpy as np
        glyphs = obs["obs"]["glyphs"]
        print(f"Glyphs shape: {glyphs.shape}")
        print(f"Unique glyphs: {np.unique(glyphs)[:20]}")
        
        # Look for stairs in glyphs
        stairs_glyph = np.where(glyphs == 2378)  # '>' glyph
        if len(stairs_glyph[0]) > 0:
            print(f"Stairs found in glyphs at: {list(zip(stairs_glyph[1], stairs_glyph[0]))}")
        else:
            print("No stairs found in glyphs")

if __name__ == "__main__":
    debug_env()