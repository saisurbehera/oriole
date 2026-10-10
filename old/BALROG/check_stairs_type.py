#!/usr/bin/env python3

"""
Check if we have down stairs or up stairs
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env
import numpy as np

def check_stairs():
    # Create environment
    class MockConfig:
        def __init__(self):
            self.nle_kwargs = {"character": "@", "max_episode_steps": 100}
            self.agent = type('obj', (object,), {'max_image_history': 0})
            self.envs = type('obj', (object,), {'nle_kwargs': self.nle_kwargs})
    
    env = make_nle_env("nle", "NetHackChallenge-v0", MockConfig())
    
    print("=== Checking Stairs Types ===")
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        obs, _ = env.reset()
        
        # Get glyphs
        glyphs = obs["obs"]["glyphs"]
        
        # Check for stairs
        down_stairs = np.where(glyphs == 2378)  # >
        up_stairs = np.where(glyphs == 2377)    # <
        
        if len(down_stairs[0]) > 0:
            print(f"  ✓ Found DOWN stairs (>) at {len(down_stairs[0])} positions")
            
        if len(up_stairs[0]) > 0:
            print(f"  ✓ Found UP stairs (<) at {len(up_stairs[0])} positions")
            
        # Check text
        long_text = obs["text"]["long_term_context"]
        if "map:" in long_text:
            map_start = long_text.find("map:") + 5
            map_end = long_text.find("\nAgent", map_start)
            if map_end > 0:
                map_text = long_text[map_start:map_end]
                
                if '>' in map_text:
                    print("  ✓ Found '>' in text map")
                if '<' in map_text:
                    print("  ✓ Found '<' in text map")
                    
        # Check language observation
        if "language observation:" in long_text:
            lang_start = long_text.find("language observation:")
            lang_end = long_text.find("\n\n", lang_start)
            lang_obs = long_text[lang_start:lang_end]
            
            if "stairs down" in lang_obs:
                print("  ✓ Language mentions 'stairs down'")
            if "stairs up" in lang_obs:
                print("  ✓ Language mentions 'stairs up'")
            if "staircase down" in lang_obs:
                print("  ✓ Language mentions 'staircase down'")

if __name__ == "__main__":
    check_stairs()