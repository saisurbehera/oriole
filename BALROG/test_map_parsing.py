#!/usr/bin/env python3

"""
Test map parsing to see why stairs aren't found
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env

def test_map_parsing():
    # Create environment
    class MockConfig:
        def __init__(self):
            self.nle_kwargs = {"character": "@", "max_episode_steps": 1000}
            self.agent = type('obj', (object,), {'max_image_history': 0})
            self.envs = type('obj', (object,), {'nle_kwargs': self.nle_kwargs})
    
    env = make_nle_env("nle", "NetHackChallenge-v0", MockConfig())
    
    print("=== Map Parsing Test ===")
    
    # Search for stairs
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        obs, _ = env.reset()
        
        for step in range(30):
            # Get observation
            text = obs.get("text", {})
            long_text = text.get("long_term_context", "")
            
            # Check language observation
            lang_obs = ""
            if "language observation:" in long_text:
                lang_start = long_text.index("language observation:")
                lang_section = long_text[lang_start:].split("\n\n")[0]
                lang_obs = lang_section.replace("language observation:\n", "")
            
            if "stairs" in lang_obs.lower():
                print(f"\n  Step {step}: Stairs mentioned in language!")
                print(f"  Language: {lang_obs[:100]}...")
                
                # Get map section
                if "map:" in long_text:
                    map_start = long_text.index("map:") + 4
                    map_section = long_text[map_start:]
                    map_end = map_section.find("\n\n")
                    if map_end > 0:
                        map_section = map_section[:map_end]
                    
                    print("\n  Map text:")
                    map_lines = map_section.strip().split('\n')
                    for i, line in enumerate(map_lines[:10]):
                        if '@' in line or '>' in line:
                            print(f"    Line {i}: {repr(line[:50])}")
                    
                    # Check if > is in map
                    has_stairs = any('>' in line for line in map_lines)
                    print(f"\n  Has '>' in map: {has_stairs}")
                    
                    # Show raw glyphs at stairs location
                    glyphs = obs["obs"]["glyphs"]
                    import numpy as np
                    stairs_glyphs = np.where(glyphs == 2378)
                    if len(stairs_glyphs[0]) > 0:
                        print(f"  Stairs glyph 2378 found at: {list(zip(stairs_glyphs[1][:5], stairs_glyphs[0][:5]))}")
                
                break
            
            # Random action
            if step < 5:
                action = "search"
            else:
                action = ["south", "east", "north", "west"][step % 4]
                
            obs, _, done, _, _ = env.step(action)
            if done:
                break

if __name__ == "__main__":
    test_map_parsing()