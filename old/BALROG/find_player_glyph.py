#!/usr/bin/env python3

"""
Find the correct glyph value for the player
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env
import numpy as np

def find_player_glyph():
    # Create environment
    class MockConfig:
        def __init__(self):
            self.nle_kwargs = {"character": "@", "max_episode_steps": 100}
            self.agent = type('obj', (object,), {'max_image_history': 0})
            self.envs = type('obj', (object,), {'nle_kwargs': self.nle_kwargs})
    
    env = make_nle_env("nle", "NetHackChallenge-v0", MockConfig())
    obs, _ = env.reset()
    
    # Get map text
    long_text = obs["text"]["long_term_context"]
    map_start = long_text.index("map:") + 4
    map_text = long_text[map_start:].split("\n\n")[0]
    map_lines = map_text.strip().split('\n')
    
    # Find @ in map
    player_text_pos = None
    for y, line in enumerate(map_lines):
        if '@' in line:
            x = line.index('@')
            player_text_pos = (x, y)
            print(f"Player @ found in text at: {player_text_pos}")
            break
    
    if player_text_pos:
        # Get glyph at that position
        glyphs = obs["obs"]["glyphs"]
        px, py = player_text_pos
        player_glyph = glyphs[py, px]
        print(f"Glyph at player position: {player_glyph}")
        
        # Verify by checking all positions of this glyph
        positions = np.where(glyphs == player_glyph)
        print(f"\nAll positions with glyph {player_glyph}:")
        for i in range(len(positions[0])):
            y, x = positions[0][i], positions[1][i]
            print(f"  ({x}, {y})")
            
        # Show glyph distribution
        print(f"\nGlyph analysis:")
        unique, counts = np.unique(glyphs, return_counts=True)
        for g, c in zip(unique, counts):
            if c == 1:  # Player should be unique
                print(f"  Glyph {g}: appears {c} time (potential player)")

if __name__ == "__main__":
    find_player_glyph()