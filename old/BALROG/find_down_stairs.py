#!/usr/bin/env python3

"""
Find downward stairs in the environment
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env
import numpy as np

def find_down_stairs():
    # Create environment
    class MockConfig:
        def __init__(self):
            self.nle_kwargs = {"character": "@", "max_episode_steps": 100}
            self.agent = type('obj', (object,), {'max_image_history': 0})
            self.envs = type('obj', (object,), {'nle_kwargs': self.nle_kwargs})
    
    env = make_nle_env("nle", "NetHackChallenge-v0", MockConfig())
    
    print("=== Finding Down Stairs ===")
    
    # Reset
    obs, info = env.reset()
    
    # Look for all possible stair glyphs
    if "obs" in obs and "glyphs" in obs["obs"]:
        glyphs = obs["obs"]["glyphs"]
        unique_glyphs = np.unique(glyphs)
        
        print(f"All unique glyphs: {unique_glyphs}")
        
        # Let's check what each glyph represents
        # Common NetHack glyphs:
        # 2378 = '>' (stairs up or down)
        # 2379 = '<' (stairs up)
        # And there might be other values
        
        # Look for '<' character glyphs
        down_stairs_candidates = [2379]  # '<' character
        
        for candidate in down_stairs_candidates:
            positions = np.where(glyphs == candidate)
            if len(positions[0]) > 0:
                print(f"Found glyph {candidate} at positions: {list(zip(positions[1], positions[0]))}")
        
        # Also check for any other stairs-like glyphs
        potential_stairs = [2378, 2379, 2380, 2381, 2382]
        for glyph_num in potential_stairs:
            if glyph_num in unique_glyphs:
                positions = np.where(glyphs == glyph_num)
                print(f"Glyph {glyph_num} found at: {list(zip(positions[1], positions[0]))}")
    
    # Try to move around and explore to find downward stairs
    print("\n=== Exploring for down stairs ===")
    
    # Try some basic movements to explore
    movements = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
    
    for i, movement in enumerate(movements):
        if i > 10:  # Don't explore too long
            break
            
        print(f"\nTrying movement: {movement}")
        
        obs, reward, done, truncated, info = env.step(movement)
        
        # Check if we found any different stairs
        if "obs" in obs and "glyphs" in obs["obs"]:
            glyphs = obs["obs"]["glyphs"]
            tty_cursor = obs["obs"]["tty_cursor"]
            
            # Look for any stairs near the player
            player_y, player_x = tty_cursor
            
            # Check a 3x3 area around player
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = player_y + dy, player_x + dx
                    if 0 <= ny < glyphs.shape[0] and 0 <= nx < glyphs.shape[1]:
                        glyph = glyphs[ny, nx]
                        if glyph in [2378, 2379]:  # Potential stairs
                            print(f"  Stairs glyph {glyph} found at ({nx}, {ny})")
        
        # Check message to see if we found down stairs
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        
        if "message:" in long_text:
            msg_start = long_text.find("message:") + 9
            msg_end = long_text.find("\n", msg_start)
            if msg_end == -1:
                msg_end = len(long_text)
            message = long_text[msg_start:msg_end]
            if "staircase" in message.lower():
                print(f"  Found stairs message: '{message}'")
        
        if done or truncated:
            print("Episode ended!")
            break
    
    # Try the 'look' action to see what's around
    print("\n=== Final look around ===")
    obs, reward, done, truncated, info = env.step("look")
    
    text = obs.get("text", {})
    long_text = text.get("long_term_context", "")
    print(f"Look result: {long_text[-500:]}")  # Last 500 chars

if __name__ == "__main__":
    find_down_stairs()