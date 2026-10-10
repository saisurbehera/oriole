#!/usr/bin/env python3

"""
Debug environment messages to understand why down isn't working
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env
import numpy as np

def debug_messages():
    # Create environment
    class MockConfig:
        def __init__(self):
            self.nle_kwargs = {"character": "@", "max_episode_steps": 100}
            self.agent = type('obj', (object,), {'max_image_history': 0})
            self.envs = type('obj', (object,), {'nle_kwargs': self.nle_kwargs})
    
    env = make_nle_env("nle", "NetHackChallenge-v0", MockConfig())
    
    print("=== Environment Message Debug ===")
    
    # Reset
    obs, info = env.reset()
    
    # Find where player starts and where stairs are
    if "obs" in obs and "glyphs" in obs["obs"]:
        glyphs = obs["obs"]["glyphs"]
        tty_cursor = obs["obs"]["tty_cursor"]
        
        print(f"Player cursor: {tty_cursor}")
        
        # Find all stairs
        stairs_glyphs = np.where(glyphs == 2378)
        print(f"Stairs positions: {list(zip(stairs_glyphs[1], stairs_glyphs[0]))}")
        
        # Check what's at cursor position
        cursor_glyph = glyphs[tty_cursor[0], tty_cursor[1]]
        print(f"Glyph at cursor: {cursor_glyph}")
        
        # Is the player on stairs?
        on_stairs = cursor_glyph == 2378
        print(f"Player on stairs: {on_stairs}")
    
    # Try a few actions and see the messages
    actions_to_try = ["down", ">", "search", "look", "wait"]
    
    for action in actions_to_try:
        print(f"\n--- Trying action: {action} ---")
        
        obs, reward, done, truncated, info = env.step(action)
        
        # Get the message
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        
        if "message:" in long_text:
            msg_start = long_text.find("message:") + 9
            msg_end = long_text.find("\n", msg_start)
            if msg_end == -1:
                msg_end = len(long_text)
            message = long_text[msg_start:msg_end]
            print(f"Message: '{message}'")
        
        print(f"Reward: {reward}")
        
        # Check if depth changed
        if "Dlvl:" in long_text:
            import re
            depth_match = re.search(r'Dlvl:(\d+)', long_text)
            if depth_match:
                depth = int(depth_match.group(1))
                print(f"Depth: {depth}")
        
        # Check player position
        if "obs" in obs and "glyphs" in obs["obs"]:
            tty_cursor = obs["obs"]["tty_cursor"]
            glyphs = obs["obs"]["glyphs"]
            cursor_glyph = glyphs[tty_cursor[0], tty_cursor[1]]
            print(f"New cursor: {tty_cursor}, glyph: {cursor_glyph}")
            
            # Is the player still on stairs?
            on_stairs = cursor_glyph == 2378
            print(f"Still on stairs: {on_stairs}")
            
        if done or truncated:
            print("Episode ended!")
            break
    
    # Show full observation structure
    print("\n=== Full Text Context ===")
    text = obs.get("text", {})
    long_text = text.get("long_term_context", "")
    print(long_text[:1000])  # First 1000 chars

if __name__ == "__main__":
    debug_messages()