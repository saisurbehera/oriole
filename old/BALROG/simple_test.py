#!/usr/bin/env python3

"""
Simple test - just use random exploration to find and use downward stairs
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env
import random
import re

def simple_test():
    # Create environment
    class MockConfig:
        def __init__(self):
            self.nle_kwargs = {"character": "@", "max_episode_steps": 1000}
            self.agent = type('obj', (object,), {'max_image_history': 0})
            self.envs = type('obj', (object,), {'nle_kwargs': self.nle_kwargs})
    
    env = make_nle_env("nle", "NetHackChallenge-v0", MockConfig())
    
    print("=== Simple Random Exploration Test ===")
    
    # Reset
    obs, info = env.reset()
    
    actions = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
    
    for step in range(500):
        # Get current depth
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        
        depth_match = re.search(r'Dlvl:(\d+)', long_text)
        depth = int(depth_match.group(1)) if depth_match else 1
        
        # Check for down stairs message
        message = ""
        if "message:" in long_text:
            msg_start = long_text.find("message:") + 9
            msg_end = long_text.find("\n", msg_start)
            if msg_end == -1:
                msg_end = len(long_text)
            message = long_text[msg_start:msg_end].strip()
        
        # Look for down stairs in the text
        down_stairs_found = False
        if "map:" in long_text:
            map_section = long_text[long_text.find("map:"):]
            if "<" in map_section:
                down_stairs_found = True
                
        # Decide action
        if "staircase down" in message.lower():
            action = "down"
            print(f"Step {step}: Found down stairs message! Trying down")
        elif down_stairs_found and step % 5 == 0:  # Try down occasionally when we see stairs
            action = "down"
            print(f"Step {step}: Trying down (< found in map)")
        else:
            action = random.choice(actions)
            
        # Take action
        obs, reward, done, truncated, info = env.step(action)
        
        # Check new depth
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        
        depth_match = re.search(r'Dlvl:(\d+)', long_text)
        new_depth = int(depth_match.group(1)) if depth_match else 1
        
        if new_depth > depth:
            print(f"\n🎉 SUCCESS! Reached level {new_depth} at step {step}!")
            
            # Show final score
            score_match = re.search(r'S:(\d+)', long_text)
            score = int(score_match.group(1)) if score_match else 0
            print(f"Final score: {score}")
            break
            
        if done or truncated:
            print(f"\nEpisode ended at step {step}, final depth={new_depth}")
            break
            
        # Print progress occasionally
        if step % 50 == 0:
            print(f"Step {step}: depth={depth}, action={action}")
            if message:
                print(f"  Message: {message}")

if __name__ == "__main__":
    simple_test()