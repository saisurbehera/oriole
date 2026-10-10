#!/usr/bin/env python3

"""
Show a full observation to understand the format
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env

def show_observation():
    # Create environment
    class MockConfig:
        def __init__(self):
            self.nle_kwargs = {"character": "@", "max_episode_steps": 100}
            self.agent = type('obj', (object,), {'max_image_history': 0})
            self.envs = type('obj', (object,), {'nle_kwargs': self.nle_kwargs})
    
    env = make_nle_env("nle", "NetHackChallenge-v0", MockConfig())
    obs, _ = env.reset()
    
    print("=== Full Observation Structure ===\n")
    
    # Get long_term_context
    long_text = obs["text"]["long_term_context"]
    
    print("Full long_term_context:")
    print("=" * 80)
    print(long_text)
    print("=" * 80)
    
    # Show where each section starts
    print("\nSection markers:")
    sections = ["message:", "language observation:", "map:", "Agent", "Dlvl:", "HP:"]
    for section in sections:
        if section in long_text:
            idx = long_text.index(section)
            print(f"  '{section}' at position {idx}")

if __name__ == "__main__":
    show_observation()