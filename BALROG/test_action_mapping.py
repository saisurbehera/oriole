#!/usr/bin/env python3

"""
Test action mapping for down stairs
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env

def test_actions():
    # Create environment
    class MockConfig:
        def __init__(self):
            self.nle_kwargs = {"character": "@", "max_episode_steps": 100}
            self.agent = type('obj', (object,), {'max_image_history': 0})
            self.envs = type('obj', (object,), {'nle_kwargs': self.nle_kwargs})
    
    env = make_nle_env("nle", "NetHackChallenge-v0", MockConfig())
    
    print("=== Testing Action Mapping ===")
    
    # Reset
    obs, _ = env.reset()
    
    # Test various actions
    test_actions = ["north", "south", "east", "west", "down", ">", "wait", "search"]
    
    for action in test_actions:
        try:
            obs, reward, done, _, _ = env.step(action)
            print(f"✓ Action '{action}' works")
        except Exception as e:
            print(f"✗ Action '{action}' failed: {e}")
            
    # Check action space
    print(f"\nAction space type: {type(env.action_space)}")
    if hasattr(env.action_space, 'actions'):
        print(f"Available actions: {env.action_space.actions[:10]}...")

if __name__ == "__main__":
    test_actions()