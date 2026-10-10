#!/usr/bin/env python3

"""
Check the actual structure of NetHack observations
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env
import json

def check_observation():
    # Create environment config
    class MockEnvConfig:
        def __init__(self):
            self.nle_kwargs = {
                "character": "@",
                "max_episode_steps": 1000,
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
    
    print("=== NetHack Observation Structure ===")
    
    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    
    # Show observation keys
    print("\n1. Observation keys:", list(obs.keys()))
    
    # Check text structure
    if "text" in obs:
        print("\n2. Text keys:", list(obs["text"].keys()))
        
        # Show each text component
        for key, value in obs["text"].items():
            print(f"\n3. obs['text']['{key}']:")
            if isinstance(value, str):
                # Show first 200 chars
                print(f"   Length: {len(value)}")
                print(f"   Preview: {repr(value[:200])}")
                
                # Check for map markers
                if "map:" in value:
                    print("   ✓ Contains 'map:' marker")
                if "@" in value:
                    print("   ✓ Contains '@' (player)")
                if ">" in value:
                    print("   ✓ Contains '>' (down stairs)")
                if "<" in value:
                    print("   ✓ Contains '<' (up stairs)")
    
    # Check obs sub-key
    if "obs" in obs:
        print(f"\n4. obs['obs'] keys: {list(obs['obs'].keys())}")
        
        # Check glyphs in obs['obs']
        if "glyphs" in obs['obs']:
            print(f"\n   Glyphs shape: {obs['obs']['glyphs'].shape}")
        
            # Look for stairs glyph
            import numpy as np
            unique_glyphs = np.unique(obs['obs']['glyphs'])
            print(f"   Unique glyphs: {len(unique_glyphs)}")
            
            # Common glyph values
            STAIRS_DOWN = 2378  # '>'
            STAIRS_UP = 2377    # '<'
            PLAYER = 64         # '@'
            
            if STAIRS_DOWN in unique_glyphs:
                positions = np.where(obs['obs']['glyphs'] == STAIRS_DOWN)
                print(f"   ✓ Found down stairs at positions: {list(zip(positions[1], positions[0]))}")
            
            if PLAYER in unique_glyphs:
                positions = np.where(obs['obs']['glyphs'] == PLAYER)
                print(f"   ✓ Found player at positions: {list(zip(positions[1], positions[0]))}")
    
    # Move around and search to find stairs
    print("\n5. Searching for stairs...")
    actions = ["search"] * 5 + ["south"] * 3 + ["east"] * 3 + ["search"] * 5
    
    for i, action in enumerate(actions):
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, _, done, _ = step_result
        else:
            obs, _, done, _, _ = step_result
        
        # Check for stairs in text
        long_text = obs.get("text", {}).get("long_term_context", "")
        
        # Extract language observation section
        lang_obs = ""
        if "language observation:" in long_text:
            lang_start = long_text.index("language observation:")
            lang_section = long_text[lang_start:].split("\n\n")[0]
            lang_obs = lang_section
            
        if "stairs" in lang_obs.lower() or "stairs" in long_text.lower():
            print(f"\n   Step {i}: Found stairs mention!")
            print(f"   Language obs: {lang_obs}")
            
            # Check glyphs
            if "obs" in obs and "glyphs" in obs['obs']:
                import numpy as np
                if 2378 in obs['obs']['glyphs']:
                    positions = np.where(obs['obs']['glyphs'] == 2378)
                    print(f"   Glyph positions: {list(zip(positions[1], positions[0]))}")
            
            # Check long_term_context for map
            long_text = obs.get("text", {}).get("long_term_context", "")
            if "map:" in long_text:
                map_start = long_text.index("map:")
                map_section = long_text[map_start:map_start+500]
                print(f"\n   Map section:")
                print(map_section)
            
            break
        
        if done:
            break

if __name__ == "__main__":
    check_observation()