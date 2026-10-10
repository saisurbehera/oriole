#!/usr/bin/env python3

"""
Properly analyze NetHack observation structure
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.environments.nle.nle_env import make_nle_env
import numpy as np

def analyze_observation():
    # Create environment
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
    
    env = make_nle_env("nle", "NetHackChallenge-v0", MockFullConfig())
    
    # Reset
    obs, _ = env.reset()
    
    print("=== NetHack Observation Analysis ===\n")
    
    # 1. Extract language observation correctly
    long_text = obs["text"]["long_term_context"]
    if "language observation:" in long_text:
        lang_start = long_text.index("language observation:")
        lang_end = long_text.find("\n\n", lang_start)
        lang_obs = long_text[lang_start:lang_end].replace("language observation:\n", "")
        print("1. Language observation:")
        print(f"   {lang_obs[:200]}...")
    
    # 2. Extract map correctly
    if "map:" in long_text:
        map_start = long_text.index("map:")
        map_text = long_text[map_start+4:]  # Skip "map:"
        # Map ends at next section or end of string
        map_end = map_text.find("\n\n")
        if map_end > 0:
            map_text = map_text[:map_end]
        
        print("\n2. Map text representation:")
        map_lines = map_text.strip().split('\n')[:10]  # First 10 lines
        for i, line in enumerate(map_lines):
            print(f"   Line {i:2d}: {repr(line[:40])}")
            
    # 3. Analyze glyphs
    glyphs = obs["obs"]["glyphs"]
    print(f"\n3. Glyph analysis:")
    print(f"   Shape: {glyphs.shape}")
    
    # Common NetHack glyphs
    GLYPH_MAP = {
        2359: '.',  # floor
        2360: '#',  # wall  
        2378: '>',  # stairs down
        2377: '<',  # stairs up
        43: '+',    # closed door
        124: '|',   # vertical wall
        45: '-',    # horizontal wall
        64: '@',    # player
    }
    
    # Count occurrences
    unique, counts = np.unique(glyphs, return_counts=True)
    print(f"\n   Glyph counts:")
    for glyph, count in zip(unique, counts):
        char = GLYPH_MAP.get(glyph, f'?({glyph})')
        print(f"      {glyph:4d} ({char}): {count:3d} times")
    
    # 4. Find player position
    # The @ character might be represented differently
    # Check the map text for @ position
    print(f"\n4. Looking for player...")
    player_found = False
    for y, line in enumerate(map_lines):
        if '@' in line:
            x = line.index('@')
            print(f"   Found @ in map text at position ({x}, {y})")
            player_found = True
            
    # Also check glyphs
    player_positions = np.where(glyphs == 64)
    if len(player_positions[0]) > 0:
        player_y, player_x = player_positions[0][0], player_positions[1][0]
        print(f"   Found player glyph 64 at ({player_x}, {player_y})")
        
        # Show area around player
        print("   Area around player (5x5):")
        for dy in range(-2, 3):
            line = "   "
            for dx in range(-2, 3):
                y, x = player_y + dy, player_x + dx
                if 0 <= y < 21 and 0 <= x < 79:
                    glyph = glyphs[y, x]
                    char = GLYPH_MAP.get(glyph, '?')
                    line += f" {char}"
                else:
                    line += "  "
            print(line)
    
    # 5. Check for stairs in initial observation
    print("\n5. Checking for stairs...")
    stairs_positions = np.where(glyphs == 2378)
    if len(stairs_positions[0]) > 0:
        print(f"   ✓ Found {len(stairs_positions[0])} stairs positions:")
        # Show first few
        for i in range(min(5, len(stairs_positions[0]))):
            y, x = stairs_positions[0][i], stairs_positions[1][i]
            print(f"     Position {i+1}: ({x}, {y})")
            
    # Search for more stairs
    print("\n6. Searching for stairs...")
    stairs_found = False
    
    for step in range(50):
        # Random exploration
        actions = ["south", "east", "north", "west"]
        if step < 10:
            action = "search"
        else:
            action = actions[step % 4]
        
        obs, _, done, _ = env.step(action)
        
        # Check for stairs
        glyphs = obs["obs"]["glyphs"]
        stairs_down = np.where(glyphs == 2378)
        
        if len(stairs_down[0]) > 0 and not stairs_found:
            stairs_found = True
            stairs_y, stairs_x = stairs_down[0][0], stairs_down[1][0]
            print(f"\n   ✓ Found stairs down at step {step}!")
            print(f"     Stairs position: ({stairs_x}, {stairs_y})")
            
            # Get player position
            player_pos = np.where(glyphs == 64)
            if len(player_pos[0]) > 0:
                py, px = player_pos[0][0], player_pos[1][0]
                print(f"     Player position: ({px}, {py})")
                distance = abs(px - stairs_x) + abs(py - stairs_y)
                print(f"     Distance: {distance}")
            
            # Show language observation
            long_text = obs["text"]["long_term_context"]
            if "language observation:" in long_text:
                lang_start = long_text.index("language observation:")
                lang_end = long_text.find("\n\n", lang_start)
                lang_obs = long_text[lang_start:lang_end]
                print(f"\n     Language observation:")
                for line in lang_obs.split('\n')[1:]:  # Skip header
                    if line.strip():
                        print(f"       {line}")
            
            break
            
        if done:
            break
    
    if not stairs_found:
        print("   ✗ No stairs found in 50 steps")

if __name__ == "__main__":
    analyze_observation()