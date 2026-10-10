#!/usr/bin/env python3

"""
Debug stairs issue - show exactly what happens
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from nle import nethack
import gym
import numpy as np

def debug_stairs():
    # Create basic NLE environment
    env = gym.make("NetHackChallenge-v0")
    
    print("=== Debug Stairs V2 ===")
    
    # Reset environment
    obs = env.reset()
    
    # NetHack action mapping
    ACTION_MEANINGS = env.unwrapped.actions
    print(f"Available actions: {ACTION_MEANINGS}")
    
    # Find the index for '>'
    down_stairs_idx = None
    for i, action in enumerate(ACTION_MEANINGS):
        if action == ord('>'):
            down_stairs_idx = i
            print(f"Down stairs action '>' is at index {i}")
            
    if down_stairs_idx is None:
        print("ERROR: Could not find '>' action!")
        return
    
    def show_area_around_player(obs):
        glyphs = obs['glyphs']
        
        # Find player
        player_pos = np.where(glyphs == 64)  # @ = 64
        if len(player_pos[0]) == 0:
            return None, None
            
        py, px = player_pos[0][0], player_pos[1][0]
        
        # Show 11x11 area
        print("\nArea around player:")
        print("    ", end="")
        for dx in range(-5, 6):
            print(f"{dx:2d}", end=" ")
        print()
        
        stairs_positions = []
        
        for dy in range(-5, 6):
            print(f"{dy:3d} ", end="")
            for dx in range(-5, 6):
                y, x = py + dy, px + dx
                if 0 <= y < 21 and 0 <= x < 79:
                    g = glyphs[y, x]
                    if g == 64:  # player
                        print(" @ ", end="")
                    elif g == 2378:  # down stairs
                        print(" > ", end="")
                        stairs_positions.append((y, x))
                    elif g == 2359:  # floor
                        print(" . ", end="")
                    elif g == 2360:  # wall
                        print(" # ", end="")
                    else:
                        print(f"{g:3d}", end="")
                else:
                    print("   ", end="")
            print()
            
        return (py, px), stairs_positions
    
    # Try searching and moving around
    steps = 0
    max_steps = 100
    stairs_found = False
    
    while steps < max_steps:
        # Show current state
        player_pos, stairs_positions = show_area_around_player(obs)
        
        if stairs_positions and not stairs_found:
            stairs_found = True
            print(f"\n🔍 STAIRS FOUND at positions: {stairs_positions}")
            py, px = player_pos
            sy, sx = stairs_positions[0]
            print(f"Player at ({px}, {py}), stairs at ({sx}, {sy})")
            
            # Check if we're on the stairs
            if py == sy and px == sx:
                print("\n✅ Player is ON the stairs! Trying '>' action...")
                
                # Try the down stairs action
                print(f"Executing action {down_stairs_idx} ('>') ...")
                result = env.step(down_stairs_idx)
                obs = result[0]
                reward = result[1]
                
                # Check message
                if 'message' in obs:
                    msg = obs['message']
                    if isinstance(msg, bytes):
                        msg = msg.decode('ascii', errors='ignore')
                    print(f"Message: '{msg}'")
                
                # Check depth
                if 'blstats' in obs:
                    depth = obs['blstats'][12]
                    print(f"Depth after action: {depth}")
                    if depth == 2:
                        print("🎉 SUCCESS! Reached level 2!")
                        return
                    else:
                        print("❌ Still on level 1")
                        
                        # Try search action to see if we need to search first
                        print("\nTrying search action...")
                        search_idx = None
                        for i, action in enumerate(ACTION_MEANINGS):
                            if action == ord('s'):
                                search_idx = i
                                break
                        
                        if search_idx:
                            result = env.step(search_idx)
                            obs = result[0]
                            if 'message' in obs:
                                msg = obs['message']
                                if isinstance(msg, bytes):
                                    msg = msg.decode('ascii', errors='ignore')
                                print(f"Search message: '{msg}'")
                                
                            # Try > again
                            print("\nTrying '>' again after search...")
                            result = env.step(down_stairs_idx)
                            obs = result[0]
                            
                            if 'blstats' in obs:
                                depth = obs['blstats'][12]
                                print(f"Depth after second attempt: {depth}")
                
                return
        
        # Random movement to explore
        if steps % 5 == 0:
            # Search
            for i, action in enumerate(ACTION_MEANINGS):
                if action == ord('s'):
                    result = env.step(i)
                    obs = result[0]
                    break
        else:
            # Move randomly
            move_actions = []
            for i, action in enumerate(ACTION_MEANINGS):
                if action in [ord('h'), ord('j'), ord('k'), ord('l')]:
                    move_actions.append(i)
            
            if move_actions:
                action = np.random.choice(move_actions)
                result = env.step(action)
                obs = result[0]
        
        steps += 1
        
        if result[2]:  # done
            break
    
    print(f"\nExploration ended after {steps} steps")

if __name__ == "__main__":
    debug_stairs()