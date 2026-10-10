#!/usr/bin/env python3

"""
Simple debug for stairs - no LLM, just manual exploration
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from nle import nethack
import gym

def debug_stairs():
    # Create basic NLE environment
    env = gym.make("NetHackChallenge-v0")
    
    print("=== Debug Stairs (Manual) ===")
    
    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result
    
    # Manual exploration commands to find stairs
    # NetHack action mapping
    actions = {
        'north': 1,      # k
        'south': 2,      # j  
        'east': 3,       # l
        'west': 4,       # h
        'northeast': 5,  # u
        'southeast': 6,  # n
        'southwest': 7,  # b
        'northwest': 8,  # y
        'wait': 16,      # .
        'search': 18,    # s
        'open': 19,      # o
        'kick': 21,      # ^D (kick)
        '>': 23,         # > (down stairs)
        '<': 22,         # < (up stairs)
    }
    
    def show_map(obs):
        # Get the map
        glyphs = obs['glyphs']
        
        # Find player position
        player_y, player_x = None, None
        for y in range(21):
            for x in range(79):
                if glyphs[y, x] == 64:  # @ symbol
                    player_y, player_x = y, x
                    
        if player_y is None:
            return
            
        # Show 5x5 area around player
        print("\nMap around player:")
        for dy in range(-5, 6):
            line = ""
            for dx in range(-5, 6):
                y, x = player_y + dy, player_x + dx
                if 0 <= y < 21 and 0 <= x < 79:
                    glyph = glyphs[y, x]
                    # Convert glyph to char
                    if glyph == 64:
                        line += "@"
                    elif glyph == 2378:  # stairs down >
                        line += ">"
                    elif glyph == 2377:  # stairs up <
                        line += "<"
                    elif glyph == 2359:  # floor .
                        line += "."
                    elif glyph == 2360:  # wall
                        line += "#"
                    elif glyph == 43:    # door |
                        line += "|"
                    elif glyph == 45:    # door -
                        line += "-"
                    else:
                        line += " "
                else:
                    line += " "
            print(line)
    
    # Exploration sequence
    sequence = ['search'] * 10 + ['south'] * 5 + ['east'] * 5 + ['search'] * 10
    
    stairs_found = False
    for i, action_name in enumerate(sequence):
        if action_name not in actions:
            continue
            
        action = actions[action_name]
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            truncated = False
        else:
            obs, reward, done, truncated, info = step_result
        
        # Check for stairs
        glyphs = obs['glyphs']
        for y in range(21):
            for x in range(79):
                if glyphs[y, x] == 2378:  # stairs down
                    if not stairs_found:
                        print(f"\n🔍 Stairs found at step {i}!")
                        stairs_found = True
                        show_map(obs)
                        
                        # Find player position
                        player_y, player_x = None, None
                        for py in range(21):
                            for px in range(79):
                                if glyphs[py, px] == 64:
                                    player_y, player_x = py, px
                                    
                        if player_y and player_x:
                            stairs_y, stairs_x = y, x
                            print(f"Player at ({player_x}, {player_y})")
                            print(f"Stairs at ({stairs_x}, {stairs_y})")
                            
                            # Try to move to stairs
                            if player_x < stairs_x:
                                print("Moving east to stairs...")
                                result = env.step(actions['east'])
                                obs = result[0]
                            elif player_x > stairs_x:
                                print("Moving west to stairs...")
                                result = env.step(actions['west'])
                                obs = result[0]
                            elif player_y < stairs_y:
                                print("Moving south to stairs...")
                                result = env.step(actions['south'])
                                obs = result[0]
                            elif player_y > stairs_y:
                                print("Moving north to stairs...")
                                result = env.step(actions['north'])
                                obs = result[0]
                                
                            # Now try to use stairs
                            print("\nTrying to use stairs with '>' action...")
                            result = env.step(actions['>'])
                            if len(result) == 4:
                                obs, reward, done, info = result
                            else:
                                obs, reward, done, truncated, info = result
                            
                            # Check if we moved down
                            message = obs.get('message', b'').decode('ascii', errors='ignore')
                            print(f"Message: {message}")
                            print(f"Reward: {reward}")
                            
                            # Check depth
                            if 'blstats' in obs:
                                depth = obs['blstats'][12]  # depth stat
                                print(f"Current depth: {depth}")
                                if depth == 2:
                                    print("✅ SUCCESS! Reached level 2!")
                                else:
                                    print("❌ Still on level 1")
                                    
                            return
        
        if done or truncated:
            break
            
    print("\nDid not find stairs in exploration sequence")

if __name__ == "__main__":
    debug_stairs()