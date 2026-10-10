#!/usr/bin/env python3
"""
Test script to verify position tracking in real_llm_mcts agent.
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.real_llm_mcts import RealLLMMCTSAgent
from balrog.agents.dummy import DummyAgent


def test_position_parsing():
    """Test that the agent correctly parses position from observations"""
    
    # Create a dummy agent
    def client_factory():
        return DummyClient()
    
    from balrog.prompt_builder.history import PromptBuilder
    prompt_builder = PromptBuilder()
    
    agent = RealLLMMCTSAgent(client_factory, prompt_builder)
    
    # Test observation with map data
    test_obs = {
        'text': {
            'long_term_context': '''map:
        #############################
        #.......#.........#.........#
        #...@...#.........#.........#  
        #.......#.........#.........#
        #############################
        
Agent St:18/04 Dx:13 Co:18 In:10 Wi:5 Ch:6 Lawful  
Dlvl:1 $:49 HP:16(16) Pw:2(2) AC:6  Xp:1/5 T:13 
'''
        }
    }
    
    print("Testing position parsing from map...")
    state = agent._parse_game_state(test_obs)
    
    print(f"Parsed position: ({state['x']}, {state['y']})")
    print(f"Expected position: (4, 2) where @ is located")
    
    # Check if position was parsed correctly
    if state['x'] == 4 and state['y'] == 2:
        print("✅ Position parsing works correctly!")
    else:
        print("❌ Position parsing failed!")
        
    # Test another position
    test_obs2 = {
        'text': {
            'long_term_context': '''map:
        #############################
        #.......#.........#.........#
        #.......#.........#....@....#  
        #.......#.........#.........#
        #############################
        
Agent St:18/04 Dx:13 Co:18 In:10 Wi:5 Ch:6 Lawful  
Dlvl:1 $:49 HP:16(16) Pw:2(2) AC:6  Xp:1/5 T:13 
'''
        }
    }
    
    print("\nTesting different position...")
    state2 = agent._parse_game_state(test_obs2)
    print(f"Parsed position: ({state2['x']}, {state2['y']})")
    print(f"Expected position: (20, 2) where @ is located")
    
    if state2['x'] == 20 and state2['y'] == 2:
        print("✅ Position parsing works for different locations!")
    else:
        print("❌ Position parsing failed for different locations!")
        
    # Test movement prediction
    print("\nTesting movement prediction...")
    new_state = agent._predict_state_after_action(state2, 'north')
    print(f"After 'north' from (20, 2): ({new_state['x']}, {new_state['y']})")
    print(f"Expected: (20, 1)")
    
    if new_state['x'] == 20 and new_state['y'] == 1:
        print("✅ Movement prediction works!")
    else:
        print("❌ Movement prediction failed!")
        
    print(f"Original position: ({state2['x']}, {state2['y']})")
    print(f"Predicted after north: ({new_state['x']}, {new_state['y']})")


if __name__ == "__main__":
    test_position_parsing()