#!/usr/bin/env python3

"""
Debug the final MCTS agent
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.mcts_final import MCTSFinal
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env

def debug():
    # Create config
    class MockConfig:
        def __init__(self):
            self.client_name = "openai"
            self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
            self.base_url = "http://0.0.0.0:8080/v1"
            self.generate_kwargs = {"temperature": 0.1, "max_tokens": 50}
            self.timeout = 30
            self.max_retries = 3
            self.delay = 1
            self.alternate_roles = False
    
    class MockAgentConfig:
        def __init__(self):
            self.max_text_history = 4
            self.max_image_history = 0
            self.max_cot_history = 1
            self.max_icl_history = 0
            self.cache_icl = False
            
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    # Create environment config
    class MockEnvConfig:
        def __init__(self):
            self.nle_kwargs = {
                "character": "@",
                "max_episode_steps": 100,
                "savedir": None,
                "save_ttyrec_every": 0,
                "skip_more": True
            }
    
    class MockFullConfig:
        def __init__(self):
            self.envs = MockEnvConfig()
            self.agent = MockAgentConfig()
    
    # Create environment
    full_config = MockFullConfig()
    env = make_nle_env("nle", "NetHackChallenge-v0", full_config)
    
    # Create agent
    config = MockConfig()
    agent_config = MockAgentConfig()
    
    client_factory = create_llm_client(config)
    prompt_builder = create_prompt_builder(agent_config)
    
    agent = MCTSFinal(client_factory, prompt_builder)
    
    print("=== Debug MCTS Final ===")
    
    try:
        # Reset
        obs, info = env.reset()
        agent.reset()
        
        # Run more steps to see if it reaches stairs
        for step in range(50):
            print(f"\nStep {step}:")
            
            # Get action
            response = agent.act(obs)
            action = response.completion
            reasoning = response.reasoning
            
            # Parse state to see what agent sees
            state = agent._parse_game_state(obs)
            
            # Only print key events
            if step < 5 or state['on_stairs'] or action == "down":
                print(f"  Action: {action}")
                print(f"  Player pos: {state['player_pos']}")
                print(f"  Stairs pos: {state['stairs_pos']}")
                
            if state['on_stairs']:
                print("  🎯 ON STAIRS!")
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            
            # Check if we reached level 2
            new_state = agent._parse_game_state(obs)
            if new_state['depth'] == 2:
                print(f"\n🎉 SUCCESS! Reached level 2 at step {step}!")
                break
                
            if done or truncated:
                print(f"\nEpisode ended at step {step}, depth={new_state['depth']}")
                break
                
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()