#!/usr/bin/env python3

"""
Test the working MCTS agent
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.mcts_working import MCTSWorking
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env

def test_working():
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
                "max_episode_steps": 300,
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
    
    agent = MCTSWorking(client_factory, prompt_builder)
    
    print("=== Testing Working MCTS Agent ===")
    
    try:
        # Reset
        obs, info = env.reset()
        agent.reset()
        
        # Run steps
        for step in range(150):
            # Get action
            response = agent.act(obs)
            action = response.completion
            reasoning = response.reasoning
            
            # Parse state to see what agent sees
            state = agent._parse_game_state(obs)
            
            # Print key information
            if step % 10 == 0 or state['down_stairs_visible'] or state['on_down_stairs']:
                print(f"\nStep {step}:")
                print(f"  Action: {action}")
                print(f"  Reasoning: {reasoning}")
                print(f"  Player pos: {state['player_pos']}")
                print(f"  Down stairs pos: {state['down_stairs_pos']}")
                print(f"  On down stairs: {state['on_down_stairs']}")
                print(f"  Down stairs visible: {state['down_stairs_visible']}")
                print(f"  Depth: {state['depth']}")
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            
            # Check if we reached level 2
            new_state = agent._parse_game_state(obs)
            if new_state['depth'] > state['depth']:
                print(f"\n🎉 SUCCESS! Reached level {new_state['depth']} at step {step}!")
                print(f"Final score: {new_state['score']}")
                break
                
            if done or truncated:
                print(f"\nEpisode ended at step {step}, final depth={new_state['depth']}")
                break
                
        print("\n=== Final State ===")
        final_state = agent._parse_game_state(obs)
        print(f"Final depth: {final_state['depth']}")
        print(f"Final score: {final_state['score']}")
        print(f"Final position: {final_state['player_pos']}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_working()