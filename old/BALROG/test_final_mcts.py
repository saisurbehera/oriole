#!/usr/bin/env python3

"""
Final test of MCTS with detailed debugging
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.feedback_mcts_working import FeedbackMCTSWorking
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env

def test_mcts():
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
                "max_episode_steps": 500,
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
    
    agent = FeedbackMCTSWorking(client_factory, prompt_builder)
    
    print("=== Final MCTS Test ===")
    
    # Run multiple episodes
    successes = 0
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        
        # Reset
        obs, info = env.reset()
        agent.reset()
        
        found_stairs = False
        reached_level_2 = False
        
        for step in range(100):
            # Get action
            response = agent.act(obs)
            action = response.completion
            
            # Parse state
            state = agent._parse_game_state(obs)
            
            # Show when we find stairs
            if state["stairs_pos"] and not found_stairs:
                found_stairs = True
                print(f"  ✓ Found stairs at step {step}: player={state['player_pos']}, stairs={state['stairs_pos']}")
            
            # Show when on stairs
            if state["on_stairs"] and action == ">":
                print(f"  ⬇️ On stairs at step {step}, going down!")
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # Check depth
            new_state = agent._parse_game_state(obs)
            if new_state["depth"] == 2:
                reached_level_2 = True
                successes += 1
                print(f"  🎉 SUCCESS! Reached level 2 at step {step}!")
                break
            
            if done:
                print(f"  ✗ Episode ended at step {step}, depth={new_state['depth']}")
                break
    
    print(f"\n=== Summary ===")
    print(f"Success rate: {successes}/3 ({successes/3*100:.0f}%)")
    
    if successes > 0:
        print("\n🎉 MCTS AGENT CAN REACH LEVEL 2!")
    else:
        print("\n❌ Still not working...")
        
        # Show what went wrong
        print("\nDebugging last episode:")
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        
        if "map:" in long_text:
            map_start = long_text.index("map:") + 4
            print("\nMap section after 'map:':")
            print(repr(long_text[map_start:map_start+200]))

if __name__ == "__main__":
    test_mcts()