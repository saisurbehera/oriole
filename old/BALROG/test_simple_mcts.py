#!/usr/bin/env python3

"""
Test the Simple MCTS Agent
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

# Import directly to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location("simple_mcts_agent", "/home/sai/Desktop/oriole/BALROG/balrog/agents/simple_mcts_agent.py")
simple_mcts_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simple_mcts_module)
SimpleMCTSAgent = simple_mcts_module.SimpleMCTSAgent

from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env

def test_simple_mcts():
    # Create config
    class MockConfig:
        def __init__(self):
            self.client_name = "openai"
            self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
            self.base_url = "http://0.0.0.0:8080/v1"
            self.generate_kwargs = {"temperature": 0.1, "max_tokens": 150}
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
                "max_episode_steps": 200,
                "no_progress_timeout": 100,
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
    
    agent = SimpleMCTSAgent(client_factory, prompt_builder)
    
    print("=== Testing Simple MCTS Agent ===")
    print(f"MCTS iterations per move: {agent.iterations_per_move}")
    print(f"Simulation depth: {agent.simulation_depth}")
    print(f"Exploration constant: {agent.exploration_constant}")
    
    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}
    
    agent.reset()
    
    total_reward = 0
    steps = 0
    max_steps = 100
    current_depth = 1
    
    for step in range(max_steps):
        try:
            # Get action from agent
            response = agent.act(obs)
            action = response.completion
            reasoning = response.reasoning
            
            # Check depth
            text = obs.get("text", {}).get("long_term_context", "")
            if "Dlvl:2" in text:
                print(f"\n🎉 SUCCESS! Reached dungeon level 2 at step {steps}!")
                current_depth = 2
                break
            
            # Print progress
            if step < 10 or action == ">" or step % 20 == 0:
                print(f"\nStep {step + 1}: {action} ({reasoning})")
                
                # Check environment
                lang_obs = obs.get("text", {}).get("language_observation", "")
                if "stairs" in lang_obs:
                    print(f"  [STAIRS VISIBLE!]")
            
            # Take action
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
                
            total_reward += reward
            steps += 1
            
            if done:
                print(f"\nEpisode ended at step {steps}")
                break
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
    
    print(f"\n=== Results ===")
    print(f"Steps: {steps}")
    print(f"Reward: {total_reward}")
    print(f"Final depth: {current_depth}")
    
    if current_depth >= 2:
        print("\n✅ Simple MCTS successfully reached level 2!")
    else:
        print("\n❌ Simple MCTS did not reach level 2")

if __name__ == "__main__":
    test_simple_mcts()