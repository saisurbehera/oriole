#!/usr/bin/env python3

"""
Quick test of planning agent on single NetHack episode
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.planning_first_agent import PlanningFirstAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env
import json

def quick_test():
    # Create config
    class MockConfig:
        def __init__(self):
            self.client_name = "openai"
            self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
            self.base_url = "http://0.0.0.0:8080/v1"
            self.generate_kwargs = {"temperature": 0.2, "max_tokens": 200}
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
                "max_episode_steps": 50,  # Short test
                "no_progress_timeout": 20,
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
    
    agent = PlanningFirstAgent(client_factory, prompt_builder)
    
    print("=== Quick NetHack Test with Planning Agent ===")
    
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
    max_steps = 20  # Very short test
    
    print(f"Initial observation keys: {list(obs.keys()) if hasattr(obs, 'keys') else type(obs)}")
    
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # Get action from agent
        try:
            response = agent.act(obs)
            action = response.completion
            reasoning = response.reasoning
            
            print(f"Action: {action}")
            print(f"Reasoning: {reasoning}")
            
            # Take action in environment
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                print(f"Unexpected step result length: {len(step_result)}")
                break
            total_reward += reward
            steps += 1
            
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            
            if done:
                print(f"Episode finished at step {steps}")
                break
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
    
    print(f"\n=== Final Results ===")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward}")
    print(f"Final info: {info if 'info' in locals() else 'N/A'}")
    
    # Check if knowledge server was used
    if hasattr(agent, 'current_plan') and agent.current_plan:
        print(f"Final plan goal: {agent.current_plan.goal}")
        print(f"Plan steps completed: {agent.plan_step}/{len(agent.current_plan.steps)}")

if __name__ == "__main__":
    quick_test()