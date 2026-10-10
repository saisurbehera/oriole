#!/usr/bin/env python3

"""
Test a single episode with detailed logging
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.planning_first_agent import PlanningFirstAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env

def test_single_episode():
    # Create config
    class MockConfig:
        def __init__(self):
            self.client_name = "openai"
            self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
            self.base_url = "http://0.0.0.0:8080/v1"
            self.generate_kwargs = {"temperature": 0.2, "max_tokens": 100}
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
                "max_episode_steps": 1000,
                "no_progress_timeout": 200,
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
    
    # Disable knowledge server for testing
    import os
    os.environ['DISABLE_KNOWLEDGE_SERVER'] = '1'
    
    agent = PlanningFirstAgent(client_factory, prompt_builder)
    
    print("=== Single Episode Test ===")
    
    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    agent.reset()
    
    stairs_found = False
    level_2_reached = False
    step_count = 0
    
    for step in range(100):
        # Get action
        response = agent.act(obs)
        action = response.completion
        
        # Log key events
        text = obs.get("text", {})
        lang_obs = text.get("language_observation", "")
        long_text = text.get("long_term_context", "")
        
        if "stairs" in lang_obs.lower() and not stairs_found:
            stairs_found = True
            print(f"\n🔍 Step {step}: Found stairs!")
            print(f"   Observation: {lang_obs}")
            print(f"   Action: {action}")
            
        if action == ">":
            print(f"\n⬇️  Step {step}: Trying to go down stairs")
            print(f"   Observation: {lang_obs}")
        
        # Take action
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, done, truncated, info = step_result
            done = done or truncated
        step_count += 1
        
        # Check if we reached level 2
        new_long_text = obs.get("text", {}).get("long_term_context", "")
        if "Dlvl:2" in new_long_text and not level_2_reached:
            level_2_reached = True
            print(f"\n🎉 Step {step}: REACHED LEVEL 2!")
            break
            
        if done:
            print(f"\nEpisode ended at step {step}")
            break
    
    # Final results
    print(f"\n=== Results ===")
    print(f"Steps taken: {step_count}")
    print(f"Stairs found: {'Yes' if stairs_found else 'No'}")
    print(f"Level 2 reached: {'Yes' if level_2_reached else 'No'}")
    
    # Parse final depth
    final_text = obs.get("text", {}).get("long_term_context", "")
    if "Dlvl:" in final_text:
        depth = final_text.split("Dlvl:")[1].split()[0]
        print(f"Final depth: {depth}")

if __name__ == "__main__":
    test_single_episode()