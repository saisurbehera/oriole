#!/usr/bin/env python3

"""
Test specifically for reaching level 2
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.planning_first_agent import PlanningFirstAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env
import json

def test_stairs():
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
                "max_episode_steps": 1000,  # More steps
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
    
    agent = PlanningFirstAgent(client_factory, prompt_builder)
    
    print("=== Testing Stairs Usage - Goal: Reach Level 2 ===")
    
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
    max_steps = 100  # Enough to find and use stairs
    current_depth = 1
    
    for step in range(max_steps):
        # Get action from agent
        try:
            response = agent.act(obs)
            action = response.completion
            reasoning = response.reasoning
            
            # Check depth
            text = obs.get("text", {}).get("long_term_context", "")
            if "Dlvl:2" in text:
                print(f"\n🎉 SUCCESS! Reached dungeon level 2 at step {steps}!")
                current_depth = 2
                break
            
            # Only print important steps
            if step < 10 or "stairs" in reasoning.lower() or action == ">" or step % 20 == 0:
                print(f"\nStep {step + 1}: Action: {action} | Reasoning: {reasoning}")
                
                # Check if we see stairs
                lang_obs = obs.get("text", {}).get("language_observation", "")
                if "stairs" in lang_obs:
                    print(f"  [STAIRS VISIBLE: {lang_obs}]")
                
                # Show what's in the map if on early steps
                if step < 5:
                    text = obs.get("text", {}).get("long_term_context", "")
                    map_section = text.split("map:")[-1].split("Agent")[0]
                    # Check for stairs in map
                    if '<' in map_section or '>' in map_section:
                        print("  [STAIRS IN MAP!]")
            
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
            
            if done:
                print(f"\nEpisode finished at step {steps}")
                break
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
    
    print(f"\n=== Final Results ===")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward}")
    print(f"Final depth: Dlvl:{current_depth}")
    
    if current_depth >= 2:
        print("\n✅ SUCCESSFULLY REACHED LEVEL 2!")
    else:
        print("\n❌ Failed to reach level 2")
        
        # Analyze why
        if hasattr(agent, 'current_plan') and agent.current_plan:
            print(f"Last plan goal: {agent.current_plan.goal}")
            print(f"Last plan steps: {agent.current_plan.steps}")

if __name__ == "__main__":
    test_stairs()