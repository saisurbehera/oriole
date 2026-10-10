#!/usr/bin/env python3

"""
Debug why '>' action isn't working when stairs are visible
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.feedback_mcts_agent import FeedbackMCTSAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env

def debug_stairs():
    # Create config
    class MockConfig:
        def __init__(self):
            self.client_name = "openai"
            self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
            self.base_url = "http://0.0.0.0:8080/v1"
            self.generate_kwargs = {"temperature": 0.2, "max_tokens": 50}
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
    
    agent = FeedbackMCTSAgent(client_factory, prompt_builder)
    
    print("=== Debug Stairs Action ===")
    
    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}
    
    agent.reset()
    
    stairs_found = False
    stairs_attempts = 0
    
    for step in range(50):
        # Get text info
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        lang_obs = text.get("language_observation", "")
        
        # Check for stairs
        if "stairs" in lang_obs.lower() or not stairs_found:
            # Get action
            response = agent.act(obs)
            action = response.completion
            
            # Check if we found stairs
            if "stairs" in lang_obs.lower():
                if not stairs_found:
                    print(f"\n🔍 Stairs found at step {step}!")
                    print(f"Language obs: {lang_obs}")
                    stairs_found = True
                
                # If trying to use stairs
                if action == ">":
                    stairs_attempts += 1
                    print(f"\nAttempt {stairs_attempts} to use stairs:")
                    print(f"  Action: {action}")
                    
                    # Show map around player
                    map_section = long_text.split("map:")[-1].split("Agent")[0] if "map:" in long_text else ""
                    map_lines = map_section.strip().split('\n')[:10]
                    
                    print("  Map (first 10 lines):")
                    for i, line in enumerate(map_lines):
                        if '@' in line:
                            print(f"    {i}: {line[:40]}  <- PLAYER")
                        elif '>' in line or '<' in line:
                            print(f"    {i}: {line[:40]}  <- STAIRS")
                        else:
                            print(f"    {i}: {line[:40]}")
            
            # Take action
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
            
            # Check what happened after '>'
            if stairs_found and action == ">":
                new_lang_obs = obs.get("text", {}).get("language_observation", "")
                new_long_text = obs.get("text", {}).get("long_term_context", "")
                
                print(f"  Result:")
                print(f"    Reward: {reward}")
                print(f"    New lang obs: {new_lang_obs}")
                
                # Check if we're still on level 1
                if "Dlvl:1" in new_long_text:
                    print("    ❌ Still on level 1")
                elif "Dlvl:2" in new_long_text:
                    print("    ✅ SUCCESS! Reached level 2!")
                    break
                    
                # Check for messages
                message = obs.get("text", {}).get("message", "")
                if message:
                    print(f"    Message: {message}")
            
            if done:
                break
                
            # Stop after finding stairs and trying a few times
            if stairs_attempts >= 5:
                print("\nStopping after 5 attempts")
                break

if __name__ == "__main__":
    debug_stairs()