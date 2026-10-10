#!/usr/bin/env python3

"""
Debug stairs position detection
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.planning_first_agent import PlanningFirstAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env
import json

def debug_stairs():
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
    
    agent = PlanningFirstAgent(client_factory, prompt_builder)
    
    print("=== Debugging Stairs Position ===")
    
    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}
    
    agent.reset()
    
    # Parse initial state
    game_state = agent._parse_game_state(obs)
    
    print(f"\nInitial game state:")
    print(f"Player position: {game_state['player_pos']}")
    print(f"Stairs info: {game_state['stairs_info']}")
    print(f"Depth: {game_state['depth']}")
    
    # Show map
    text = obs.get("text", {}).get("long_term_context", "")
    map_lines = text.split("map:")[-1].split("Agent")[0].strip().split('\n')
    
    print(f"\nMap (first 10 lines):")
    for i, line in enumerate(map_lines[:10]):
        print(f"{i:2}: {repr(line[:40])}")
        # Check all characters for potential stairs
        for j, char in enumerate(line):
            if char == '<':
                print(f"    ^ STAIRS UP '<' at column {j}")
            elif char == '>':
                print(f"    ^ STAIRS DOWN '>' at column {j}")
            elif char == '@':
                print(f"    ^ PLAYER '@' at column {j}")
            elif char == '\\':
                print(f"    ^ POSSIBLE STAIRS '\\' at column {j}")
            elif char in '.':
                pass  # Floor
            elif char not in ' -|':
                print(f"    ^ UNKNOWN '{char}' at column {j}")
    
    # Check language observation
    lang_obs = obs.get("text", {}).get("language_observation", "")
    print(f"\nLanguage observation: {lang_obs}")
    
    # Try to move to stairs and use them
    for step in range(10):
        response = agent.act(obs)
        action = response.completion
        
        print(f"\nStep {step + 1}: Action: {action}")
        
        # Take action
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        elif len(step_result) == 5:
            obs, reward, done, truncated, info = step_result
            done = done or truncated
        
        # Check depth
        text = obs.get("text", {}).get("long_term_context", "")
        if "Dlvl:2" in text:
            print(f"\n🎉 SUCCESS! Reached level 2!")
            break
        
        # Parse new state
        game_state = agent._parse_game_state(obs)
        print(f"Player: {game_state['player_pos']}, Stairs: {game_state['stairs_info']}")
        
        if done:
            break

if __name__ == "__main__":
    debug_stairs()