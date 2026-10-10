#!/usr/bin/env python3

"""
Test the MCTS Planning Agent
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.mcts_planning_agent import MCTSPlanningAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env
import json

def test_mcts():
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
    
    agent = MCTSPlanningAgent(client_factory, prompt_builder)
    
    print("=== Testing MCTS Planning Agent ===")
    print(f"MCTS iterations per turn: {agent.mcts_iterations}")
    print(f"Simulation depth: {agent.simulation_depth}")
    print(f"Exploration weight: {agent.exploration_weight}")
    
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
    max_steps = 50  # Run for 50 steps
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
            
            # Print important steps
            if step < 5 or "stairs" in reasoning.lower() or action == ">" or step % 10 == 0:
                print(f"\nStep {step + 1}: Action: {action}")
                print(f"  Reasoning: {reasoning}")
                
                # Check if we see stairs
                lang_obs = obs.get("text", {}).get("language_observation", "")
                if "stairs" in lang_obs:
                    print(f"  [STAIRS VISIBLE: {lang_obs[:50]}...]")
            
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
            import traceback
            traceback.print_exc()
            break
    
    print(f"\n=== Final Results ===")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward}")
    print(f"Final depth: Dlvl:{current_depth}")
    
    if current_depth >= 2:
        print("\n✅ MCTS agent successfully reached level 2!")
    else:
        print("\n❌ MCTS agent did not reach level 2")

if __name__ == "__main__":
    test_mcts()