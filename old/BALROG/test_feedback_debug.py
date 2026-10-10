#!/usr/bin/env python3

"""
Debug test for feedback MCTS agent
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.feedback_mcts_agent import FeedbackMCTSAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env

def test_feedback_debug():
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
    
    agent = FeedbackMCTSAgent(client_factory, prompt_builder)
    
    print("=== Feedback MCTS Debug Test ===")
    
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
    
    for step in range(max_steps):
        try:
            # Get action from agent
            response = agent.act(obs)
            action = response.completion
            reasoning = response.reasoning
            
            # Print every 10 steps or important actions
            if step < 5 or step % 20 == 0 or action in [">", "open", "kick"]:
                print(f"\nStep {step + 1}:")
                print(f"  Action: {action}")
                print(f"  Reasoning: {reasoning}")
                
                # Show learned values
                if agent.action_values:
                    print("  Learned values:")
                    for ctx, actions in list(agent.action_values.items())[:2]:
                        print(f"    Context: {ctx}")
                        for act, stats in list(actions.items())[:3]:
                            if stats.visit_count > 0:
                                print(f"      {act}: avg={stats.average_reward:.2f}, visits={stats.visit_count}")
            
            # Take action
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
                
            total_reward += reward
            steps += 1
            
            # Check for level 2
            text = obs.get("text", {}).get("long_term_context", "")
            if "Dlvl:2" in text:
                print(f"\n🎉 SUCCESS! Reached level 2 at step {steps}!")
                break
            
            if done:
                print(f"\nEpisode ended at step {steps}")
                break
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"\n=== Final Results ===")
    print(f"Steps: {steps}")
    print(f"Total reward: {total_reward}")
    print(f"Contexts learned: {len(agent.action_values)}")
    
    # Show all learned contexts
    print("\nAll learned contexts:")
    for ctx in agent.action_values.keys():
        print(f"  - {ctx}")

if __name__ == "__main__":
    test_feedback_debug()