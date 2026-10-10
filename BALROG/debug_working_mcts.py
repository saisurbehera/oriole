#!/usr/bin/env python3

"""
Debug the working MCTS agent
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.feedback_mcts_working import FeedbackMCTSWorking
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env

def debug_mcts():
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
    
    agent = FeedbackMCTSWorking(client_factory, prompt_builder)
    
    print("=== Debug Working MCTS ===")
    
    # Reset environment
    obs, info = env.reset()
    agent.reset()
    
    stairs_found = False
    on_stairs = False
    
    for step in range(100):
        # Get action
        response = agent.act(obs)
        action = response.completion
        reasoning = response.reasoning
        
        # Parse state to show what agent sees
        state = agent._parse_game_state(obs)
        
        # Log key events
        if state["stairs_visible"] and not stairs_found:
            stairs_found = True
            print(f"\n🔍 Step {step}: Found stairs!")
            print(f"   Player pos: {state['player_pos']}")
            print(f"   Stairs pos: {state['stairs_pos']}")
            print(f"   Action: {action}")
            print(f"   Reasoning: {reasoning}")
        
        if state["on_stairs"] and not on_stairs:
            on_stairs = True
            print(f"\n✅ Step {step}: ON STAIRS!")
            print(f"   Position: {state['player_pos']}")
            print(f"   Action: {action}")
            print(f"   Reasoning: {reasoning}")
        
        if action == ">" and state["player_pos"]:
            print(f"\n⬇️  Step {step}: Attempting stairs")
            print(f"   Player pos: {state['player_pos']}")
            print(f"   Stairs pos: {state['stairs_pos']}")
            print(f"   On stairs: {state['on_stairs']}")
        
        # Take action
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
        # Check if we reached level 2
        new_state = agent._parse_game_state(obs)
        if new_state["depth"] == 2:
            print(f"\n🎉 SUCCESS! Reached level 2 at step {step}!")
            print(f"   Final position: {new_state['player_pos']}")
            break
        
        if done:
            print(f"\nEpisode ended at step {step}")
            print(f"   Final depth: {new_state['depth']}")
            break
    
    # Show final learned values
    print("\n=== Learned Values ===")
    for context, actions in agent.action_values.items():
        if any(stats.visit_count > 0 for stats in actions.values()):
            print(f"\nContext: {context}")
            sorted_actions = sorted(
                [(a, s) for a, s in actions.items() if s.visit_count > 0],
                key=lambda x: x[1].average_reward,
                reverse=True
            )
            for action, stats in sorted_actions[:3]:
                print(f"  {action}: avg={stats.average_reward:.2f}, visits={stats.visit_count}")

if __name__ == "__main__":
    debug_mcts()