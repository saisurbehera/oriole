#!/usr/bin/env python3

"""
Focused test of Simple MCTS Agent to reach level 2
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

# Import base agent directly
from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse, create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env

# Now import our simple MCTS implementation
from balrog.agents.simple_mcts_agent import SimpleMCTSAgent, MCTSNode

def test_focused_mcts():
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
                "max_episode_steps": 300,
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
    
    # Adjust MCTS parameters for better performance
    agent.iterations_per_move = 30  # More iterations
    agent.simulation_depth = 10     # Deeper simulations
    agent.exploration_constant = 2.0  # More exploration
    
    print("=== Focused MCTS Test - Reach Level 2 ===")
    print(f"MCTS iterations: {agent.iterations_per_move}")
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
    max_steps = 200
    current_depth = 1
    stairs_seen = False
    stairs_count = 0
    
    for step in range(max_steps):
        try:
            # Get action from agent
            response = agent.act(obs)
            action = response.completion
            
            # Check environment
            text = obs.get("text", {}).get("long_term_context", "")
            lang_obs = obs.get("text", {}).get("language_observation", "")
            
            # Check for stairs
            if "stairs" in lang_obs.lower():
                if not stairs_seen:
                    print(f"\n🔍 First stairs sighted at step {step}!")
                    stairs_seen = True
                stairs_count += 1
            
            # Check depth
            if "Dlvl:2" in text:
                print(f"\n🎉 SUCCESS! Reached level 2 at step {steps}!")
                current_depth = 2
                break
            
            # Print progress for important actions
            if (step < 10 or 
                action == ">" or 
                "stairs" in lang_obs.lower() or 
                step % 30 == 0):
                print(f"\nStep {step + 1}: {action}")
                if stairs_seen:
                    print(f"  [Stairs visible! Count: {stairs_count}]")
            
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
                # Check death reason
                if "language_observation" in obs.get("text", {}):
                    death_msg = obs["text"]["language_observation"]
                    if "died" in death_msg or "killed" in death_msg:
                        print(f"Death: {death_msg}")
                break
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"\n=== Final Results ===")
    print(f"Steps taken: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final depth: Level {current_depth}")
    print(f"Stairs seen: {stairs_seen} (count: {stairs_count})")
    
    if current_depth >= 2:
        print("\n✅ MCTS successfully reached level 2!")
        print("The MCTS planning approach worked!")
    else:
        print("\n❌ MCTS did not reach level 2")
        if stairs_seen:
            print("But it did find stairs - needs better execution")
        else:
            print("Failed to find stairs - needs better exploration")

if __name__ == "__main__":
    test_focused_mcts()