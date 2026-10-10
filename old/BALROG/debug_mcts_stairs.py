#!/usr/bin/env python3

"""
Debug MCTS agent's stair usage
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.feedback_mcts_fixed import FeedbackMCTSFixed
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env
import pickle

def debug_mcts_stairs():
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
    
    # Create agent with pretrained values
    config = MockConfig()
    agent_config = MockAgentConfig()
    
    client_factory = create_llm_client(config)
    prompt_builder = create_prompt_builder(agent_config)
    
    # Load pretrained values if available
    try:
        with open('mcts_trained_values.pkl', 'rb') as f:
            pretrained_values = pickle.load(f)
        print("Loaded pretrained values")
    except:
        pretrained_values = None
        print("No pretrained values found")
    
    agent = FeedbackMCTSFixed(client_factory, prompt_builder, pretrained_values)
    
    print("=== Debug MCTS Stairs ===")
    
    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    
    agent.reset()
    
    stairs_found = False
    stairs_attempts = 0
    
    for step in range(200):
        # Get current state
        text = obs.get("text", {})
        lang_obs = text.get("language_observation", "")
        long_text = text.get("long_term_context", "")
        
        # Parse map to find positions
        map_section = long_text.split("map:")[-1].split("Agent")[0] if "map:" in long_text else ""
        lines = map_section.strip().split('\n')[:21]
        
        player_pos = None
        stairs_pos = None
        
        for y, line in enumerate(lines):
            if '@' in line:
                player_pos = (line.index('@'), y)
            if '>' in line:
                stairs_pos = (line.index('>'), y)
        
        # Get action from agent
        response = agent.act(obs)
        action = response.completion
        
        # Log when we find stairs
        if stairs_pos and not stairs_found:
            stairs_found = True
            print(f"\n🔍 Step {step}: Found stairs at {stairs_pos}!")
            print(f"   Player at: {player_pos}")
            print(f"   Distance: {abs(player_pos[0]-stairs_pos[0]) + abs(player_pos[1]-stairs_pos[1]) if player_pos else 'unknown'}")
        
        # Log navigation and attempts
        if stairs_found:
            if player_pos and stairs_pos:
                distance = abs(player_pos[0]-stairs_pos[0]) + abs(player_pos[1]-stairs_pos[1])
                if distance == 0:
                    print(f"\n✅ Step {step}: On stairs! Action: {action}")
                elif distance <= 3:
                    print(f"   Step {step}: Near stairs (distance={distance}), action: {action}")
        
        if action == ">":
            stairs_attempts += 1
            print(f"\n⬇️  Step {step}: Attempting stairs (attempt #{stairs_attempts})")
            print(f"   Player position: {player_pos}")
            print(f"   Stairs position: {stairs_pos}")
            if player_pos and stairs_pos:
                if player_pos == stairs_pos:
                    print("   ✓ Correctly positioned on stairs")
                else:
                    print(f"   ✗ NOT on stairs! Distance: {abs(player_pos[0]-stairs_pos[0]) + abs(player_pos[1]-stairs_pos[1])}")
        
        # Take action
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, done, truncated, info = step_result
            done = done or truncated
        
        # Check if we reached level 2
        new_long_text = obs.get("text", {}).get("long_term_context", "")
        if "Dlvl:2" in new_long_text:
            print(f"\n🎉 SUCCESS! Reached level 2 at step {step}!")
            break
        
        if done:
            print(f"\nEpisode ended at step {step}")
            break
    
    # Final summary
    print(f"\n=== Summary ===")
    print(f"Stairs found: {stairs_found}")
    print(f"Stairs attempts: {stairs_attempts}")
    
    # Show learned context for stairs
    if hasattr(agent, 'action_values'):
        print("\n=== Learned Values ===")
        for context, actions in agent.action_values.items():
            if 'stairs' in context:
                print(f"\nContext: {context}")
                for action, stats in actions.items():
                    if stats.visit_count > 0:
                        print(f"  {action}: avg_reward={stats.average_reward:.2f}, visits={stats.visit_count}")

if __name__ == "__main__":
    debug_mcts_stairs()