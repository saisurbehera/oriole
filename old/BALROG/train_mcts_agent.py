#!/usr/bin/env python3

"""
Training script for MCTS agent
Runs multiple episodes to build up action-value estimates
"""

import sys
import json
import pickle
from datetime import datetime
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.feedback_mcts_agent import FeedbackMCTSAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env

def train_mcts(num_episodes=50, save_path="mcts_trained_values.pkl"):
    # Create config
    class MockConfig:
        def __init__(self):
            self.client_name = "openai"
            self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
            self.base_url = "http://0.0.0.0:8080/v1"
            self.generate_kwargs = {"temperature": 0.3, "max_tokens": 50}
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
    
    agent = FeedbackMCTSAgent(client_factory, prompt_builder)
    
    # Training stats
    episode_rewards = []
    max_depths = []
    contexts_discovered = set()
    
    print("=== MCTS Training ===")
    print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset environment
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        
        agent.reset()  # This preserves learned values!
        
        episode_reward = 0
        max_depth = 1
        steps = 0
        
        # Run episode
        done = False
        while not done and steps < 500:
            # Get action
            response = agent.act(obs)
            action = response.completion
            
            # Take action
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
                
            episode_reward += reward
            steps += 1
            
            # Track depth
            text = obs.get("text", {}).get("long_term_context", "")
            if "Dlvl:2" in text:
                max_depth = max(max_depth, 2)
                print(f"  Episode {episode + 1}: Reached level 2!")
            elif "Dlvl:3" in text:
                max_depth = max(max_depth, 3)
                print(f"  Episode {episode + 1}: Reached level 3!")
        
        # Record episode stats
        episode_rewards.append(episode_reward)
        max_depths.append(max_depth)
        
        # Track contexts
        for context in agent.action_values.keys():
            contexts_discovered.add(context)
        
        # Progress report every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            avg_depth = sum(max_depths[-10:]) / 10
            print(f"\nEpisodes {episode-8}-{episode+1}:")
            print(f"  Avg reward: {avg_reward:.2f}")
            print(f"  Avg max depth: {avg_depth:.2f}")
            print(f"  Unique contexts: {len(contexts_discovered)}")
            print(f"  Total experiences: {sum(sum(a.visit_count for a in actions.values()) for actions in agent.action_values.values())}")
            
            # Show best actions for key contexts
            for context in ["stairs_visible", "door_adjacent|in_room|stairs_visible"]:
                if context in agent.action_values:
                    print(f"\n  Best actions for '{context}':")
                    actions = agent.action_values[context]
                    sorted_actions = sorted(actions.items(), 
                                          key=lambda x: x[1].average_reward, 
                                          reverse=True)[:3]
                    for act, stats in sorted_actions:
                        if stats.visit_count > 0:
                            print(f"    {act}: avg={stats.average_reward:.2f}, visits={stats.visit_count}")
    
    # Save trained values
    print(f"\nSaving trained values to {save_path}...")
    # Convert defaultdict to regular dict for pickling
    values_to_save = {}
    for context, actions in agent.action_values.items():
        values_to_save[context] = dict(actions)
    
    with open(save_path, 'wb') as f:
        pickle.dump(values_to_save, f)
    
    # Save training summary
    summary = {
        "episodes": num_episodes,
        "timestamp": datetime.now().isoformat(),
        "final_stats": {
            "avg_reward": sum(episode_rewards) / len(episode_rewards),
            "max_depth_reached": max(max_depths),
            "unique_contexts": len(contexts_discovered),
            "total_experiences": sum(sum(a.visit_count for a in actions.values()) 
                                   for actions in agent.action_values.values())
        },
        "depth_distribution": {
            "level_1": max_depths.count(1),
            "level_2": max_depths.count(2),
            "level_3_plus": len([d for d in max_depths if d >= 3])
        }
    }
    
    with open("mcts_training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Training Complete ===")
    print(f"Final statistics:")
    print(f"  Average reward: {summary['final_stats']['avg_reward']:.2f}")
    print(f"  Max depth reached: {summary['final_stats']['max_depth_reached']}")
    print(f"  Unique contexts learned: {summary['final_stats']['unique_contexts']}")
    print(f"  Total action-value updates: {summary['final_stats']['total_experiences']}")
    
    return agent

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--save", type=str, default="mcts_trained_values.pkl", help="Path to save trained values")
    args = parser.parse_args()
    
    train_mcts(num_episodes=args.episodes, save_path=args.save)