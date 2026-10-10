#!/usr/bin/env python3

"""
Training script for Real LLM-MCTS agent
Runs multiple episodes to improve performance through experience
"""

import sys
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.real_llm_mcts import RealLLMMCTSAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder
from balrog.environments.nle.nle_env import make_nle_env

class TrainingConfig:
    """Configuration for training"""
    def __init__(self):
        # Client config for LLM
        self.client_name = "vllm"
        self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
        self.base_url = "http://localhost:8080/v1"
        self.generate_kwargs = {"temperature": 1.0, "max_tokens": 4096}
        self.timeout = 60
        self.max_retries = 5
        self.delay = 2
        self.alternate_roles = False
        
        # Agent config
        self.max_text_history = 16
        self.max_image_history = 0
        self.max_cot_history = 1
        self.max_icl_history = 1000
        self.cache_icl = False
        
        # Environment config
        self.nle_kwargs = {
            "character": "@",
            "max_episode_steps": 200,  # Longer episodes for exploration
            "no_progress_timeout": 150,
            "savedir": None,
            "archivefile": None,
            "allow_all_yn_questions": True,
            "allow_all_modes": True,
        }
        
        # Training config
        self.num_episodes = 20  # Start with fewer episodes due to LLM cost
        self.save_interval = 5   # Save progress every 5 episodes
        self.max_steps_per_episode = 200
        
    def get(self, key, default=None):
        return getattr(self, key, default)

def train_real_llm_mcts(config=None):
    """Train the Real LLM-MCTS agent"""
    
    if config is None:
        config = TrainingConfig()
    
    print("🚀 Starting Real LLM-MCTS Training...")
    print(f"Episodes: {config.num_episodes}")
    print(f"Max steps per episode: {config.max_steps_per_episode}")
    print(f"Model: {config.model_id}")
    
    # Create agent
    client_factory = create_llm_client(config)
    prompt_builder = create_prompt_builder(config)
    agent = RealLLMMCTSAgent(client_factory, prompt_builder)
    
    # Create environment
    task = "NetHackChallenge-v0"
    
    # Create a mock config structure for environment
    class MockEnvConfig:
        def __init__(self):
            self.envs = config
    
    env_config = MockEnvConfig()
    env_config.envs.nle_kwargs = config.nle_kwargs
    env_config.agent = config
    
    env = make_nle_env("nle", task, env_config)
    
    # Training tracking
    training_results = {
        'episodes': [],
        'best_episode_return': 0.0,
        'stairs_found_count': 0,
        'level_2_reached_count': 0,
        'start_time': datetime.now().isoformat()
    }
    
    total_start_time = time.time()
    
    for episode in range(config.num_episodes):
        print(f"\n=== Episode {episode + 1}/{config.num_episodes} ===")
        episode_start_time = time.time()
        
        # Reset environment and agent
        obs = env.reset()
        agent.reset()
        
        episode_rewards = []
        episode_actions = []
        steps = 0
        done = False
        
        while not done and steps < config.max_steps_per_episode:
            # Get action from agent
            action_response = agent.act(obs)
            action = action_response.completion
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_actions.append(action)
            steps += 1
            
            # Log progress periodically
            if steps % 20 == 0:
                current_depth = info.get('dungeon_level', 1) if info else 1
                print(f"  Step {steps}: action={action}, reward={reward:.2f}, depth={current_depth}")
        
        # Episode summary
        episode_time = time.time() - episode_start_time
        total_reward = sum(episode_rewards)
        final_depth = info.get('dungeon_level', 1) if info else 1
        stairs_found = '>' in str(obs.get('text', {}).get('map', '')) if obs else False
        level_2_reached = final_depth > 1
        
        # Track statistics
        if stairs_found:
            training_results['stairs_found_count'] += 1
        if level_2_reached:
            training_results['level_2_reached_count'] += 1
        if total_reward > training_results['best_episode_return']:
            training_results['best_episode_return'] = total_reward
        
        # Save episode results
        episode_result = {
            'episode': episode + 1,
            'steps': steps,
            'total_reward': total_reward,
            'episode_time': episode_time,
            'final_depth': final_depth,
            'stairs_found': stairs_found,
            'level_2_reached': level_2_reached,
            'actions': episode_actions,
            'action_counts': {action: episode_actions.count(action) for action in set(episode_actions)}
        }
        training_results['episodes'].append(episode_result)
        
        print(f"Episode {episode + 1} Results:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Final Depth: {final_depth}")
        print(f"  Stairs Found: {stairs_found}")
        print(f"  Level 2 Reached: {level_2_reached}")
        print(f"  Time: {episode_time:.1f}s")
        print(f"  Top Actions: {dict(list(episode_result['action_counts'].items())[:5])}")
        
        # Save progress periodically
        if (episode + 1) % config.save_interval == 0:
            save_training_progress(training_results, episode + 1)
    
    # Final training summary
    total_time = time.time() - total_start_time
    print(f"\n🎯 Training Complete!")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Episodes: {config.num_episodes}")
    print(f"Best Episode Return: {training_results['best_episode_return']:.2f}")
    print(f"Stairs Found: {training_results['stairs_found_count']}/{config.num_episodes} episodes")
    print(f"Level 2 Reached: {training_results['level_2_reached_count']}/{config.num_episodes} episodes")
    
    # Calculate success rates
    stairs_rate = training_results['stairs_found_count'] / config.num_episodes * 100
    level_2_rate = training_results['level_2_reached_count'] / config.num_episodes * 100
    print(f"Stairs Discovery Rate: {stairs_rate:.1f}%")
    print(f"Level 2 Success Rate: {level_2_rate:.1f}%")
    
    # Save final results
    save_training_progress(training_results, "final")
    
    return training_results

def save_training_progress(results, episode_num):
    """Save training progress to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path("training_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    filename = f"real_llm_mcts_training_{episode_num}_{timestamp}.json"
    filepath = results_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📁 Progress saved: {filepath}")

if __name__ == "__main__":
    # Run training
    config = TrainingConfig()
    
    # Allow command-line customization
    if len(sys.argv) > 1:
        config.num_episodes = int(sys.argv[1])
    if len(sys.argv) > 2:
        config.max_steps_per_episode = int(sys.argv[2])
    
    train_real_llm_mcts(config)