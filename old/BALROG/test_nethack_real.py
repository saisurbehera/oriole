#!/usr/bin/env python3
"""
Real NetHack test for hierarchical agent
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import nle
import gymnasium as gym
import numpy as np
from collections import namedtuple
import requests
import json
import time

# Mock LLMResponse for testing
LLMResponse = namedtuple('LLMResponse', ['model_id', 'completion', 'stop_reason', 'input_tokens', 'output_tokens', 'reasoning'])

# Import our hierarchical agent
from balrog.agents.hierarchical_dreamcoder_v2 import HierarchicalDreamCoderAgentV2

class RealLLMClient:
    """Real client that uses the vLLM server"""
    
    def __init__(self):
        self.base_url = "http://0.0.0.0:8080/v1"
        self.model = "meta-llama/Llama-3.2-1B-Instruct"
        
    def generate(self, messages):
        """Generate response using vLLM"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                completion = data["choices"][0]["message"]["content"].strip().lower()
                return LLMResponse(
                    model_id=self.model,
                    completion=completion,
                    stop_reason="complete",
                    input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                    output_tokens=data.get("usage", {}).get("completion_tokens", 0),
                    reasoning="vllm_response"
                )
            else:
                print(f"LLM Error {response.status_code}: {response.text}")
                return LLMResponse(
                    model_id=self.model,
                    completion="search",
                    stop_reason="error",
                    input_tokens=0,
                    output_tokens=0,
                    reasoning="error_fallback"
                )
                
        except Exception as e:
            print(f"LLM Exception: {e}")
            return LLMResponse(
                model_id=self.model,
                completion="search",
                stop_reason="error", 
                input_tokens=0,
                output_tokens=0,
                reasoning="exception_fallback"
            )

class MockPromptBuilder:
    def update_action(self, action): pass
    def update_observation(self, obs): pass
    def reset(self): pass

class MockBaseAgent:
    def __init__(self, client_factory, prompt_builder): pass

def convert_nle_obs_to_balrog_format(nle_obs):
    """Convert raw NLE observation to BALROG-like format"""
    
    # Extract text representation
    chars = nle_obs.get('chars', np.array([]))
    colors = nle_obs.get('colors', np.array([]))
    
    if chars.size > 0:
        # Convert character matrix to ASCII
        ascii_lines = []
        for row in chars:
            line = ''.join([chr(c) if 32 <= c <= 126 else ' ' for c in row])
            ascii_lines.append(line.rstrip())
        
        # Remove empty lines at the end
        while ascii_lines and not ascii_lines[-1].strip():
            ascii_lines.pop()
            
        ascii_map = '\n'.join(ascii_lines)
    else:
        ascii_map = "Unknown map"
    
    # Extract stats
    stats = nle_obs.get('blstats', np.array([]))
    
    if len(stats) >= 26:
        hp_current = stats[10]
        hp_max = stats[11] 
        hunger = stats[18]
        experience = stats[8]
        turn = stats[20]
        
        stats_text = f"HP:{hp_current}({hp_max}) Exp:{experience} T:{turn}"
        if hunger < 1:
            stats_text += " Hungry"
        if hunger < -1: 
            stats_text += " Weak"
    else:
        stats_text = "Stats unknown"
    
    # Extract score/depth info
    score = nle_obs.get('score', 0)
    depth = nle_obs.get('depth', 1)
    
    return {
        'text': {
            'long_term_context': ascii_map,
            'short_term_context': stats_text
        },
        'score': score,
        'depth': depth
    }

def action_string_to_nle_action(action_str):
    """Convert string action to NLE action number"""
    
    # Common NetHack actions mapping
    action_map = {
        'north': 1,      # North
        'northeast': 2,  # Northeast  
        'east': 3,       # East
        'southeast': 4,  # Southeast
        'south': 5,      # South
        'southwest': 6,  # Southwest
        'west': 7,       # West
        'northwest': 8,  # Northwest
        'wait': 9,       # Wait/rest
        'pickup': 10,    # Pickup
        'search': 11,    # Search
        'kick': 12,      # Kick
        'open': 13,      # Open
        'close': 14,     # Close
        'down': 15,      # Go down stairs
        'up': 16,        # Go up stairs
    }
    
    return action_map.get(action_str.lower(), 9)  # Default to wait

def test_with_real_nethack():
    """Test hierarchical agent with real NetHack"""
    
    print("🏰 Testing Hierarchical Agent with Real NetHack")
    print("=" * 50)
    
    # Replace BaseAgent with mock
    import balrog.agents.hierarchical_dreamcoder_v2 as agent_module
    agent_module.BaseAgent = MockBaseAgent
    
    # Create the hierarchical agent
    client_factory = lambda: RealLLMClient()
    prompt_builder = MockPromptBuilder()
    
    agent = HierarchicalDreamCoderAgentV2(client_factory, prompt_builder)
    
    print("✓ Hierarchical agent created")
    
    # Create NetHack environment
    env = gym.make('NetHackChallenge-v0')
    
    print("✓ NetHack environment created")
    print(f"✓ Action space size: {env.action_space.n}")
    
    # Run one episode
    print("\n🎮 Starting NetHack episode...")
    
    obs, info = env.reset()
    total_reward = 0
    total_steps = 0
    max_steps = 100  # Limit for testing
    
    print(f"Initial score: {obs.get('score', 0)}")
    print(f"Initial depth: {obs.get('depth', 1)}")
    
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # Convert NLE observation to BALROG format
        balrog_obs = convert_nle_obs_to_balrog_format(obs)
        
        # Show current map (first few lines)
        map_lines = balrog_obs['text']['long_term_context'].split('\n')[:10]
        print("Current view:")
        for line in map_lines:
            if line.strip():
                print(f"  {line}")
        
        print(f"Stats: {balrog_obs['text']['short_term_context']}")
        
        # Get action from hierarchical agent
        start_time = time.time()
        result = agent.act(balrog_obs)
        thinking_time = time.time() - start_time
        
        action_str = result.completion
        print(f"Agent decision: '{action_str}' (took {thinking_time:.2f}s)")
        print(f"Reasoning: {result.reasoning}")
        
        # Convert to NLE action
        nle_action = action_string_to_nle_action(action_str)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(nle_action)
        done = terminated or truncated
        
        total_reward += reward
        total_steps += 1
        
        print(f"Reward: {reward}, Total: {total_reward}")
        print(f"Score: {obs.get('score', 0)}, Depth: {obs.get('depth', 1)}")
        
        if done:
            print(f"\n🏁 Episode ended after {total_steps} steps")
            break
            
        if reward > 0:
            print("🎉 Got positive reward!")
            
        # Small delay to see what's happening
        time.sleep(0.5)
    
    env.close()
    
    print(f"\n📊 Final Results:")
    print(f"  Steps taken: {total_steps}")
    print(f"  Total reward: {total_reward}")
    print(f"  Final score: {obs.get('score', 0)}")  
    print(f"  Final depth: {obs.get('depth', 1)}")
    
    if total_reward > 0:
        print("✅ Agent achieved positive reward!")
    elif obs.get('score', 0) > 0:
        print("✅ Agent achieved positive score!")
    elif obs.get('depth', 1) > 1:
        print("✅ Agent descended to deeper level!")
    else:
        print("ℹ️  Agent exploring (which is normal for early episodes)")
    
    return total_reward, total_steps

if __name__ == "__main__":
    test_with_real_nethack()