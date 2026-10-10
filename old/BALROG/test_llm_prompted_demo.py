#!/usr/bin/env python3
"""
Demo the LLM Prompted Agent with real NetHack
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import gymnasium as gym
import nle
import time
from collections import namedtuple

# Setup
LLMResponse = namedtuple('LLMResponse', ['model_id', 'completion', 'stop_reason', 'input_tokens', 'output_tokens', 'reasoning'])

from balrog.agents.llm_prompted_agent import LLMPromptedAgent, PromptStrategy

class VLLMClient:
    """vLLM client"""
    def __init__(self):
        self.base_url = "http://0.0.0.0:8080/v1"
        self.model = "meta-llama/Llama-3.2-1B-Instruct"
        
    def generate(self, messages):
        import requests
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 30,
                    "temperature": 0.1
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return LLMResponse(
                    model_id=self.model,
                    completion=data["choices"][0]["message"]["content"].strip(),
                    stop_reason="complete",
                    input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                    output_tokens=data.get("usage", {}).get("completion_tokens", 0),
                    reasoning="vllm"
                )
        except Exception as e:
            print(f"LLM Error: {e}")
            
        return LLMResponse(self.model, "wait", "error", 0, 0, "error")

class MockPromptBuilder:
    def update_action(self, action): pass
    def update_observation(self, obs): pass
    def reset(self): pass

class MockBaseAgent:
    def __init__(self, client_factory, prompt_builder): pass

def convert_obs_to_balrog(nle_obs):
    """Convert NLE observation to BALROG format"""
    chars = nle_obs.get('chars', [[]])
    
    # Convert to ASCII
    ascii_lines = []
    for row in chars:
        line = ''.join([chr(int(c)) if 32 <= int(c) <= 126 else ' ' for c in row])
        ascii_lines.append(line)
        
    # Get the main game area (skip status lines)
    game_lines = []
    for i, line in enumerate(ascii_lines):
        if i < len(ascii_lines) - 2:  # Skip last 2 status lines
            game_lines.append(line)
            
    ascii_map = '\n'.join(game_lines) if game_lines else "No map data"
    
    # Stats
    stats = nle_obs.get('blstats', [0] * 26)
    hp_current = int(stats[10]) if len(stats) > 10 else 10
    hp_max = int(stats[11]) if len(stats) > 11 else 10
    
    return {
        'text': {
            'long_term_context': ascii_map,
            'short_term_context': f"HP:{hp_current}({hp_max})"
        },
        'score': nle_obs.get('score', 0),
        'depth': nle_obs.get('depth', 1)
    }

def action_to_nle(action_str):
    """Convert string action to NLE number"""
    mapping = {
        'north': 1, 'northeast': 2, 'east': 3, 'southeast': 4,
        'south': 5, 'southwest': 6, 'west': 7, 'northwest': 8,
        'wait': 9, 'pickup': 10, 'search': 11, 'kick': 12,
        'open': 13, 'close': 14, 'down': 15, 'up': 16
    }
    return mapping.get(action_str.lower(), 9)

def main():
    print("🎮 LLM PROMPTED AGENT DEMO")
    print("=" * 50)
    
    # Mock base agent
    import balrog.agents.llm_prompted_agent as agent_module
    agent_module.BaseAgent = MockBaseAgent
    
    # Create agent with step-by-step strategy (best performing)
    agent = LLMPromptedAgent(
        lambda: VLLMClient(),
        MockPromptBuilder(),
        prompt_strategy=PromptStrategy.STEP_BY_STEP
    )
    
    print("✓ Agent created with step-by-step prompting")
    
    # Create environment
    env = gym.make('NetHackChallenge-v0')
    print("✓ NetHack environment created")
    
    # Run episode
    obs, info = env.reset()
    
    print("\n🏃 Running episode...")
    print("-" * 50)
    
    total_reward = 0
    for step in range(30):  # 30 steps max
        # Convert observation
        balrog_obs = convert_obs_to_balrog(obs)
        
        # Show game state (first 5 lines)
        map_lines = balrog_obs['text']['long_term_context'].split('\n')[:5]
        print(f"\nStep {step + 1}:")
        for line in map_lines:
            if line.strip():
                print(f"  {line[:50]}")  # First 50 chars
                
        # Get action
        start = time.time()
        result = agent.act(balrog_obs)
        elapsed = time.time() - start
        
        action_str = result.completion
        print(f"Action: '{action_str}' ({elapsed:.2f}s)")
        print(f"Context: {result.reasoning}")
        
        # Execute
        nle_action = action_to_nle(action_str)
        obs, reward, terminated, truncated, info = env.step(nle_action)
        
        total_reward += reward
        
        if reward > 0:
            print(f"🎉 Got reward: {reward}")
            
        if terminated or truncated:
            print(f"\n🏁 Episode ended!")
            break
            
        time.sleep(0.2)  # Slow down to see what's happening
        
    env.close()
    
    # Results
    print("\n" + "=" * 50)
    print("📊 RESULTS:")
    print(f"  Total reward: {total_reward}")
    print(f"  Final score: {obs.get('score', 0)}")
    print(f"  Final depth: {obs.get('depth', 1)}")
    print(f"  Steps taken: {step + 1}")
    
    if obs.get('depth', 1) > 1:
        print("\n✅ SUCCESS! Agent descended to a deeper level!")
    elif obs.get('score', 0) > 0:
        print("\n✅ SUCCESS! Agent achieved positive score!")
    else:
        print("\n📈 Agent explored the dungeon (normal for early learning)")

if __name__ == "__main__":
    main()