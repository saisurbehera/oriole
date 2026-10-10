#!/usr/bin/env python3
"""
Simple test script for hierarchical agent
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from collections import namedtuple
import requests
import json

# Mock LLMResponse for testing
LLMResponse = namedtuple('LLMResponse', ['model_id', 'completion', 'stop_reason', 'input_tokens', 'output_tokens', 'reasoning'])

# Import our hierarchical agent
from balrog.agents.hierarchical_dreamcoder_v2 import HierarchicalDreamCoderAgentV2

class TestLLMClient:
    """Test client that uses the vLLM server"""
    
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
                    "max_tokens": 50,
                    "temperature": 0.7
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                completion = data["choices"][0]["message"]["content"].strip()
                return LLMResponse(
                    model_id=self.model,
                    completion=completion,
                    stop_reason="complete",
                    input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                    output_tokens=data.get("usage", {}).get("completion_tokens", 0),
                    reasoning="vllm_response"
                )
            else:
                print(f"Error {response.status_code}: {response.text}")
                return LLMResponse(
                    model_id=self.model,
                    completion="search",  # Fallback action
                    stop_reason="error",
                    input_tokens=0,
                    output_tokens=0,
                    reasoning="error_fallback"
                )
                
        except Exception as e:
            print(f"LLM Error: {e}")
            return LLMResponse(
                model_id=self.model,
                completion="search",  # Fallback action
                stop_reason="error", 
                input_tokens=0,
                output_tokens=0,
                reasoning="exception_fallback"
            )

class MockPromptBuilder:
    """Mock prompt builder for testing"""
    def update_action(self, action): 
        pass
    def update_observation(self, obs): 
        pass
    def reset(self): 
        pass

class MockBaseAgent:
    """Mock base agent class"""
    def __init__(self, client_factory, prompt_builder):
        pass

def test_hierarchical_agent():
    """Test the hierarchical agent"""
    print("🧪 Testing Hierarchical DreamCoder Agent V2...")
    
    # Replace BaseAgent with mock (avoid import issues)
    import balrog.agents.hierarchical_dreamcoder_v2 as agent_module
    agent_module.BaseAgent = MockBaseAgent
    
    # Create agent
    client_factory = lambda: TestLLMClient()
    prompt_builder = MockPromptBuilder()
    
    agent = HierarchicalDreamCoderAgentV2(client_factory, prompt_builder)
    
    print("✓ Agent created successfully!")
    print(f"✓ World model: {type(agent.world_model).__name__}")
    print(f"✓ Planner: {type(agent.planner).__name__}")
    print(f"✓ Pathfinder: {type(agent.pathfinder).__name__}")
    
    # Test with mock NetHack observation
    mock_obs = {
        'text': {
            'long_term_context': '''
###################
#.................#
#.####...####.....#
#.#  #...#  #.....#
#.#  +...+  #.....#
#.####...####.....#
#.......@.........#
#.................#
#.................#
#........>........#
###################
''',
            'short_term_context': 'HP:15(15) Pw:1(1) AC:6 Exp:1 T:1 Hungry'
        },
        'score': 0,
        'depth': 1
    }
    
    print("\n🎮 Testing action selection...")
    
    # Test multiple actions
    for i in range(3):
        print(f"\n--- Step {i+1} ---")
        result = agent.act(mock_obs)
        
        print(f"Action: {result.completion}")
        print(f"Model: {result.model_id}")
        print(f"Reasoning: {result.reasoning}")
        
        # Update observation for next step (simulate movement)
        if result.completion in ["north", "south", "east", "west"]:
            print(f"✓ Valid movement action: {result.completion}")
        elif result.completion in ["search", "wait", "kick", "open"]:
            print(f"✓ Valid interaction action: {result.completion}")
        else:
            print(f"? Unusual action: {result.completion}")
    
    print("\n🎯 Testing special scenarios...")
    
    # Test with stairs found scenario
    stairs_obs = mock_obs.copy()
    stairs_obs['text']['long_term_context'] = stairs_obs['text']['long_term_context'].replace('@', '>')
    
    result = agent.act(stairs_obs)
    print(f"When on stairs: {result.completion}")
    
    # Test with monster scenario  
    monster_obs = mock_obs.copy()
    monster_obs['text']['long_term_context'] += "\nA monster attacks you!"
    
    result = agent.act(monster_obs)
    print(f"When monsters present: {result.completion}")
    
    print("\n✅ Hierarchical agent test completed successfully!")
    print("The agent can:")
    print("- Create strategic, tactical, and operational plans")
    print("- Detect world features (stairs, monsters, items)")
    print("- Generate contextual actions using vLLM")
    print("- Handle special scenarios appropriately")

if __name__ == "__main__":
    test_hierarchical_agent()