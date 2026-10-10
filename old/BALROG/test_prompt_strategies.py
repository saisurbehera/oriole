#!/usr/bin/env python3
"""
Test different prompting strategies with vLLM
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import gymnasium as gym
import nle
import requests
import json
import time
from collections import namedtuple

# Mock the base classes
LLMResponse = namedtuple('LLMResponse', ['model_id', 'completion', 'stop_reason', 'input_tokens', 'output_tokens', 'reasoning'])

from balrog.agents.llm_prompted_agent import LLMPromptedAgent, PromptStrategy, ActionContext

class VLLMClient:
    """Real vLLM client"""
    def __init__(self):
        self.base_url = "http://0.0.0.0:8080/v1"
        self.model = "meta-llama/Llama-3.2-1B-Instruct"
        
    def generate(self, messages):
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 20,  # Short responses
                    "temperature": 0.1  # More deterministic
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
                print(f"Error: {response.text}")
                return LLMResponse(self.model, "wait", "error", 0, 0, "error")
                
        except Exception as e:
            print(f"Exception: {e}")
            return LLMResponse(self.model, "wait", "error", 0, 0, "exception")

class MockPromptBuilder:
    def update_action(self, action): pass
    def update_observation(self, obs): pass
    def reset(self): pass

class MockBaseAgent:
    def __init__(self, client_factory, prompt_builder): pass

def test_prompt_strategy(strategy: PromptStrategy, test_cases):
    """Test a specific prompting strategy"""
    print(f"\n{'='*50}")
    print(f"Testing {strategy.value} strategy")
    print('='*50)
    
    # Mock the base agent
    import balrog.agents.llm_prompted_agent as agent_module
    agent_module.BaseAgent = MockBaseAgent
    
    # Create agent
    agent = LLMPromptedAgent(
        lambda: VLLMClient(),
        MockPromptBuilder(),
        prompt_strategy=strategy
    )
    
    results = []
    
    for i, (name, context) in enumerate(test_cases):
        print(f"\nTest {i+1}: {name}")
        print(f"Context: stairs={context.stairs_visible}, "
              f"dir={context.stairs_direction}, "
              f"monster={context.monster_adjacent}")
        
        # Get action using internal method
        start_time = time.time()
        
        if strategy == PromptStrategy.MULTIPLE_CHOICE:
            action = agent._get_action_multiple_choice(context)
        elif strategy == PromptStrategy.CONSTRAINED_GENERATION:
            action = agent._get_action_constrained(context)
        elif strategy == PromptStrategy.EXAMPLE_BASED:
            action = agent._get_action_with_examples(context)
        else:  # STEP_BY_STEP
            action = agent._get_action_step_by_step(context)
            
        elapsed = time.time() - start_time
        
        print(f"→ Action: '{action}' (took {elapsed:.2f}s)")
        
        # Evaluate quality
        is_good = False
        if context.stairs_visible and context.stairs_direction:
            is_good = action == context.stairs_direction
        elif context.monster_adjacent:
            is_good = action in ["wait", "kick"]
        elif context.door_present:
            is_good = action in ["kick", "open"]
        else:
            is_good = action in ["north", "south", "east", "west", "search"]
            
        results.append((name, action, is_good, elapsed))
        print(f"✓ Good" if is_good else "✗ Suboptimal")
        
    # Summary
    good_count = sum(1 for _, _, is_good, _ in results if is_good)
    avg_time = sum(t for _, _, _, t in results) / len(results)
    
    print(f"\n{strategy.value} Summary:")
    print(f"  Success rate: {good_count}/{len(results)} ({100*good_count/len(results):.0f}%)")
    print(f"  Average time: {avg_time:.2f}s")
    
    return good_count, avg_time

def main():
    print("🧪 TESTING PROMPT STRATEGIES FOR NETHACK")
    
    # Test cases
    test_cases = [
        ("Stairs visible north", ActionContext(
            stairs_visible=True, stairs_direction="north"
        )),
        ("Stairs visible east", ActionContext(
            stairs_visible=True, stairs_direction="east"
        )),
        ("Monster adjacent", ActionContext(
            monster_adjacent=True, monster_present=True
        )),
        ("Door present", ActionContext(
            door_present=True
        )),
        ("Empty room", ActionContext()),
        ("Low HP with monster", ActionContext(
            hp_low=True, monster_present=True
        )),
        ("Stairs visible south", ActionContext(
            stairs_visible=True, stairs_direction="south"
        )),
        ("Hungry in corridor", ActionContext(
            hungry=True, in_corridor=True, in_room=False
        )),
    ]
    
    # Test each strategy
    results = {}
    for strategy in PromptStrategy:
        success_count, avg_time = test_prompt_strategy(strategy, test_cases)
        results[strategy] = (success_count, avg_time)
        time.sleep(1)  # Be nice to the server
        
    # Final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    
    for strategy, (success, avg_time) in results.items():
        print(f"{strategy.value:20} - Success: {success}/8 ({100*success/8:.0f}%), "
              f"Time: {avg_time:.2f}s")
        
    # Recommend best
    best_strategy = max(results.items(), key=lambda x: (x[1][0], -x[1][1]))[0]
    print(f"\n🏆 BEST STRATEGY: {best_strategy.value}")
    print(f"   Use this in your config: prompt_strategy: {best_strategy.value}")

if __name__ == "__main__":
    main()