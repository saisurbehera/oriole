#!/usr/bin/env python3

"""
Test JSON parsing for planning agent
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.planning_first_agent import PlanningFirstAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder

def test_json_parsing():
    # Create mock config
    class MockConfig:
        def __init__(self):
            self.client_name = "openai"
            self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
            self.base_url = "http://0.0.0.0:8080/v1"
            self.generate_kwargs = {"temperature": 0.2, "max_tokens": 300}
            self.timeout = 60
            self.max_retries = 3
            self.delay = 2
            self.alternate_roles = False
    
    class MockAgentConfig:
        def __init__(self):
            self.max_text_history = 8
            self.max_image_history = 0
            self.max_cot_history = 1
            self.max_icl_history = 0
            self.cache_icl = False
            
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    # Create agent
    config = MockConfig()
    agent_config = MockAgentConfig()
    
    client_factory = create_llm_client(config)
    prompt_builder = create_prompt_builder(agent_config)
    
    agent = PlanningFirstAgent(client_factory, prompt_builder)
    
    print("=== Testing JSON Parsing ===")
    
    # Test 1: Valid JSON
    test_response = '''Here is my plan:
{
  "situation": "I am in a dungeon room with stairs nearby",
  "threats": ["kobold zombie", "low health"],
  "opportunities": ["stairs up", "safe area"],
  "goal": "reach stairs safely",
  "steps": [
    "Move toward stairs",
    "Avoid zombie",
    "Use stairs to ascend"
  ]
}
That's my analysis.'''
    
    plan = agent._parse_plan_response(test_response)
    if plan:
        print("✅ JSON parsing successful!")
        print(f"Goal: {plan.goal}")
        print(f"Steps: {plan.steps}")
        print(f"Threats: {plan.threats}")
    else:
        print("❌ JSON parsing failed")
    
    # Test 2: Test actual LLM call
    print("\n=== Testing LLM JSON Response ===")
    
    prompt = """You are playing NetHack. You have excellent health (18/18 HP). You are on dungeon level 1. Stairs going up are east.

Respond with ONLY valid JSON in this exact format:

{
  "situation": "Brief description of your current state",
  "threats": ["danger1", "danger2"],
  "opportunities": ["helpful_thing1", "helpful_thing2"],
  "goal": "One clear objective",
  "steps": [
    "Step 1: specific action and reason",
    "Step 2: specific action and reason",
    "Step 3: specific action and reason"
  ]
}

Be specific about directions and actions."""
    
    try:
        messages = [{"role": "user", "content": prompt}]
        response = agent.client.generate(messages)
        print(f"LLM Response: {response.completion}")
        
        plan = agent._parse_plan_response(response.completion)
        if plan:
            print("✅ LLM JSON parsing successful!")
            print(f"Goal: {plan.goal}")
            print(f"Steps: {plan.steps}")
        else:
            print("❌ LLM JSON parsing failed")
            
    except Exception as e:
        print(f"LLM call failed: {e}")

if __name__ == "__main__":
    test_json_parsing()