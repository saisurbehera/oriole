#!/usr/bin/env python3

"""
Test script to verify knowledge server integration
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.planning_first_agent import PlanningFirstAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder

def test_knowledge_integration():
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
    
    print("=== Testing Knowledge Server Integration ===")
    
    # Test 1: Combat advice
    print("\n1. Testing combat advice:")
    kobold_advice = agent.get_combat_advice("kobold zombie")
    print(f"Kobold zombie advice: {kobold_advice}")
    
    # Test 2: Item advice  
    print("\n2. Testing item advice:")
    sword_advice = agent.get_item_advice(")")
    print(f"')' symbol advice: {sword_advice}")
    
    # Test 3: Level advice
    print("\n3. Testing level advice:")
    level_advice = agent.get_level_advice(1)
    print(f"Level 1 advice: {level_advice}")
    
    # Test 4: Direct query
    print("\n4. Testing direct query:")
    health_advice = agent.query_knowledge_server("How can I heal when low on health in NetHack?")
    print(f"Healing advice: {health_advice}")

if __name__ == "__main__":
    test_knowledge_integration()