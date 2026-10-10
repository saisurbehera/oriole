#!/usr/bin/env python3

"""
Debug script to see what the planning agent is actually doing
"""

import sys
sys.path.append('/home/sai/Desktop/oriole/BALROG')

from balrog.agents.planning_first_agent import PlanningFirstAgent
from balrog.client import create_llm_client
from balrog.prompt_builder import create_prompt_builder

def test_planning():
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
    
    # Mock observation
    mock_obs = {
        "text": {
            "long_term_context": """message:


language observation:
vertical wall near west
horizontal wall very near north, south, southwest, and northwest
northeast room corner very near northeast
vertical wall very near east
southeast corner very near southeast
horizontal closed door very near southsoutheast
tame little dog adjacent northeast
stairs up adjacent east

cursor:
Yourself a cavewoman
(x=21, y=8)

map:
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                  ------                                                        
                  |...d|                                                        
                  |..@<|                                                        
                  |.....                                                        
                  ----+-                                                        
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
Agent the Troglodyte           St:18/02 Dx:11 Co:19 In:10 Wi:8 Ch:7 Lawful S:0  
Dlvl:1 $:0 HP:18(18) Pw:1(1) AC:8 Xp:1/0 T:2                                    

""",
            "language_observation": "stairs up adjacent east"
        }
    }
    
    # Test planning
    print("=== Testing Planning Agent ===")
    
    # First action (should trigger planning)
    print("\n1. First action (planning should trigger):")
    response1 = agent.act(mock_obs)
    print(f"Action: {response1.completion}")
    print(f"Reasoning: {response1.reasoning}")
    
    if agent.current_plan:
        print(f"\nPlan created:")
        print(f"Goal: {agent.current_plan.goal}")
        print(f"Steps: {agent.current_plan.steps}")
        print(f"Situation: {agent.current_plan.situation_understanding}")
    
    # Second action (should execute plan)
    print("\n2. Second action (should execute plan):")
    response2 = agent.act(mock_obs)
    print(f"Action: {response2.completion}")
    print(f"Reasoning: {response2.reasoning}")
    
    # Show internal state
    print(f"\nInternal state:")
    print(f"Plan step: {agent.plan_step}")
    print(f"Plan status: {agent.plan_status}")
    print(f"Turn count: {agent.turn_count}")

if __name__ == "__main__":
    test_planning()