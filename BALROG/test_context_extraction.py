#!/usr/bin/env python3
"""Test context extraction"""

import sys
sys.path.insert(0, '.')

from balrog.agents.llm_prompted_agent import ActionContext, LLMPromptedAgent

# Sample NetHack observation
sample_obs = {
    'text': {
        'long_term_context': """
                                          ---------                             
                                          |$......+                             
                                          ....d@..|                             
                                          |....$..|                             
                                          ---------                             
""",
        'short_term_context': "Agent the Plunderess           St:18/03 Dx:16 Co:16 In:7 Wi:8 Ch:6 Neutral S:0  Dlvl:1 $:0 HP:16(16) Pw:2(2) AC:7 Xp:1/0 T:31"
    }
}

# Mock the agent to test context extraction
class MockClient:
    pass

class MockPromptBuilder:
    def update_action(self, action): pass
    def update_observation(self, obs): pass

# Create agent instance
agent = LLMPromptedAgent(lambda: MockClient(), MockPromptBuilder())

# Extract context
context = agent._extract_context(sample_obs)

print("Context Extraction Test")
print("=" * 50)
print(f"Map:\n{sample_obs['text']['long_term_context']}")
print("-" * 50)
print(f"Stairs visible: {context.stairs_visible}")
print(f"Stairs direction: {context.stairs_direction}")
print(f"Door present: {context.door_present}")
print(f"Monster present: {context.monster_present}")
print(f"Monster adjacent: {context.monster_adjacent}")
print(f"Item visible: {context.item_visible}")
print(f"HP low: {context.hp_low}")
print(f"HP critical: {context.hp_critical}")
print(f"Hungry: {context.hungry}")

# Test with stairs
print("\n" + "=" * 50)
print("Testing with stairs:")
sample_obs2 = {
    'text': {
        'long_term_context': """
                                          ---------                             
                                          |......>|                             
                                          .....@..|                             
                                          |.......|                             
                                          ---------                             
""",
        'short_term_context': "HP:16(16)"
    }
}

context2 = agent._extract_context(sample_obs2)
print(f"Stairs visible: {context2.stairs_visible}")
print(f"Stairs direction: {context2.stairs_direction}")