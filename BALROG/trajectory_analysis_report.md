# NetHack LLM Agent Trajectory Analysis Report

## Executive Summary

After analyzing the trajectory files from both `hierarchical_rl` and `expert_guided_rl` agents using Llama-3.2-1B, I've identified critical failures in the LLM outputs that explain why both agents achieve 0% progression.

## Key Findings

### 1. Invalid Action Outputs (Expert-Guided Agent)

The expert_guided_rl agent frequently outputs invalid actions that cannot be parsed by NetHack:
- **Comma (`,`)**: Output 470 times - intended for item pickup but not a valid NetHack command
- **Less-than (`<`)**: Output 180 times - intended for descending stairs but should be `>`
- **Double quotes (`"`)**: Output as actions, causing parse failures

**Root Cause**: The agent's pattern definitions in `expert_guided_rl_agent.py` contain raw characters instead of valid NetHack commands:
```python
# Line 175
action_sequence=["navigate_to_item", ","]  # Should be "pickup" or similar

# Line 139  
action_sequence=["navigate_to_stairs", "<"]  # Should be ">" for down stairs
```

### 2. Action Distribution Analysis

**Hierarchical RL Agent** (10,681 steps across 10 runs):
- Top action: `northeast` (2,129 times combined with space prefix)
- Valid directional movements dominate
- No invalid action errors
- Reasoning: Mostly "Skill: explore_new" (3,529 times)

**Expert-Guided RL Agent** (11,318 steps across 5 runs):
- Top action: `wait` (2,451 times) - suggests defensive/stuck behavior
- Many invalid actions defaulting to `esc`
- Reasoning: Mix of strategies like "full_exploration" and "speed_descent"

### 3. LLM Understanding Issues

Both agents show signs of poor prompt understanding:

1. **Hierarchical Agent**: 
   - Successfully outputs valid actions
   - But reasoning is repetitive ("Skill: explore_new" dominates)
   - Suggests limited strategic thinking

2. **Expert-Guided Agent**:
   - Outputs invalid characters as actions
   - Suggests the LLM doesn't understand the action space
   - Strategy selection seems reasonable but execution fails

### 4. Token Usage Comparison

- **Hierarchical**: 181,700 input / 18,170 output tokens (10:1 ratio)
- **Expert-Guided**: 410,100 input / 41,010 output tokens (10:1 ratio)
- Expert-guided uses 2.3x more tokens for half the episodes

## Failure Patterns

### Pattern 1: Action Space Misunderstanding
The expert_guided agent's LLM outputs raw symbols (`,`, `<`, `"`) instead of valid NetHack commands. This suggests:
- Poor prompt design that doesn't clearly specify valid actions
- The 1B model may be too small to understand the action mapping
- Hard-coded invalid actions in pattern definitions

### Pattern 2: Exploration Loop
The hierarchical agent gets stuck in exploration loops:
- Repeatedly outputs movement commands in patterns
- Doesn't adapt strategy when stuck
- Suggests poor state tracking or memory

### Pattern 3: No Learning/Adaptation
Both agents show no progression (0% in all runs), indicating:
- No effective learning from failures
- Strategies don't improve over time
- State/context not properly communicated to LLM

## Recommendations

1. **Fix Action Mappings**: Update expert_guided patterns to use valid NetHack commands
2. **Improve Prompts**: Clearly specify valid action space and provide examples
3. **Add Action Validation**: Validate LLM outputs before sending to environment
4. **Enhance State Representation**: Better communicate game state changes to LLM
5. **Consider Larger Models**: 1B parameters may be insufficient for this task
6. **Add Feedback Loop**: Include previous action outcomes in prompts

## Technical Details

The analysis revealed that CSV parsing in trajectories has issues with embedded commas and special characters in observations. The expert_guided agent's patterns directly include invalid NetHack commands that need translation but aren't getting it.

Both agents suffer from the small model size (Llama-3.2-1B) which appears unable to properly understand and follow the complex prompts required for NetHack gameplay.