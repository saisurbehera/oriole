# Hierarchical RL Results Summary

## Overview
Created hierarchical RL agents that use LLM for high-level decisions and heuristics for execution.

## Agents Tested

### 1. HierarchicalRLAgent
- **Approach**: LLM selects skills, heuristics execute them
- **Results**: 0.00% progression
- **Token Usage**: 181,700 input / 18,170 output tokens
- **Skills Available**: 13 different skills (explore, combat, flee, etc.)
- **Issue**: Agent stuck in "explore_new" skill repeatedly

### 2. ExpertGuidedRLAgent  
- **Approach**: Uses expert patterns and strategies
- **Results**: 0.00% progression
- **Token Usage**: 410,100 input / 41,010 output tokens
- **Strategies**: speed_descent, full_exploration, tactical_combat
- **Issue**: High token usage but no better performance

## Key Findings

### What Worked:
1. **Skill abstraction**: Breaking down NetHack into high-level skills is conceptually sound
2. **Heuristic execution**: Low-level execution without LLM works
3. **Context extraction**: Successfully parsed game state for decisions

### What Didn't Work:
1. **LLM skill selection**: 1B model couldn't select appropriate skills
2. **Complex prompts**: Too much information overwhelmed the model
3. **Strategy learning**: No improvement over episodes
4. **Token efficiency**: Used many tokens without benefit

## Token Usage Analysis
- Simple heuristic agent: 0 tokens, 0.31% progression
- Hierarchical RL: 200K tokens, 0.00% progression  
- Expert-guided RL: 450K tokens, 0.00% progression

**Conclusion**: More LLM usage doesn't improve performance with 1B model

## Recommendations

### 1. Simplify Skill Selection
Instead of 13 skills, reduce to 3-4 core skills:
- EXPLORE
- COMBAT
- DESCEND
- ESCAPE

### 2. Binary Decision Tree
Use LLM for simple binary choices:
- "Fight or flee?"
- "Explore or descend?"
- "Left or right?"

### 3. Pre-trained Patterns
Instead of learning online, pre-compute successful patterns from expert data

### 4. Hybrid Approach
- Use heuristics for 95% of decisions
- LLM only for critical junctions
- Maximum 1-2 LLM calls per 100 steps

## Next Steps

1. **Simplify skills** to 3-4 core behaviors
2. **Extract expert patterns** from trajectory data
3. **Create decision tree** with binary LLM choices
4. **Test with minimal LLM calls**
5. **Focus on one aspect** (e.g., just reaching stairs)

The hierarchical approach is sound, but needs extreme simplification for 1B models.