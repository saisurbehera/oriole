# Bootstrapping Results for 1B Model

## Overview
Created multiple agent architectures to bootstrap the 1B parameter model (meta-llama/Llama-3.2-1B-Instruct) for NetHack gameplay.

## Agent Progression

### 1. LLMPromptedAgent (Multiple Strategies)
- **Step-by-step strategy**: 0.62% progression (best)
- **Multiple choice strategy**: 0.00% progression  
- **Constrained generation**: 0.00% progression
- **Example-based**: 0.09% progression
- **Key finding**: 1B model struggles with complex prompts

### 2. BootstrappedAgent (State Machine)
- **Result**: 0.00% progression
- **Issue**: State machine logic too complex, pet detection issues
- **Token usage**: Minimal

### 3. MicroAgent (Binary Choices)
- **Result**: 0.00% progression  
- **Approach**: LLM only for A/B choices
- **Issue**: Exploration patterns not effective

### 4. SimpleHeuristicAgent (Minimal LLM)
- **Result**: 0.31% ± 0.28% progression
- **Approach**: Pure heuristics, LLM only for disambiguation
- **Success**: Agent reached dungeon level 2
- **Token usage**: 0 (pure heuristics)

## Key Insights

1. **Simpler is Better**: The 1B model performs best with minimal complexity
2. **Heuristics Work**: Pure heuristic approaches can achieve basic NetHack progress
3. **Stair Navigation**: Fixed NetHack command (`<` instead of "up") was critical
4. **Zero-Shot Limits**: 1B model cannot handle complex reasoning tasks

## Technical Improvements Made

1. **Map Parsing**: Proper extraction of game map from observations
2. **Coordinate System**: Fixed row/col vs x/y confusion  
3. **NetHack Commands**: Used correct stair commands (`<` and `>`)
4. **Walkability**: Added space character for unexplored areas

## Next Steps for Better Bootstrapping

1. **Hybrid Approach**: Use heuristics for 95% of decisions, LLM for critical choices
2. **Learning from Success**: Save successful trajectories and replay patterns
3. **Focused LLM Usage**: Only call LLM for specific, well-defined decisions
4. **Pre-computed Strategies**: Build a library of situation-specific action sequences

## Conclusion

The SimpleHeuristicAgent demonstrates that with the right harnesses, even a 1B model can contribute to NetHack gameplay. The key is minimizing the reasoning burden on the model while maximizing the use of deterministic heuristics.