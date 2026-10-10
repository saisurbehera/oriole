# MCTS Integration Summary

## Work Completed

### 1. Planning-First Agent Enhancement
- Successfully integrated NetHack knowledge server (39,504 facts)
- Added query caching to prevent repeated requests
- Implemented JSON-based planning format for reliable 1B model parsing
- Fixed stair navigation logic (> goes down, < goes up)
- Added fallback planning for better exploration

**Status**: Agent runs but struggles to find stairs consistently (0% progression)

### 2. TreeQuest MCTS Integration
- Explored TreeQuest repository - a flexible MCTS library by SakanaAI
- Created `MCTSPlanningAgent` that integrates TreeQuest with VLLM
- Features:
  - Uses LLM for state evaluation
  - Gets action suggestions from LLM
  - Combines heuristic and LLM-based scoring
  
**Status**: Dependency issues with TreeQuest (missing pandas import in pymc_interface.py)

### 3. Simple MCTS Implementation
- Created `SimpleMCTSAgent` - a standalone MCTS implementation
- Features:
  - No external dependencies
  - UCB-based node selection
  - Random simulation rollouts
  - Configurable exploration parameters
  
**Status**: Working but needs optimization (0% progression)

## Key Findings

### What Works
1. **Simple Heuristic Agent**: Achieved 30.78% progression using minimal LLM calls
2. **Knowledge Server Integration**: Successfully queries game knowledge but needs better utilization
3. **JSON Prompting**: Reliable for 1B parameter models

### Challenges
1. **Stair Finding**: Agents struggle to systematically explore and find stairs
2. **LLM Response Quality**: 1B model sometimes fails to generate valid plans
3. **Exploration vs Exploitation**: MCTS needs better balance for NetHack's sparse rewards

## MCTS Algorithm Overview

Monte Carlo Tree Search (MCTS) builds a search tree by:
1. **Selection**: Traverse tree using UCB to balance exploration/exploitation
2. **Expansion**: Add new nodes for untried actions
3. **Simulation**: Random playout to estimate state value
4. **Backpropagation**: Update statistics back up the tree

For NetHack, MCTS could help by:
- Planning multiple moves ahead
- Evaluating long-term consequences
- Balancing exploration of new areas with exploitation of known good paths

## Next Steps

1. **Optimize Simple MCTS Agent**:
   - Add better state representation (include map analysis)
   - Implement smarter simulation policy (not just random)
   - Use LLM for critical decisions only

2. **Fix TreeQuest Integration**:
   - Install missing dependencies (pandas, pymc)
   - Or extract just the core MCTS logic

3. **Hybrid Approach**:
   - Use heuristics for common actions (like simple_heuristic agent)
   - Apply MCTS for strategic decisions (stairs, combat, items)

4. **Learn from Success**:
   - Analyze why simple_heuristic agent works
   - Incorporate its exploration strategy into MCTS

## Code Locations

- Planning-first agent: `/balrog/agents/planning_first_agent.py`
- MCTS with TreeQuest: `/balrog/agents/mcts_planning_agent.py`  
- Simple MCTS: `/balrog/agents/simple_mcts_agent.py`
- TreeQuest library: `/treequest/`
- Test scripts: `/test_stairs_usage.py`, `/test_mcts_focused.py`

## Performance Comparison

| Agent | Progression | Notes |
|-------|------------|-------|
| Simple Heuristic | 30.78% | Minimal LLM, good exploration |
| Planning First | 0% | Gets stuck despite knowledge |
| Simple MCTS | 0% | Needs better heuristics |
| Naive | Variable | Baseline comparison |