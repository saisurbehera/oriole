# SFT + Tree-MCTS Integration Strategy

## Current Problem
- **Random MCTS**: Current MCTS does random action exploration without strategic guidance
- **No Expert Knowledge**: Tree search doesn't leverage expert strategies from SFT training
- **Inefficient Search**: Wastes computation on obviously bad moves

## Proposed Solution: SFT-Guided Tree-MCTS

### 1. SFT Model as MCTS Policy Network
**Replace random action sampling with SFT-trained policy**

```python
class SFTGuidedMCTS:
    def __init__(self, sft_model, tree_search_depth=6):
        self.sft_model = sft_model  # Trained on expert data
        self.tree_search_depth = tree_search_depth
        
    def expand_node(self, node, game_state):
        # Get SFT policy distribution over actions
        action_probs = self.sft_model.get_action_distribution(game_state)
        
        # Sample top-k actions for tree expansion
        top_actions = self.sample_top_k_actions(action_probs, k=5)
        
        # Expand only promising actions, not random ones
        for action in top_actions:
            child_node = self.create_child_node(node, action)
            node.add_child(child_node)
```

### 2. Multi-Level Decision Making

**Stage 1: SFT Strategic Planning**
- Use SFT to identify high-level subgoals (find_stairs, combat, etc.)
- Get expert-like action preferences for current situation

**Stage 2: Tree-MCTS Tactical Search** 
- Use tree search to look ahead within the SFT-suggested actions
- Evaluate short-term consequences of expert-recommended moves
- Handle dynamic situations that require multi-step planning

### 3. Integration Architecture

```python
class ExpertGuidedAgent:
    def __init__(self, sft_model, mcts_tree):
        self.sft_model = sft_model      # Expert strategy 
        self.mcts_tree = mcts_tree      # Tactical lookahead
        
    def act(self, observation):
        # Step 1: SFT strategic assessment
        subgoal = self.sft_model.determine_subgoal(observation)
        candidate_actions = self.sft_model.get_expert_actions(observation, subgoal)
        
        # Step 2: MCTS tactical evaluation
        if len(candidate_actions) > 1:
            # Use tree search to pick best among expert suggestions
            best_action = self.mcts_tree.search(observation, candidate_actions)
        else:
            # Trust expert when only one good choice
            best_action = candidate_actions[0]
            
        return best_action
```

### 4. Training Strategy

**Phase 1: SFT Pre-training**
- Train policy network on expert trajectories
- Learn strategic patterns: exploration, combat, equipment management
- Output: Model that can suggest expert-like actions

**Phase 2: SFT + MCTS Integration** 
- Use SFT model to guide MCTS node expansion
- Tree search evaluates only expert-suggested actions
- Combines expert intuition with tactical lookahead

**Phase 3: Self-Play Improvement (Future)**
- Run SFT+MCTS agent to generate new trajectories  
- Use successful runs to improve both SFT and MCTS components
- Iterative improvement through expert iteration

### 5. Technical Implementation

**SFT Model Interface:**
```python
class NetHackSFTModel:
    def get_action_distribution(self, observation: str) -> Dict[str, float]:
        """Return probability distribution over actions"""
        
    def determine_subgoal(self, observation: str) -> str:
        """Identify current strategic objective"""
        
    def get_top_k_actions(self, observation: str, k: int = 5) -> List[str]:
        """Get k most likely expert actions"""
```

**MCTS Integration:**
```python  
class SFTGuidedTreeMCTS:
    def __init__(self, sft_model, exploration_constant=1.4):
        self.sft_model = sft_model
        self.exploration_constant = exploration_constant
        
    def selection(self, node):
        # UCB1 with SFT prior probabilities
        # Prioritize nodes with high SFT probability
        
    def expansion(self, node, game_state):
        # Only expand actions suggested by SFT model
        # Don't waste time on obviously bad moves
        
    def simulation(self, node, game_state):
        # Use SFT model for rollout policy instead of random
        # More accurate value estimates
```

### 6. Expected Benefits

**Better Action Selection:**
- MCTS explores expert-recommended actions, not random ones
- Higher probability of finding good moves quickly

**Strategic Consistency:**
- SFT provides long-term strategic direction
- MCTS handles short-term tactical decisions

**Computational Efficiency:**
- Focus tree search on promising actions only
- Reduce wasted computation on bad moves

**Improved Performance:**
- Combine expert knowledge with planning capability
- Better than pure SFT (handles novel situations)
- Better than pure MCTS (guided by expert intuition)

### 7. Implementation Priority

1. **First**: Train SFT model on expert data (current task)
2. **Second**: Integrate SFT into existing MCTS agent 
3. **Third**: Test combined SFT+MCTS performance
4. **Fourth**: Iterate and improve based on results

This approach leverages both expert demonstrations (SFT) and planning capability (MCTS) for optimal NetHack performance.