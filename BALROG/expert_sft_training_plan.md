# Expert SFT Training Plan for NetHack

## Overview
Two-stage approach: Expert Iteration SFT → RL fine-tuning (future)

## Stage 1: Supervised Fine-Tuning (SFT) with Expert Data

### Data Analysis
- **Source**: `/home/sai/Desktop/oriole/BALROG/nld-aa-taster/nle_data/`
- **Format**: 20 expert sessions, each with 100 episodes 
- **Content**: TTY recordings (.ttyrec3.bz2) + metadata (.xlogfile)
- **Quality**: Points 1539-53678, depths 1-7, diverse strategies

### Training Pipeline Components

#### 1. Data Processing
- Parse TTY recordings to extract (observation, action) pairs
- Convert ANSI escape sequences to structured game state
- Filter episodes by quality (focus on higher-scoring runs)
- Create subgoal annotations for strategic training

#### 2. Subgoal Modification Strategy
Break down expert episodes into strategic subgoals:
- **Early Game** (turns 1-1000): Find stairs, collect basic equipment
- **Mid Game** (turns 1000-10000): Level progression, combat mastery
- **Late Game** (turns 10000+): Advanced strategy, dungeon mastery

#### 3. Training Data Format
For each game step, create training examples:
```python
{
    "observation": "parsed_game_state_text",
    "subgoal": "current_strategic_objective", 
    "action": "expert_action_taken",
    "reasoning": "why_expert_chose_this_action"
}
```

#### 4. Model Architecture
- Base: Current `real_llm_mcts` agent structure
- Input: Game state + subgoal context
- Output: Action + reasoning (JSON format)
- Training: Standard SFT loss on expert actions

### Implementation Steps

#### Step 1: Data Extraction
1. Parse TTY files to extract game states and actions
2. Map raw NetHack data to BALROG observation format
3. Quality filter: Focus on episodes with points > 5000
4. Create train/val split (80/20)

#### Step 2: Subgoal Annotation
1. Automatically detect subgoals from game state:
   - "find_stairs" (when stairs not visible)
   - "collect_equipment" (when inventory sparse)
   - "level_progression" (when on stairs)
   - "combat_engagement" (when monsters present)
   - "resource_management" (when hungry/injured)

#### Step 3: Training Infrastructure
1. Convert expert data to training format
2. Create SFT training script with proper hyperparameters
3. Implement evaluation metrics (action accuracy, trajectory similarity)
4. Set up training monitoring and checkpointing

#### Step 4: Model Training
1. Fine-tune on expert trajectories with subgoal conditioning
2. Validate on held-out expert episodes
3. Compare trained model vs baseline on evaluation metrics

### Expected Outcomes
- Agent learns expert-level early game exploration
- Improved stairs discovery and level progression
- Better equipment management and combat decisions
- Higher scores and deeper dungeon exploration

## Implementation Timeline
1. **Data processing** (today): Parse expert trajectories
2. **Subgoal design** (today): Create strategic annotations  
3. **Training setup** (today): Implement SFT pipeline
4. **Model training** (next): Run SFT experiments
5. **Evaluation** (next): Test trained agent performance

## Future: Stage 2 - RL Fine-tuning
After SFT baseline established:
- Use SFT model as initialization for RL training
- Design reward function based on NetHack scoring
- Implement PPO/other RL algorithm for policy improvement
- Compare SFT+RL vs SFT-only performance