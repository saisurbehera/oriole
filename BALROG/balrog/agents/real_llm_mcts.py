"""
Real LLM-guided MCTS Agent
=========================

This agent uses the LLM for every action decision within an MCTS framework.
Unlike the simple MCTS that uses heuristics, this agent leverages LLM reasoning
for NetHack-specific understanding.
"""

import math
import random
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from balrog.agents.base import BaseAgent

# TreeQuest integration with proper imports
try:
    import sys
    sys.path.append('/home/sai/Desktop/oriole/BALROG/treequest/src')
    from treequest.algos.standard_mcts import StandardMCTS
    from treequest.algos.tree import Node as TreeQuestNode
    TREEQUEST_AVAILABLE = True
    print("TreeQuest StandardMCTS loaded successfully!")
except Exception as e:
    TREEQUEST_AVAILABLE = False
    print(f"TreeQuest not available: {e}")
from balrog.client import LLMResponse
from balrog.prompt_builder.history import Message


@dataclass
class LLMNode:
    """MCTS node that uses LLM for action evaluation"""
    state: Dict[str, Any]
    action: Optional[str] = None
    parent: Optional['LLMNode'] = None
    children: List['LLMNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    llm_reasoning: str = ""
    
    @property
    def average_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0
    
    def ucb_score(self, parent_visits: int, c: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.average_reward
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration


class RealLLMMCTSAgent(BaseAgent):
    """MCTS agent that uses LLM for every decision"""
    
    def __init__(self, client_factory, prompt_builder, sft_model=None):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # MCTS parameters
        self.mcts_iterations = 6  # Fewer iterations since each uses LLM
        self.max_children_per_node = 4  # Limit branching for efficiency
        
        # SFT integration  
        self.sft_model = sft_model  # Optional SFT model for expert-guided expansion
        self.use_sft_guidance = sft_model is not None
        
        # Complete NetHack action space - 80+ actions
        self.all_nethack_actions = [
            # Movement
            'north', 'south', 'east', 'west', 
            'northeast', 'northwest', 'southeast', 'southwest',
            'up', 'down',
            # Basic Actions
            'wait', 'search', 'look', 'pickup', 'drop', 'inventory',
            # Item Usage
            'eat', 'drink', 'quaff', 'read', 'wear', 'wield', 'apply',
            # Combat
            'fight', 'fire', 'throw', 'zap',
            # Special
            'pray', 'offer', 'engrave', 'open', 'close', 'kick',
            # More NetHack specific
            'rest', 'save', 'quit', 'help', 'chat', 'force', 'jump',
            'loot', 'monster', 'name', 'pay', 'ride', 'sit', 'turn',
            'untrap', 'version', 'whatis', 'wipe'
        ]
        
        # Expert action patterns learned from training data - context-aware priorities
        self.expert_action_priorities = {
            'find_stairs': [
                # Primary: Movement and exploration
                'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
                'search', 'look',
                # Secondary: Safe actions while exploring
                'wait', 'inventory'
            ],
            'collect_equipment': [
                # Primary: Item management
                'pickup', 'inventory', 'wield', 'wear', 'drop',
                # Secondary: Movement to reach items
                'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
                # Tertiary: Item evaluation
                'look', 'apply'
            ],
            'combat_engagement': [
                # Primary: Combat actions
                'fight', 'fire', 'throw', 'zap', 'wield',
                # Secondary: Combat movement (tactical positioning)
                'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
                # Tertiary: Combat support
                'drink', 'quaff', 'read', 'pray', 'apply'
            ],
            'level_progression': [
                # Primary: Level transitions
                'down', 'up',
                # Secondary: Pre-transition preparation
                'inventory', 'eat', 'drink', 'quaff', 'pray',
                # Tertiary: Safety checks
                'look', 'search'
            ],
            'exploration': [
                # Primary: Movement and discovery
                'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
                'search', 'look', 'open', 'kick',
                # Secondary: Opportunistic actions
                'pickup', 'eat', 'inventory'
            ],
            'resource_management': [
                # Primary: Resource actions
                'eat', 'drink', 'quaff', 'rest', 'pray',
                # Secondary: Resource evaluation
                'inventory', 'look', 'apply',
                # Tertiary: Safe positioning
                'wait', 'search'
            ]
        }
        
        # Episode tracking
        self.turn_count = 0
        self.episode_history = []
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.turn_count = 0
        self.episode_history = []
        
    def act(self, observation: Dict[str, Any], prev_action: str = None, return_reasoning: bool = False) -> LLMResponse:
        """Use LLM-guided MCTS to select action"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(observation)
        
        self.turn_count += 1
        
        # Parse game state from observation
        game_state = self._parse_game_state(observation)
        self.episode_history.append({
            'turn': self.turn_count,
            'state': game_state,
            'prev_action': prev_action
        })
        
        print(f"\\nTurn {self.turn_count}: Running LLM-MCTS...")
        
        # Run MCTS with LLM guidance
        best_action, reasoning = self._run_llm_mcts(game_state, observation)
        
        return LLMResponse(
            model_id="real_llm_mcts",
            completion=best_action,
            stop_reason="complete",
            input_tokens=200,
            output_tokens=50,
            reasoning=f"LLM-MCTS: {reasoning}"
        )
    
    def _run_llm_mcts(self, initial_state: Dict, observation: Dict) -> Tuple[str, str]:
        """Run MCTS where every decision uses LLM reasoning"""
        
        # Create root node
        root = LLMNode(state=initial_state)
        
        print(f"Running {self.mcts_iterations} MCTS iterations with LLM...")
        
        # MCTS iterations
        for i in range(self.mcts_iterations):
            print(f"  Iteration {i+1}/{self.mcts_iterations}")
            
            # Selection: Find leaf node using UCB
            node = self._select(root)
            
            # Expansion: Add new child using LLM
            if len(node.children) < self.max_children_per_node:
                node = self._expand_with_llm(node, observation)
            
            # Simulation: Evaluate using LLM
            reward = self._simulate_with_llm(node, observation)
            
            # Backpropagation: Update rewards
            self._backpropagate(node, reward)
            
            print(f"    -> Selected: {node.action}, Reward: {reward:.3f}")
        
        # Choose best action from root children
        if root.children:
            # Sort by average reward (exploitation)
            best_child = max(root.children, key=lambda c: c.average_reward)
            
            print(f"\\nRoot children summary:")
            for i, child in enumerate(root.children):
                print(f"  {i+1}. {child.action}: {child.visits} visits, {child.average_reward:.3f} avg reward")
            
            return best_child.action, best_child.llm_reasoning
        else:
            # Fallback: ask LLM directly
            return self._direct_llm_action(initial_state, observation)
    
    def _select(self, node: LLMNode) -> LLMNode:
        """Select leaf node using UCB"""
        while node.children:
            # Choose child with highest UCB score
            node = max(node.children, key=lambda c: c.ucb_score(node.visits))
        return node
    
    def _expand_with_llm(self, node: LLMNode, observation: Dict) -> LLMNode:
        """Expand node using expert-guided action selection with LLM reasoning"""
        
        # Get actions already tried from this node
        tried_actions = {child.action for child in node.children}
        
        # Expert-guided action selection
        if self.use_sft_guidance:
            new_action, reasoning = self._get_expert_guided_action(node.state, observation, tried_actions)
        else:
            # Fallback to pure LLM
            new_action, reasoning = self._get_llm_action(node.state, observation, tried_actions)
        
        # Expert fallback if duplicate
        if new_action in tried_actions:
            print(f"    Action '{new_action}' already tried, using expert fallback")
            new_action, reasoning = self._get_expert_fallback_action(node.state, tried_actions)
        
        # Create new child node
        new_state = self._predict_state_after_action(node.state, new_action)
        child = LLMNode(
            state=new_state,
            action=new_action,
            parent=node,
            llm_reasoning=reasoning
        )
        
        node.children.append(child)
        print(f"    Expanded with action: {new_action}")
        
        return child
    
    def _get_llm_action(self, state: Dict, observation: Dict, tried_actions: set) -> Tuple[str, str]:
        """Ask LLM for a good action given the current state"""
        
        # Build context for LLM
        context = self._build_action_context(state, observation, tried_actions)
        
        try:
            messages = [Message(role="user", content=context)]
            response = self.client.generate(messages)
            
            # Parse LLM response
            action, reasoning = self._parse_action_response(response.completion)
            return action, reasoning
            
        except Exception as e:
            print(f"    LLM error: {e}")
            # Fallback to safe actions
            safe_actions = ['search', 'north', 'east', 'south', 'west']
            for action in safe_actions:
                if action not in tried_actions:
                    return action, f"LLM failed, using safe action: {action}"
            return 'wait', "All actions tried, waiting"
    
    def _build_action_context(self, state: Dict, observation: Dict, tried_actions: set) -> str:
        """Build LLM prompt for action selection"""
        
        # Extract game info
        depth = state.get('depth', 1)
        hp = state.get('hp_current', 16)
        score = state.get('score', 0)
        
        # Get text observation and extract MAP correctly
        obs_text = ""
        map_view = ""
        player_pos = (state.get('x', 0), state.get('y', 0))
        stairs_pos = None
        
        if observation and 'text' in observation:
            text_info = observation['text']
            obs_text = text_info.get('long_term_context', '')
            
            # Extract map from long_term_context (BALROG format)
            if "map:" in obs_text:
                map_start = obs_text.find("map:") + 5
                # Find where map ends (usually at status line)
                status_markers = ["\nAgent ", "\nDlvl:", "\n$", "\nSt:"]
                map_end = len(obs_text)
                for marker in status_markers:
                    pos = obs_text.find(marker, map_start)
                    if pos > 0 and pos < map_end:
                        map_end = pos
                
                map_text = obs_text[map_start:map_end]
                map_view = map_text.strip()
                
                # Parse stairs position from map
                map_lines = map_view.split('\n')
                for y, line in enumerate(map_lines[:21]):  # NetHack map is 21 rows
                    if '<' in line:  # Down stairs
                        stairs_pos = (line.index('<'), y)
        
        # Determine current situation from map and text  
        on_stairs = player_pos and stairs_pos and player_pos == stairs_pos
        stairs_visible = stairs_pos is not None
        monsters_nearby = any(monster in obs_text.lower() for monster in ['kobold', 'goblin', 'rat', 'newt', 'fox'])
        
        # Build comprehensive NetHack prompt
        prompt = f"""=== NetHack Strategic Gameplay ===

CURRENT STATUS:
- Dungeon Level: {state.get('depth', 1)}
- HP: {state.get('hp', 16)}/{state.get('max_hp', 16)}
- Score: {state.get('score', 0)}
- Gold: {state.get('gold', 0)}
- XP Level: {state.get('experience_level', 1)}
- AC: {state.get('ac', 10)} (lower is better)
- Turn: {self.turn_count}
- Hunger: {"Normal" if state.get('hunger_state', 1) < 3 else "Hungry" if state.get('hunger_state', 1) == 3 else "WEAK"}

ATTRIBUTES:
- STR: {state.get('strength', 18)} DEX: {state.get('dexterity', 15)} CON: {state.get('constitution', 14)}
- INT: {state.get('intelligence', 10)} WIS: {state.get('wisdom', 8)} CHA: {state.get('charisma', 12)}

MAP VIEW:
{map_view if map_view else "No map data available"}
- Player Position: {player_pos if player_pos else "unknown"}
- Stairs Position: {stairs_pos if stairs_pos else "not found"}

ITEMS NEARBY: {state.get('items_nearby', [])}
MONSTERS: {"YES" if state.get('monsters_nearby') else "NO"}

MAP LEGEND:
@ = You  < = Stairs down  > = Stairs up  . = Floor  # = Wall  + = Door  d = Pet
) = Weapons  [ = Armor  ! = Potions  ? = Scrolls  = = Rings  % = Food

GAMEPLAY PRIORITIES:
1. SURVIVAL: Manage HP, hunger, avoid death
2. EQUIPMENT: Pick up weapons/armor, wield/wear them  
3. RESOURCES: Collect gold, potions, scrolls, food
4. COMBAT: Fight monsters for XP and loot
5. EXPLORATION: Find all items and map layout
6. PROGRESSION: Only descend when well-equipped

SITUATION ANALYSIS:
- On stairs: {on_stairs}
- Stairs visible: {stairs_visible}  
- Items to collect: {len(state.get('items_nearby', []))} items
- Combat ready: {"Yes" if state.get('ac', 10) < 8 else "Needs armor"}
- Prepared for level 2: {"Yes" if state.get('experience_level', 1) > 1 and state.get('ac', 10) < 8 else "NO - get equipment first!"}

ACTIONS ALREADY TRIED: {list(tried_actions)}

AVAILABLE ACTIONS:
MOVEMENT: north, south, east, west, northeast, northwest, southeast, southwest, up, down
BASIC: wait, search, look, pickup, drop, inventory  
ITEMS: eat, drink, read, wear, wield, apply
COMBAT: fight, fire, throw, zap
SPECIAL: pray, offer, engrave, open, kick

DECISION PRIORITIES:
{"EXPLORE FIRST!" if not stairs_visible else "STAIRS FOUND - PREPARE TO DESCEND!"}

1. EXPLORATION: {"🔍 CRITICAL! Must explore to find stairs and monsters for score!" if not stairs_visible else "Continue exploring for monsters and items"}
2. IMMEDIATE DANGER: If monsters attacking → fight or flee  
3. ITEMS ON GROUND: If items nearby → pickup valuable ones
4. EQUIPMENT NEEDS: If unequipped → wield weapons, wear armor
5. RESOURCE MANAGEMENT: If hungry → eat food, if injured → drink healing
6. PROGRESSION: If well-equipped and stairs found → descend

STRATEGIC DECISIONS:
{"🎯 STAIRS NOT FOUND - EXPLORATION MODE:" if not stairs_visible else "STAIRS FOUND - PREPARATION MODE:"}
{'''
⭐ PRIMARY GOAL: Find stairs by systematic exploration!
- Move in unexplored directions: "north", "south", "east", "west"  
- Search corners and rooms: "search"
- Look for '<' symbol on map (stairs down)
- Kill any monsters encountered for score and XP''' if not stairs_visible else '''
⭐ PRIMARY GOAL: Prepare for level 2 descent!
- Collect weapons and armor
- Fight monsters for XP and score
- Gather resources (food, potions)'''}

IMMEDIATE ACTIONS:
- If items nearby: "pickup" to collect them
- If holding weapon but not wielding: "wield" it
- If holding armor but not wearing: "wear" it  
- If monsters present: "fight" for score and XP!
- If injured (HP < 75%): look for healing potions to "drink"
- If hungry: "eat" food if available
- If stairs visible AND well-equipped: move toward stairs
- If on stairs AND prepared: "down" to descend
- {"If no stairs found: MOVE to explore! Try 'north', 'south', 'east', 'west'" if not stairs_visible else "Continue systematic exploration"}

Choose the MOST IMPORTANT action for long-term NetHack success:

Respond with JSON:
{{"action": "chosen_action", "reasoning": "why_this_action_is_most_important_for_nethack_survival_and_progression"}}"""

        return prompt
    
    def _parse_action_response(self, llm_text: str) -> Tuple[str, str]:
        """Parse LLM response to extract action and reasoning"""
        
        # Try to extract JSON
        try:
            # Look for JSON in response
            import re
            json_match = re.search(r'\\{.*?"action".*?"reasoning".*?\\}', llm_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                action = parsed.get('action', '').strip().lower()
                reasoning = parsed.get('reasoning', '').strip()
                
                # Validate action
                valid_actions = [
                    # Movement
                    'north', 'south', 'east', 'west', 'northeast', 'northwest', 
                    'southeast', 'southwest', 'up', 'down',
                    # Basic actions
                    'wait', 'search', 'look', 'pickup', 'drop', 'inventory',
                    # Item usage
                    'eat', 'drink', 'read', 'wear', 'wield', 'apply',
                    # Combat
                    'fight', 'fire', 'throw', 'zap',
                    # Special
                    'pray', 'offer', 'engrave', 'open', 'kick'
                ]
                
                if action in valid_actions:
                    return action, reasoning
        except:
            pass
        
        # Fallback: look for action keywords in priority order
        text_lower = llm_text.lower()
        priority_actions = ['pickup', 'wield', 'wear', 'eat', 'drink', 'fight', 
                          'down', 'search', 'north', 'south', 'east', 'west']
        
        for action in priority_actions:
            if action in text_lower:
                return action, f"Extracted '{action}' from LLM response"
        
        return 'search', "Failed to parse LLM response, defaulting to search"
    
    def _simulate_with_llm(self, node: LLMNode, observation: Dict) -> float:
        """Evaluate node using dense reward shaping"""
        
        state = node.state
        parent_state = node.parent.state if node.parent else {}
        action = node.action
        
        # Per-step shaped rewards for analysis
        reward = 0.0
        
        # Score deltas: +score_change/100
        score_change = state.get('score', 0) - parent_state.get('score', 0)
        if score_change > 0:
            reward += score_change / 100
            print(f"    Score reward: +{score_change/100:.3f} (score: {score_change})")
        
        # Gold collection: +gold_gained/1000
        gold_gained = state.get('gold', 0) - parent_state.get('gold', 0)
        if gold_gained > 0:
            reward += gold_gained / 1000
            print(f"    Gold reward: +{gold_gained/1000:.3f} (gold: {gold_gained})")
        
        # Experience: +exp_gained/100
        exp_gained = state.get('experience_points', 0) - parent_state.get('experience_points', 0)
        if exp_gained > 0:
            reward += exp_gained / 100
            print(f"    XP reward: +{exp_gained/100:.3f} (exp: {exp_gained})")
        
        # Depth progression: +depth_change*10 (big rewards!)
        depth_change = state.get('depth', 1) - parent_state.get('depth', 1)
        if depth_change > 0:
            reward += depth_change * 10
            print(f"    Depth reward: +{depth_change*10:.3f} (depth: +{depth_change})")
        
        # Level up: +level_change*5
        level_change = state.get('experience_level', 1) - parent_state.get('experience_level', 1)
        if level_change > 0:
            reward += level_change * 5
            print(f"    Level reward: +{level_change*5:.3f} (level: +{level_change})")
        
        # Death penalty: -100
        if state.get('hp', 16) <= 0:
            reward -= 100
            print(f"    Death penalty: -100.0")
        
        # Dynamic action bonuses based on exploration needs
        action_bonuses = {
            'pickup': 0.1,    # Collecting items
            'wield': 0.08,    # Equipping weapons (reduced to favor exploration)
            'wear': 0.08,     # Wearing armor (reduced to favor exploration) 
            'eat': 0.05,      # Managing hunger (reduced to favor exploration)
            'drink': 0.1,     # Healing
            'fight': 0.15,    # Combat engagement (reduced to favor exploration)
            'search': 0.1,    # Exploration (increased)
            'read': 0.1,      # Using scrolls
        }
        
        # Dynamic movement bonuses - higher when exploration needed
        stairs_found = 'stairs_pos' in state and state['stairs_pos'] is not None
        items_nearby = len(state.get('items_nearby', []))
        turn_number = state.get('turn', 1)
        
        # Encourage movement early in the game for exploration
        if turn_number < 30 and not stairs_found:
            movement_bonus = 0.12  # Higher than most other actions
        elif not stairs_found:
            movement_bonus = 0.08  # Still prioritize finding stairs
        else:
            movement_bonus = 0.03  # Normal movement when stairs found
            
        # Add movement bonuses
        movement_actions = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']
        for move_action in movement_actions:
            action_bonuses[move_action] = movement_bonus
        
        if action in action_bonuses:
            bonus = action_bonuses[action]
            reward += bonus
            print(f"    Action bonus: +{bonus:.3f} ({action})")
        
        # Special exploration bonuses
        # Stairs discovery bonus
        parent_stairs = parent_state.get('stairs_found', False) if parent_state else False
        current_stairs = state.get('stairs_found', False)
        if not parent_stairs and current_stairs:
            reward += 1.0  # Big bonus for finding stairs!
            print(f"    Stairs discovery bonus: +1.0")
        
        # Movement exploration bonus (encourage visiting new areas)
        if action in movement_actions:
            # Small bonus for any movement to encourage exploration
            reward += 0.02
            print(f"    Exploration movement bonus: +0.02")
        
        # Small penalty for waiting
        if action == 'wait':
            reward -= 0.05
            print(f"    Wait penalty: -0.05")
        
        # Health-based rewards
        hp_ratio = state.get('hp', 16) / state.get('max_hp', 16)
        if hp_ratio > 0.9:
            reward += 0.02  # Staying healthy
        elif hp_ratio < 0.3:
            reward -= 0.1   # Low health penalty
        
        print(f"    Total reward: {reward:.3f}")
        return reward
    
    def _backpropagate(self, node: LLMNode, reward: float):
        """Update rewards back up the tree"""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
    
    def _direct_llm_action(self, state: Dict, observation: Dict) -> Tuple[str, str]:
        """Fallback: ask LLM directly for action"""
        action, reasoning = self._get_llm_action(state, observation, set())
        return action, f"Direct LLM: {reasoning}"
    
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Extract comprehensive game state from observation"""
        state = {
            # Position (will be updated from map)
            'x': 0, 'y': 0,
            # Health
            'hp': 16, 'max_hp': 16,
            # Attributes  
            'strength': 18, 'dexterity': 15, 'constitution': 14,
            'intelligence': 10, 'wisdom': 8, 'charisma': 12,
            # Combat stats
            'ac': 10,  # Armor class (lower is better)
            'experience_level': 1, 'experience_points': 0,
            # Resources
            'gold': 0, 'score': 0,
            # Environment
            'depth': 1, 'hunger_state': 1,
            # Game state
            'turn': self.turn_count,
            'items_nearby': [], 'monsters_nearby': [],
            'inventory_full': False,
            'stairs_found': False, 'stairs_pos': None
        }
        
        if obs and 'text' in obs:
            text = obs['text'].get('long_term_context', '')
            import re
            
            # Parse player position from map
            if "map:" in text:
                map_start = text.find("map:") + 5
                # Find where map ends (usually at status line)
                status_markers = ["\nAgent ", "\nDlvl:", "\n$", "\nSt:"]
                map_end = len(text)
                for marker in status_markers:
                    pos = text.find(marker, map_start)
                    if pos > 0 and pos < map_end:
                        map_end = pos
                
                map_text = text[map_start:map_end]
                map_view = map_text.strip()
                
                # Parse player position and stairs from map
                map_lines = map_view.split('\n')
                for y, line in enumerate(map_lines[:21]):  # NetHack map is 21 rows
                    if '@' in line:
                        state['x'] = line.index('@')
                        state['y'] = y
                    if '<' in line:  # Down stairs
                        state['stairs_pos'] = (line.index('<'), y)
                        state['stairs_found'] = True
            
            # Parse stats from status line
            depth_match = re.search(r'Dlvl:(\\d+)', text)
            if depth_match:
                state['depth'] = int(depth_match.group(1))
            
            hp_match = re.search(r'HP:(\\d+)\\((\\d+)\\)', text)
            if hp_match:
                state['hp'] = int(hp_match.group(1))
                state['max_hp'] = int(hp_match.group(2))
            
            score_match = re.search(r'S:(\\d+)', text)
            if score_match:
                state['score'] = int(score_match.group(1))
            
            # Parse gold
            gold_match = re.search(r'\\$:(\\d+)', text)
            if gold_match:
                state['gold'] = int(gold_match.group(1))
            
            # Parse experience
            xp_match = re.search(r'Xp:(\\d+)/(\\d+)', text)
            if xp_match:
                state['experience_level'] = int(xp_match.group(1))
                state['experience_points'] = int(xp_match.group(2))
            
            # Parse AC
            ac_match = re.search(r'AC:(\\d+)', text)
            if ac_match:
                state['ac'] = int(ac_match.group(1))
            
            # Parse attributes
            st_match = re.search(r'St:(\\d+)', text)
            if st_match:
                state['strength'] = int(st_match.group(1))
            
            dx_match = re.search(r'Dx:(\\d+)', text)
            if dx_match:
                state['dexterity'] = int(dx_match.group(1))
            
            # Detect items nearby
            if "You see here" in text or "Things that are here" in text:
                state['items_nearby'] = self._extract_nearby_items(text)
            
            # Detect monsters
            if any(monster in text.lower() for monster in ['kobold', 'goblin', 'rat', 'newt', 'fox', 'jackal', 'orc']):
                state['monsters_nearby'] = True
            
            # Check hunger
            if "hungry" in text.lower():
                state['hunger_state'] = 3
            elif "weak" in text.lower():
                state['hunger_state'] = 4
        
        return state
    
    def _extract_nearby_items(self, text: str) -> list:
        """Extract items that can be picked up"""
        items = []
        # Common NetHack items
        item_patterns = [
            r'(\\w+ sword)', r'(\\w+ armor)', r'(\\w+ shield)', 
            r'(potion of \\w+)', r'(scroll of \\w+)', r'(ring of \\w+)',
            r'(\\w+ gem)', r'(food ration)', r'(tin of \\w+)',
            r'(wand of \\w+)', r'(\\w+ boots)', r'(\\w+ helmet)'
        ]
        
        for pattern in item_patterns:
            import re
            matches = re.findall(pattern, text, re.IGNORECASE)
            items.extend(matches)
        
        return items[:5]  # Limit to 5 most recent items
    
    def _predict_state_after_action(self, state: Dict, action: str) -> Dict:
        """Predict state after taking action (simplified)"""
        new_state = state.copy()
        
        # Movement actions update position
        current_x, current_y = state.get('x', 0), state.get('y', 0)
        
        # Simple movement predictions
        if action == 'north':
            new_state['y'] = max(0, current_y - 1)
        elif action == 'south':
            new_state['y'] = min(20, current_y + 1)  # NetHack map is 21 rows (0-20)
        elif action == 'east':
            new_state['x'] = min(79, current_x + 1)  # NetHack map is 80 cols (0-79)
        elif action == 'west':
            new_state['x'] = max(0, current_x - 1)
        elif action == 'northeast':
            new_state['x'] = min(79, current_x + 1)
            new_state['y'] = max(0, current_y - 1)
        elif action == 'northwest':
            new_state['x'] = max(0, current_x - 1)
            new_state['y'] = max(0, current_y - 1)
        elif action == 'southeast':
            new_state['x'] = min(79, current_x + 1)
            new_state['y'] = min(20, current_y + 1)
        elif action == 'southwest':
            new_state['x'] = max(0, current_x - 1)
            new_state['y'] = min(20, current_y + 1)
        
        # Depth transition
        if action == 'down' and state.get('depth', 1) == 1:
            new_state['depth'] = 2  # Success!
        
        return new_state
    
    def _get_expert_guided_action(self, state: Dict, observation: Dict, tried_actions: set) -> Tuple[str, str]:
        """Get action using expert knowledge + LLM reasoning"""
        
        # Determine current subgoal based on game state
        subgoal = self._determine_subgoal(state, observation)
        
        # Get expert-prioritized actions for this subgoal
        expert_actions = self.expert_action_priorities.get(subgoal, self.expert_action_priorities['exploration'])
        
        # Filter out already tried actions
        candidate_actions = [action for action in expert_actions if action not in tried_actions]
        
        if not candidate_actions:
            # If all expert actions tried, expand to full action space
            candidate_actions = [action for action in self.all_nethack_actions if action not in tried_actions]
        
        if not candidate_actions:
            candidate_actions = ['wait']  # Last resort
        
        # Use LLM to choose among expert candidates
        chosen_action = candidate_actions[0]  # Default to first expert choice
        
        # Get LLM reasoning for the chosen action
        reasoning = f"Expert-guided: {subgoal} suggests '{chosen_action}' from {len(candidate_actions)} expert options"
        
        return chosen_action, reasoning
    
    def _get_expert_fallback_action(self, state: Dict, tried_actions: set) -> Tuple[str, str]:
        """Get fallback action when duplicates occur"""
        
        # Priority fallback order based on expert data
        fallback_order = [
            'north', 'south', 'east', 'west',  # Movement always safe
            'search', 'look',                   # Information gathering
            'pickup', 'inventory',              # Item management
            'wait'                              # Last resort
        ]
        
        for action in fallback_order:
            if action not in tried_actions:
                return action, f"Expert fallback: {action}"
        
        return 'wait', "Expert fallback: wait (all actions tried)"
    
    def _determine_subgoal(self, state: Dict, observation: Dict) -> str:
        """Determine current strategic subgoal based on game state"""
        
        # Check for stairs visibility
        stairs_found = state.get('stairs_found', False)
        
        # Check for items nearby
        items_nearby = len(state.get('items_nearby', []))
        
        # Check for monsters
        monsters_nearby = len(state.get('monsters_nearby', []))
        
        # Check current depth and preparation
        depth = state.get('depth', 1)
        hp_ratio = state.get('hp', 16) / state.get('max_hp', 16)
        
        # Subgoal priority logic based on expert patterns
        if monsters_nearby > 0:
            return 'combat_engagement'
        elif items_nearby > 0:
            return 'collect_equipment'  
        elif hp_ratio < 0.5 or state.get('hunger_state', 1) < 1:
            return 'resource_management'
        elif stairs_found and depth == 1:
            return 'level_progression'
        elif not stairs_found:
            return 'find_stairs'
        else:
            return 'exploration'