"""
LLM-guided MCTS agent inspired by AB-MCTS but without TreeQuest dependency
"""

import math
import random
import json
import traceback
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

from .base import BaseAgent
from balrog.client import LLMResponse


@dataclass
class MCTSNode:
    """MCTS tree node with LLM-generated state"""
    state: Dict[str, Any]
    action: str
    reasoning: str
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    total_reward: float = 0.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def average_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0
    
    def ucb_score(self, parent_visits: int, c: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.average_reward
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration


class LLMMCTSAgent(BaseAgent):
    """LLM-guided MCTS agent for NetHack"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # MCTS tree
        self.root = None
        self.current_observation = None
        self.turn_count = 0
        
        # Action types for diverse generation
        self.action_types = ["exploration", "navigation", "combat", "survival"]
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.root = None
        self.current_observation = None
        self.turn_count = 0
        
    def act(self, observation: Dict[str, Any], prev_action: str = None, return_reasoning: bool = False) -> LLMResponse:
        """Select action using LLM-guided MCTS"""
        self.turn_count += 1
        self.current_observation = observation
        
        # Initialize root if first turn
        if self.root is None:
            self.root = MCTSNode(
                state=self._parse_game_state(observation),
                action="start",
                reasoning="Initial state"
            )
        
        # Run MCTS iterations
        search_budget = 8  # Number of MCTS simulations
        print(f"DEBUG: Starting MCTS with {search_budget} iterations...")
        for i in range(search_budget):
            print(f"DEBUG: MCTS iteration {i+1}/{search_budget}")
            self._mcts_iteration()
        
        print(f"DEBUG: MCTS complete. Root has {len(self.root.children)} children")
        print(f"DEBUG: Root children actions: {[c.action for c in self.root.children]}")
        print(f"DEBUG: Root children visits: {[c.visits for c in self.root.children]}")
        print(f"DEBUG: Root children rewards: {[c.average_reward for c in self.root.children]}")
        
        # Select best action from root children
        if self.root.children:
            best_child = max(self.root.children, key=lambda child: child.average_reward)
            action = best_child.action
            reasoning = f"LLM-MCTS (visits: {best_child.visits}, reward: {best_child.average_reward:.3f}): {best_child.reasoning}"
            print(f"DEBUG: Selected action: {action} from {len(self.root.children)} children")
            
            # Debug all children scores
            for i, child in enumerate(self.root.children):
                print(f"DEBUG: Child {i}: action={child.action}, visits={child.visits}, avg_reward={child.average_reward:.3f}")
        else:
            # If no children, force some exploration actions instead of waiting
            exploration_actions = ["north", "south", "east", "west", "search", "look"]
            action = random.choice(exploration_actions)
            reasoning = "LLM-MCTS: no children, forced exploration"
            print(f"DEBUG: No children! Using fallback exploration action: {action}")
        
        return LLMResponse(
            model_id="llm_mcts",
            completion=action,
            stop_reason="complete",
            input_tokens=150,
            output_tokens=30,
            reasoning=reasoning
        )
    
    def _mcts_iteration(self):
        """Single MCTS iteration: Selection -> Expansion -> Simulation -> Backpropagation"""
        # Start MCTS selection from root
        node = self._select(self.root)
        print(f"DEBUG: MCTS selected node with action: {getattr(node, 'action', 'root')}")
        
        # Expansion: if leaf node is not terminal, expand it
        if not self._is_terminal(node) and not self._is_fully_expanded(node):
            node = self._expand(node)
            print(f"DEBUG: Expanded node, now has {len(node.parent.children) if node.parent else 'N/A'} siblings")
        
        # Simulation: get reward for this state
        reward = self._simulate(node)
        print(f"DEBUG: Simulated reward: {reward:.3f} for action: {node.action}")
        
        # Backpropagation: update path to root
        self._backpropagate(node, reward)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node using UCB until reaching leaf"""
        # Special case: if root has < 4 children, always return root for expansion
        # This ensures we get diverse root children before going deep
        if node == self.root and len(node.children) < 4:
            return node
            
        while node.children and not self._is_terminal(node):
            # Use UCB to select best child
            best_child = max(node.children, key=lambda child: child.ucb_score(node.visits))
            node = best_child
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by generating new child using LLM"""
        # Get list of actions not yet tried
        tried_actions = {child.action for child in node.children}
        available_actions = [
            'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
            'down', 'up', 'wait', 'search', 'open', 'kick', 'fight', 'look'
        ]
        untried_actions = [a for a in available_actions if a not in tried_actions]
        
        if not untried_actions:
            # Fully expanded, return existing child
            return random.choice(node.children) if node.children else node
        
        # Choose a specific untried action
        target_action = random.choice(untried_actions)
        
        # Choose action type for diverse exploration
        action_type = random.choice(self.action_types)
        
        # Generate new action and reasoning using LLM, biased toward target_action
        new_action, new_reasoning, new_state = self._generate_llm_action(
            node.state, action_type, target_action
        )
        
        # CRITICAL FIX: Ensure we actually use the target action to guarantee diversity
        # If LLM gives us a different action that we've already tried, force the target
        if new_action in tried_actions:
            print(f"DEBUG: LLM returned already-tried action '{new_action}', forcing target '{target_action}'")
            new_action = target_action
            new_reasoning = f"Forced {target_action} for exploration diversity"
            new_state = self._simulate_action_effect(node.state, target_action)
        
        # Double-check we're not adding duplicates
        if new_action in tried_actions:
            print(f"DEBUG: CRITICAL BUG - about to add duplicate action '{new_action}'!")
            print(f"DEBUG: Tried actions: {tried_actions}")
            print(f"DEBUG: Forcing unique action")
            # Force a completely unique action
            for action in available_actions:
                if action not in tried_actions:
                    new_action = action
                    new_reasoning = f"Forced unique action {action}"
                    new_state = self._simulate_action_effect(node.state, action)
                    break
        
        # Create new child node
        child = MCTSNode(
            state=new_state,
            action=new_action,
            reasoning=new_reasoning,
            parent=node
        )
        
        node.children.append(child)
        
        # Verify children are actually unique
        child_actions = [c.action for c in node.children]
        unique_actions = set(child_actions)
        if len(child_actions) != len(unique_actions):
            print(f"DEBUG: CRITICAL ERROR - Duplicate children detected: {child_actions}")
        
        print(f"DEBUG: Expanded node, now has {len(node.children)} children with actions: {child_actions}")
        return child
    
    def _is_fully_expanded(self, node: MCTSNode) -> bool:
        """Check if node has tried all reasonable actions"""
        tried_actions = {child.action for child in node.children}
        core_actions = ['north', 'south', 'east', 'west', 'search', 'wait']
        return len(tried_actions.intersection(core_actions)) >= 4  # At least 4 core actions tried
    
    def _simulate(self, node: MCTSNode) -> float:
        """Calculate reward for the current state"""
        return self._calculate_state_reward(node.state, node.action)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Update statistics for all nodes in path to root"""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
    
    def _generate_llm_action(self, parent_state: Dict, action_type: str, target_action: str = None) -> Tuple[str, str, Dict]:
        """Generate action using LLM with specific focus"""
        try:
            # Build context prompt
            context = self._build_llm_prompt(parent_state, action_type, target_action)
            
            # Query LLM
            from balrog.prompt_builder.history import Message
            messages = [Message(role="user", content=context)]
            response = self.client.generate(messages)
            
            # Debug: Print what LLM returned
            print(f"DEBUG: LLM response: {response.completion[:100]}...")
            
            # Parse response
            action, reasoning = self._parse_llm_response(response.completion)
            
            print(f"DEBUG: Parsed action: {action}, reasoning: {reasoning[:50]}...")
            
            # Update state based on action
            new_state = self._simulate_action_effect(parent_state, action)
            
            return action, reasoning, new_state
            
        except Exception as e:
            # Debug: Print the actual error
            print(f"DEBUG: LLM generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback with diverse actions instead of always "wait"
            fallback_actions = ["north", "south", "east", "west", "search"]
            fallback_action = random.choice(fallback_actions)
            return fallback_action, f"LLM error: {e}", parent_state.copy()
    
    def _build_llm_prompt(self, state: Dict, action_type: str, target_action: str = None) -> str:
        """Build LLM prompt based on state and action type"""
        # Get current observation for detailed context
        obs_text = ""
        if self.current_observation:
            text = self.current_observation.get("text", {})
            obs_text = text.get("long_term_context", "")
        
        # Extract map and position information
        map_context = self._extract_map_context(obs_text)
        tactical_context = self._extract_tactical_context(state, obs_text)
        
        prompt_parts = [
            f"=== NetHack Tactical Analysis ===",
            f"Focus: {action_type}",
            "",
            f"Current Status:",
            f"- Depth: Level {state.get('depth', 1)}",
            f"- HP: {state.get('hp_current', 16)}/{state.get('hp_max', 16)} ({'healthy' if state.get('hp_current', 16) > 12 else 'injured'})",
            f"- Score: {state.get('score', 0)}",
            "",
            f"Spatial Situation:",
        ]
        
        # Add map context
        prompt_parts.extend(map_context)
        
        # Add tactical context
        prompt_parts.extend([
            "",
            "Tactical Assessment:",
        ])
        prompt_parts.extend(tactical_context)
        
        # Add action guidance based on situation
        if state.get('on_stairs'):
            prompt_parts.extend([
                "",
                "🎯 CRITICAL: You are ON STAIRS! Use 'down' action NOW to descend to level 2!",
                "This is the primary objective - reaching deeper levels."
            ])
        elif state.get('stairs_visible'):
            prompt_parts.extend([
                "",
                "🎯 STAIRS DETECTED: Navigate toward stairs immediately.",
                "Priority: Reach stairs and use 'down' action."
            ])
        elif state.get('monsters_nearby'):
            prompt_parts.extend([
                "",
                "⚔️ COMBAT SITUATION: Monster detected.",
                "Options: Fight with 'fight', evade with movement, or use tactical positioning."
            ])
        else:
            prompt_parts.extend([
                "",
                "🔍 EXPLORATION MODE: Search for stairs to level 2.",
                "Strategy: Systematic exploration of unexplored areas."
            ])
        
        # Add specific guidance for target action if provided
        if target_action:
            prompt_parts.extend([
                "",
                f"Consider action: {target_action}",
                f"Evaluate: Would '{target_action}' be effective in this situation?"
            ])
        
        prompt_parts.extend([
            "",
            "Available actions: north, south, east, west, northeast, northwest, southeast, southwest, down, wait, search, open, kick, fight",
            "",
            "Choose the MOST TACTICAL action and explain your reasoning:",
            "",
            "Respond with JSON:",
            '{"action": "chosen_action", "reasoning": "tactical_analysis_of_why_this_action_is_optimal"}'
        ])
        
        return "\\n".join(prompt_parts)
    
    def _parse_llm_response(self, llm_text: str) -> Tuple[str, str]:
        """Parse LLM response to extract action and reasoning"""
        # Clean up the text first
        llm_text = llm_text.strip()
        
        # Try multiple JSON extraction strategies
        action, reasoning = self._extract_json_robust(llm_text)
        
        if action != "wait" or reasoning != "Failed to parse LLM response":
            return action, reasoning
            
        # Fallback: extract action from text patterns
        return self._fallback_action_extraction(llm_text)
    
    def _extract_json_robust(self, text: str) -> Tuple[str, str]:
        """Robust JSON extraction with multiple strategies"""
        valid_actions = [
            'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
            'down', 'up', 'wait', 'search', 'open', 'kick', 'fight', 'look'
        ]
        
        # Strategy 1: Find complete JSON objects
        import re
        json_pattern = r'\{[^{}]*"action"[^{}]*"reasoning"[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                action = parsed.get('action', '').strip().lower()
                reasoning = parsed.get('reasoning', '').strip()
                
                if action in valid_actions:
                    return action, reasoning
            except:
                continue
        
        # Strategy 2: Extract action and reasoning separately
        action_pattern = r'"action":\s*"([^"]+)"'
        reasoning_pattern = r'"reasoning":\s*"([^"]+)"'
        
        action_match = re.search(action_pattern, text, re.IGNORECASE)
        reasoning_match = re.search(reasoning_pattern, text, re.IGNORECASE)
        
        if action_match:
            action = action_match.group(1).strip().lower()
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "LLM reasoning extracted"
            
            if action in valid_actions:
                return action, reasoning
        
        # Strategy 3: Look for action names directly in quotes
        for action in valid_actions:
            if f'"{action}"' in text.lower() or f"'{action}'" in text.lower():
                return action, "Action extracted from text"
        
        return "wait", "Failed to parse LLM response"
    
    def _fallback_action_extraction(self, text: str) -> Tuple[str, str]:
        """Fallback action extraction from text patterns"""
        text_lower = text.lower()
        
        # Look for movement words
        if 'north' in text_lower: return 'north', 'Extracted from text'
        if 'south' in text_lower: return 'south', 'Extracted from text'
        if 'east' in text_lower: return 'east', 'Extracted from text'
        if 'west' in text_lower: return 'west', 'Extracted from text'
        if 'search' in text_lower: return 'search', 'Extracted from text'
        if 'fight' in text_lower: return 'fight', 'Extracted from text'
        if 'down' in text_lower: return 'down', 'Extracted from text'
        
        return "wait", "No action found in text"
    
    def _simulate_action_effect(self, state: Dict, action: str) -> Dict:
        """Simulate effect of action on game state (simplified)"""
        new_state = state.copy()
        
        # Simple state transitions based on action
        if action == "down" and state.get('on_stairs'):
            new_state['depth'] = new_state.get('depth', 1) + 1
            new_state['on_stairs'] = False
        elif action in ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']:
            # Movement might find stairs or encounter monsters
            if random.random() < 0.1:  # 10% chance to find stairs
                new_state['stairs_visible'] = True
            if random.random() < 0.05:  # 5% chance for monster encounter
                new_state['monsters_nearby'] = True
        
        return new_state
    
    def _calculate_state_reward(self, state: Dict, action: str) -> float:
        """Calculate reward for state-action pair with action diversity bonus"""
        reward = 0.5  # Base reward
        
        # Major rewards for game progress
        if state.get('depth', 1) > 1:
            reward += 0.4 * state['depth']  # Big bonus for deeper levels
        
        # Reward for good action selection
        if state.get('on_stairs') and action == "down":
            reward += 0.3  # Excellent action
        elif state.get('stairs_visible') and action in ['north', 'south', 'east', 'west']:
            reward += 0.2  # Good movement toward stairs
        elif action in ['search', 'north', 'south', 'east', 'west'] and not state.get('stairs_visible'):
            reward += 0.1  # Exploration is good
        
        # CRITICAL FIX: Add random noise to break ties and force exploration
        # This prevents MCTS from getting stuck on one action
        import random
        exploration_noise = random.uniform(-0.05, 0.05)  # Small random component
        reward += exploration_noise
        
        # Additional differentiation: different movement actions get slightly different rewards
        action_variety_bonus = {
            'north': 0.01, 'south': 0.02, 'east': 0.03, 'west': 0.04,
            'northeast': 0.05, 'northwest': 0.06, 'southeast': 0.07, 'southwest': 0.08,
            'search': 0.09, 'look': 0.10
        }
        reward += action_variety_bonus.get(action, 0.0)
        
        # Penalties
        if action == "wait" and not state.get('monsters_nearby'):
            reward -= 0.1  # Don't wait unnecessarily
        
        # HP-based rewards/penalties
        hp_ratio = state.get('hp_current', 16) / state.get('hp_max', 16)
        if hp_ratio < 0.3:
            reward -= 0.2  # Low HP is bad
        elif hp_ratio > 0.8:
            reward += 0.1  # High HP is good
        
        return max(0.0, min(1.0, reward))  # Clamp to [0, 1]
    
    def _is_terminal(self, node: MCTSNode) -> bool:
        """Check if node represents terminal state"""
        # Consider terminal if very deep (successful) or very low HP
        return (node.state.get('depth', 1) >= 5 or 
                node.state.get('hp_current', 16) <= 0 or
                len(node.children) >= 3)  # Limit branching factor
    
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Parse observation into game state dictionary"""
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        
        import re
        
        state = {
            "depth": 1,
            "hp_current": 16,
            "hp_max": 16,
            "score": 0,
            "on_stairs": False,
            "stairs_visible": False,
            "monsters_nearby": False
        }
        
        # Parse stats
        depth_match = re.search(r'Dlvl:(\\d+)', long_text)
        if depth_match:
            state["depth"] = int(depth_match.group(1))
        
        hp_match = re.search(r'HP:(\\d+)\\((\\d+)\\)', long_text)
        if hp_match:
            state["hp_current"] = int(hp_match.group(1))
            state["hp_max"] = int(hp_match.group(2))
        
        score_match = re.search(r'S:(\\d+)', long_text)
        if score_match:
            state["score"] = int(score_match.group(1))
        
        # Detect stairs and monsters
        state["on_stairs"] = ("staircase down" in long_text.lower() or 
                            "You are on the stairs" in long_text)
        state["stairs_visible"] = "<" in long_text or "staircase" in long_text.lower()
        state["monsters_nearby"] = any(monster in long_text.lower() 
                                     for monster in ["kobold", "goblin", "rat", "fox", "dog", "cat", "newt"])
        
        return state
    
    def _extract_map_context(self, obs_text: str) -> List[str]:
        """Extract spatial/map information from observation"""
        context = []
        
        if "You see here" in obs_text:
            context.append("- Items visible on ground")
        
        if "door" in obs_text.lower():
            context.append("- Door detected nearby")
            
        if "corridor" in obs_text.lower():
            context.append("- In corridor/hallway")
        elif "room" in obs_text.lower():
            context.append("- In room")
            
        if "wall" in obs_text.lower():
            context.append("- Wall blocking movement")
            
        return context if context else ["- Open area, multiple movement options"]
    
    def _extract_tactical_context(self, state: Dict, obs_text: str) -> List[str]:
        """Extract tactical information for decision making"""
        context = []
        
        # HP-based tactics
        hp_ratio = state.get('hp_current', 16) / state.get('hp_max', 16)
        if hp_ratio < 0.3:
            context.append("- CRITICAL: Low HP! Avoid combat, seek healing")
        elif hp_ratio < 0.7:
            context.append("- CAUTION: Moderate HP, avoid unnecessary risks")
        else:
            context.append("- HEALTHY: Full combat capability")
            
        # Exploration tactics
        if state.get('on_stairs'):
            context.append("- OBJECTIVE: Use 'down' action to descend!")
        elif state.get('stairs_visible'):
            context.append("- OBJECTIVE: Navigate to visible stairs")
        else:
            context.append("- OBJECTIVE: Find stairs through exploration")
            
        # Monster tactics
        if state.get('monsters_nearby'):
            context.append("- THREAT: Monster present - fight or evade decision needed")
            
        return context