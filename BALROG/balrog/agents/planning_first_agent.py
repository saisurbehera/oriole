"""
Planning-First Agent - LLM Builds Understanding and Plans Before Acting
=======================================================================

This agent focuses on:
1. Understanding the situation first
2. Creating multi-step plans 
3. Executing plans with memory
4. Learning from outcomes

The key insight: Ask LLM to THINK and PLAN, not just react.
"""

import re
import json
import random
import requests
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse


@dataclass
class GamePlan:
    """A multi-step plan created by the LLM"""
    situation_understanding: str
    threats: List[str]
    opportunities: List[str]
    goal: str
    steps: List[str]
    confidence: float = 0.5
    created_turn: int = 0
    

@dataclass
class SituationMemory:
    """Memory of past situations and their outcomes"""
    description: str
    plan_summary: str
    outcome: str
    success: bool
    lessons_learned: str = ""
    
    def to_prompt_context(self) -> str:
        """Convert to context for future planning"""
        result = "success" if self.success else "failure"
        return f"Past experience: {self.description} → Planned: {self.plan_summary} → {result}. {self.lessons_learned}"


class PlanStatus(Enum):
    """Status of current plan execution"""
    EXECUTING = "executing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class PlanningFirstAgent(BaseAgent):
    """Agent that plans first, then executes"""
    
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # NetHack knowledge server
        self.knowledge_server_url = "http://localhost:8000"
        self.knowledge_cache = {}  # Cache knowledge queries
        
        # Planning state
        self.current_plan: Optional[GamePlan] = None
        self.plan_step = 0
        self.plan_status = PlanStatus.COMPLETED
        
        # Memory systems
        self.situation_memories = deque(maxlen=20)  # Learn from past
        self.position_history = deque(maxlen=10)    # Track movement
        self.action_history = deque(maxlen=20)      # Recent actions
        
        # Game state tracking
        self.turn_count = 0
        self.last_hp = None
        self.last_score = 0
        self.last_depth = 1
        self.visited_positions = set()
        
        # Planning frequency
        self.plan_every = 5  # How often to make new plans (reduced for better execution)
        self.last_planning_turn = 0
        
    def query_knowledge_server(self, question: str, entity: str = None) -> str:
        """Query the NetHack knowledge server for game-specific information"""
        # Create cache key
        cache_key = f"{question}::{entity}" if entity else question
        
        # Check cache first
        if cache_key in self.knowledge_cache:
            return self.knowledge_cache[cache_key]
        
        try:
            payload = {
                "question": question,
                "max_facts": 5,
                "include_predictions": True
            }
            if entity:
                payload["entity"] = entity
                
            response = requests.post(
                f"{self.knowledge_server_url}/query",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer available")
                # Cache the response
                self.knowledge_cache[cache_key] = answer
                return answer
            else:
                return f"Server error: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"Connection error: {str(e)}"
    
    def get_combat_advice(self, monster_name: str) -> str:
        """Get specific advice for fighting a monster"""
        question = f"How should I fight a {monster_name} in NetHack? What are its weaknesses and strengths?"
        return self.query_knowledge_server(question, entity=monster_name.title())
    
    def get_item_advice(self, item_char: str) -> str:
        """Get advice about what an item might be"""
        question = f"In NetHack, what could the '{item_char}' symbol represent? Should I pick it up?"
        return self.query_knowledge_server(question)
    
    def get_level_advice(self, depth: int) -> str:
        """Get advice about what to expect at this dungeon level"""
        question = f"What should I expect on dungeon level {depth} in NetHack? What monsters and items are common?"
        return self.query_knowledge_server(question)
        
    def act(self, obs, prev_action=None):
        """Main action selection with planning-first approach"""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        self.turn_count += 1
        
        # Parse current situation
        game_state = self._parse_game_state(obs)
        
        # Update tracking
        self._update_state_tracking(game_state)
        
        # Check if we need a new plan
        if self._need_new_plan(game_state):
            self._create_new_plan(game_state, obs)
            
        # Execute current plan step
        action = self._execute_plan_step(game_state)
        
        # CRITICAL: If we're on stairs and not using them, override!
        if game_state["stairs_info"] and game_state["player_pos"]:
            px, py = game_state["player_pos"]
            sx, sy = game_state["stairs_info"]["pos"]
            if px == sx and py == sy:
                # We're ON the stairs! Use them!
                if game_state["stairs_info"]["type"] == "up" and game_state["depth"] == 1:
                    action = ">"  # Go down from level 1
        
        # Record action
        self.action_history.append(action)
        
        return LLMResponse(
            model_id="planning_first",
            completion=action,
            stop_reason="complete",
            input_tokens=200,  # Planning uses more tokens
            output_tokens=20,
            reasoning=self._get_reasoning()
        )
        
    def _parse_game_state(self, obs) -> Dict[str, Any]:
        """Extract key information from observation"""
        text = obs.get("text", {})
        long_text = text.get("long_term_context", "")
        lang_obs = text.get("language_observation", "")
        
        # Parse map and position
        game_map = self._parse_map(obs)
        player_pos = self._find_player(game_map)
        
        # Parse stats
        hp_match = re.search(r'HP:(\d+)\((\d+)\)', long_text)
        hp_current = 0
        hp_max = 1
        if hp_match:
            hp_current = int(hp_match.group(1))
            hp_max = int(hp_match.group(2))
            
        depth_match = re.search(r'Dlvl:(\d+)', long_text)
        depth = int(depth_match.group(1)) if depth_match else 1
        
        score_match = re.search(r'S:(\d+)', long_text)
        score = int(score_match.group(1)) if score_match else 0
        
        # Analyze immediate environment
        stairs_info = self._find_stairs(game_map)
        doors_nearby = self._check_doors_nearby(game_map, player_pos)
        items_nearby = self._check_items_nearby(game_map, player_pos)
        monsters_nearby = self._check_monsters_nearby(lang_obs)
        
        return {
            "player_pos": player_pos,
            "hp_current": hp_current,
            "hp_max": hp_max,
            "hp_percent": hp_current / hp_max if hp_max > 0 else 0,
            "depth": depth,
            "score": score,
            "stairs_info": stairs_info,
            "doors_nearby": doors_nearby,
            "items_nearby": items_nearby,
            "monsters_nearby": monsters_nearby,
            "game_map": game_map,
            "lang_obs": lang_obs,
            "turn": self.turn_count
        }
        
    def _need_new_plan(self, game_state) -> bool:
        """Determine if we need to create a new plan"""
        # No current plan
        if not self.current_plan:
            return True
            
        # Plan completed
        if self.plan_step >= len(self.current_plan.steps):
            self.plan_status = PlanStatus.COMPLETED
            return True
            
        # Emergency situations - health dropped significantly
        if (self.last_hp and 
            game_state["hp_current"] < self.last_hp - 5):
            self.plan_status = PlanStatus.INTERRUPTED
            return True
            
        # Stuck detection - not moving
        if len(set(self.position_history)) <= 2 and len(self.position_history) >= 5:
            self.plan_status = PlanStatus.INTERRUPTED
            return True
            
        # Major progress - new floor
        if game_state["depth"] != self.last_depth:
            self.plan_status = PlanStatus.COMPLETED
            return True
            
        # Periodic replanning
        if self.turn_count - self.last_planning_turn > 20:
            return True
            
        return False
        
    def _create_new_plan(self, game_state, obs):
        """Use LLM to analyze situation and create plan"""
        # Build rich context for planning
        situation_description = self._describe_situation(game_state)
        
        # Add memory context
        memory_context = self._build_memory_context()
        
        # Get domain knowledge
        domain_knowledge = self._gather_domain_knowledge(game_state)
        
        # Create planning prompt with strict JSON format
        prompt = f"""You are playing NetHack. Your ONLY goal is to go down stairs to reach deeper dungeon levels.

CURRENT SITUATION:
{situation_description}

GAME KNOWLEDGE:
{domain_knowledge}

MEMORY:
{memory_context}

Create a simple plan. Reply with ONLY this JSON:

{{
  "situation": "Where am I and what do I see",
  "threats": ["threat1"],
  "opportunities": ["opportunity1"],  
  "goal": "Find and use stairs down",
  "steps": [
    "Step 1: <action>",
    "Step 2: <action>",
    "Step 3: <action>"
  ]
}}

RULES:
- If you see stairs, goal is "Use stairs at <direction>"
- Valid actions: north, south, east, west, northeast, northwest, southeast, southwest, >, <, open, kick, search, wait
- Use > to go DOWN stairs when standing on them
- Keep it simple!"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.generate(messages)
            
            # Parse the response into a plan
            plan = self._parse_plan_response(response.completion)
            
            if plan:
                self.current_plan = plan
                self.plan_step = 0
                self.plan_status = PlanStatus.EXECUTING
                self.last_planning_turn = self.turn_count
                
                # Record the planning decision
                self._record_planning_decision(situation_description, plan)
            else:
                # Fallback plan if parsing failed
                self.current_plan = self._create_fallback_plan(game_state)
                
        except Exception as e:
            # LLM failed, create simple fallback
            self.current_plan = self._create_fallback_plan(game_state)
            
    def _describe_situation(self, game_state) -> str:
        """Create natural language description of current situation"""
        parts = []
        
        # Health status
        hp_percent = game_state["hp_percent"]
        if hp_percent > 0.8:
            health_desc = f"excellent health ({game_state['hp_current']}/{game_state['hp_max']} HP)"
        elif hp_percent > 0.5:
            health_desc = f"good health ({game_state['hp_current']}/{game_state['hp_max']} HP)"
        elif hp_percent > 0.3:
            health_desc = f"injured ({game_state['hp_current']}/{game_state['hp_max']} HP)"
        else:
            health_desc = f"critically injured ({game_state['hp_current']}/{game_state['hp_max']} HP)"
            
        parts.append(f"You have {health_desc}")
        
        # Location
        parts.append(f"You are on dungeon level {game_state['depth']}")
        
        # Immediate environment
        if game_state["stairs_info"]:
            stair_type = game_state["stairs_info"]["type"]
            stair_dir = self._get_direction_to(game_state["player_pos"], game_state["stairs_info"]["pos"])
            parts.append(f"Stairs going {stair_type} are {stair_dir}")
            
        if game_state["doors_nearby"]:
            parts.append(f"There are {len(game_state['doors_nearby'])} doors nearby")
            
        if game_state["items_nearby"]:
            parts.append(f"You see {len(game_state['items_nearby'])} items on the ground")
            
        if game_state["monsters_nearby"]:
            parts.append(f"Warning: {len(game_state['monsters_nearby'])} monsters detected nearby")
            
        # Recent movement
        if len(set(self.position_history)) <= 2:
            parts.append("You seem to be stuck or moving in circles")
            
        return ". ".join(parts) + "."
        
    def _gather_domain_knowledge(self, game_state) -> str:
        """Gather relevant domain knowledge for current situation"""
        knowledge_parts = []
        
        # Only get level advice if we changed levels
        if game_state["depth"] != self.last_depth:
            level_advice = self.get_level_advice(game_state["depth"])
            if level_advice and "error" not in level_advice.lower():
                knowledge_parts.append(f"Level {game_state['depth']} info: {level_advice}")
        
        # Get monster advice only for immediate threats (not pets)
        if game_state["monsters_nearby"]:
            # Only query about the first monster to reduce queries
            monster = game_state["monsters_nearby"][0]
            monster_advice = self.get_combat_advice(monster)
            if monster_advice and "error" not in monster_advice.lower():
                knowledge_parts.append(f"{monster.title()} combat: {monster_advice}")
        
        # Skip item queries for now - they're less critical and cause many queries
        
        # Get healing advice only once when low on health
        if game_state["hp_percent"] < 0.5 and "healing_queried" not in self.knowledge_cache:
            healing_advice = self.query_knowledge_server("How can I heal in NetHack when low on health?")
            if healing_advice and "error" not in healing_advice.lower():
                knowledge_parts.append(f"Healing strategy: {healing_advice}")
                self.knowledge_cache["healing_queried"] = True
        
        # Always remind about stairs if visible
        if game_state["stairs_info"]:
            knowledge_parts.append("REMINDER: Stairs are visible! Reaching them should be your priority.")
        
        if not knowledge_parts:
            return "Focus on finding and using stairs to descend deeper into the dungeon."
        
        return "\n".join(knowledge_parts)
        
    def _build_memory_context(self) -> str:
        """Build context from past experiences"""
        if not self.situation_memories:
            return "No previous experience to draw from."
            
        # Get most relevant recent memories
        recent_memories = list(self.situation_memories)[-3:]
        
        context_parts = []
        for memory in recent_memories:
            context_parts.append(memory.to_prompt_context())
            
        return "\n".join(context_parts)
        
    def _parse_plan_response(self, response: str) -> Optional[GamePlan]:
        """Parse JSON LLM response into a structured plan"""
        try:
            # Try to find JSON in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return None
                
            json_str = response[json_start:json_end]
            plan_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['situation', 'goal', 'steps']
            for field in required_fields:
                if field not in plan_data:
                    return None
                    
            # Extract data with defaults
            situation = plan_data.get('situation', '')
            threats = plan_data.get('threats', [])
            opportunities = plan_data.get('opportunities', [])
            goal = plan_data.get('goal', '')
            steps = plan_data.get('steps', [])
            
            # Ensure lists are actually lists
            if not isinstance(threats, list):
                threats = [str(threats)] if threats else []
            if not isinstance(opportunities, list):
                opportunities = [str(opportunities)] if opportunities else []
            if not isinstance(steps, list):
                steps = [str(steps)] if steps else []
                
            # Limit list sizes
            threats = threats[:3]
            opportunities = opportunities[:3]
            steps = steps[:5]
            
            if not steps:
                return None
                
            return GamePlan(
                situation_understanding=str(situation),
                threats=threats,
                opportunities=opportunities,
                goal=str(goal),
                steps=steps,
                created_turn=self.turn_count
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return None
            
    def _create_fallback_plan(self, game_state) -> GamePlan:
        """Create simple fallback plan when LLM fails"""
        # CRITICAL: Check if we're already on the stairs!
        if game_state["stairs_info"] and game_state["player_pos"]:
            if self._adjacent(game_state["player_pos"], game_state["stairs_info"]["pos"]):
                # We're at the stairs - USE THEM!
                goal = "USE THE STAIRS NOW"
                steps = ["Use stairs immediately", "Press > to go down", "Descend now"]
            else:
                # Navigate to stairs
                goal = "Reach and use the stairs"
                stairs_dir = self._get_direction_to(game_state["player_pos"], game_state["stairs_info"]["pos"])
                steps = [f"Move {stairs_dir} toward stairs", "Get to stairs position", "Use stairs with >"]
        elif game_state["hp_percent"] < 0.3:
            goal = "Find safety and heal"
            steps = ["Move away from threats", "Look for safe area", "Wait to recover"]
        else:
            # Need to find stairs - be more aggressive about exploration
            goal = "Find stairs to descend" 
            
            # Look for unexplored areas
            map_analysis = self._analyze_map_for_exploration(game_state["game_map"], game_state["player_pos"])
            
            if map_analysis["doors"]:
                # There are doors - open them!
                door_dir = self._get_direction_to(game_state["player_pos"], map_analysis["doors"][0])
                steps = [f"Move {door_dir} to door", "Open door", "Explore new area"]
            elif map_analysis["unexplored_directions"]:
                # Move toward unexplored areas
                direction = map_analysis["unexplored_directions"][0]
                steps = [f"Explore {direction}", "Look for passages", "Find stairs"]
            else:
                # Fallback exploration
                steps = ["Try kick on walls", "Search for secret doors", "Explore edges"]
            
        return GamePlan(
            situation_understanding="Fallback plan due to LLM error",
            threats=["Unknown"],
            opportunities=["Unknown"],
            goal=goal,
            steps=steps,
            confidence=0.3,
            created_turn=self.turn_count
        )
        
    def _execute_plan_step(self, game_state) -> str:
        """Execute the current step of the plan"""
        if not self.current_plan or self.plan_step >= len(self.current_plan.steps):
            # Force replanning instead of waiting
            self.current_plan = None
            return "search"  # Better than wait - might find hidden features
            
        current_step = self.current_plan.steps[self.plan_step]
        
        # Convert plan step to concrete action
        action = self._step_to_action(current_step, game_state)
        
        # Advance plan
        self.plan_step += 1
        
        # If this was the last step, clear plan to force replanning
        if self.plan_step >= len(self.current_plan.steps):
            self.current_plan = None
        
        return action
        
    def _step_to_action(self, step: str, game_state) -> str:
        """Convert a plan step to a concrete NetHack action"""
        step_lower = step.lower()
        
        # PRIORITY 1: Direct stair commands
        if ">" in step or "press >" in step_lower:
            return ">"
        elif "<" in step or "press <" in step_lower:
            return "<"
            
        # PRIORITY 2: Stair navigation
        elif "stairs" in step_lower or "descend" in step_lower or "use stairs" in step_lower:
            stairs_info = game_state["stairs_info"]
            if stairs_info and game_state["player_pos"]:
                if self._adjacent(game_state["player_pos"], stairs_info["pos"]):
                    # In NetHack: '>' goes down, '<' goes up
                    # But "stairs up" means stairs that lead up FROM BELOW
                    # So on level 1, "stairs up" should use '>' to go down
                    return ">" if stairs_info["type"] == "up" else "<"
                else:
                    # Navigate toward stairs
                    return self._navigate_to(game_state["player_pos"], stairs_info["pos"])
            else:
                # No stairs visible - explore
                return self._explore_direction(game_state)
        
        # Movement patterns
        elif "north" in step_lower:
            return "north"
        elif "south" in step_lower:
            return "south"
        elif "east" in step_lower:
            return "east"
        elif "west" in step_lower:
            return "west"
                    
        # Door handling
        elif "door" in step_lower or "open" in step_lower:
            if "kick" in step_lower:
                return "kick"
            else:
                return "open"
                
        # Item interaction
        elif "pick" in step_lower or "item" in step_lower:
            return ","
            
        # Safety actions
        elif "wait" in step_lower or "rest" in step_lower:
            return "wait"
        elif "search" in step_lower:
            return "search"
            
        # Movement toward goals
        elif "move" in step_lower or "go" in step_lower:
            # Try to infer direction from context
            if game_state["stairs_info"]:
                return self._navigate_to(game_state["player_pos"], game_state["stairs_info"]["pos"])
            else:
                # Explore in a direction we haven't been
                return self._explore_direction(game_state)
                
        # Default fallback
        return self._explore_direction(game_state)
        
    def _record_planning_decision(self, situation: str, plan: GamePlan):
        """Record that we made a planning decision"""
        # This will be used later to learn from outcomes
        pass
        
    def _update_state_tracking(self, game_state):
        """Update our tracking of game state"""
        # Update position history
        if game_state["player_pos"]:
            self.position_history.append(game_state["player_pos"])
            self.visited_positions.add((game_state["depth"], game_state["player_pos"]))
            
        # Update state tracking
        self.last_hp = game_state["hp_current"]
        self.last_score = game_state["score"]
        self.last_depth = game_state["depth"]
        
    def _get_reasoning(self) -> str:
        """Get current reasoning for display"""
        if not self.current_plan:
            return "No plan"
            
        current_step = "completed"
        if self.plan_step < len(self.current_plan.steps):
            current_step = self.current_plan.steps[self.plan_step]
            
        return f"Goal: {self.current_plan.goal} | Step {self.plan_step + 1}: {current_step}"
        
    # Utility methods (keeping them simple for now)
    def _parse_map(self, obs):
        """Parse the game map"""
        text = obs.get("text", {})
        map_text = text.get("long_term_context", "")
        lines = map_text.split('\n')
        
        map_start = -1
        for i, line in enumerate(lines):
            if line.strip() == "map:":
                map_start = i + 1
                break
                
        if map_start == -1:
            return []
            
        game_map = []
        for i in range(map_start, min(map_start + 21, len(lines))):
            if i < len(lines):
                if lines[i].strip().startswith('Agent'):
                    break
                # Keep the original line to preserve column positions
                game_map.append(lines[i])
                
        return game_map
        
    def _find_player(self, game_map):
        """Find player position"""
        for row, line in enumerate(game_map):
            if '@' in line:
                return (row, line.index('@'))
        return None
        
    def _find_stairs(self, game_map):
        """Find stairs and their type"""
        for row, line in enumerate(game_map):
            if '>' in line:
                return {'pos': (row, line.index('>')), 'type': 'down'}
            elif '<' in line:
                return {'pos': (row, line.index('<')), 'type': 'up'}
        
        # Also check language observation for stairs
        # This is a backup in case map parsing misses it
        return None
        
    def _check_doors_nearby(self, game_map, player_pos):
        """Check for doors near player"""
        if not player_pos:
            return []
            
        doors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                row, col = player_pos[0] + dr, player_pos[1] + dc
                if 0 <= row < len(game_map) and 0 <= col < len(game_map[row]):
                    if game_map[row][col] == '+':
                        doors.append((row, col))
        return doors
        
    def _check_items_nearby(self, game_map, player_pos):
        """Check for items near player"""
        if not player_pos:
            return []
            
        items = []
        item_chars = '$!?/="()[]*%'
        
        for row, line in enumerate(game_map):
            for col, char in enumerate(line):
                if char in item_chars:
                    # Check if it's reasonably close
                    if player_pos and abs(row - player_pos[0]) + abs(col - player_pos[1]) <= 5:
                        items.append((row, col, char))
        return items
        
    def _check_monsters_nearby(self, lang_obs):
        """Check for monsters mentioned in language observation"""
        monster_words = [
            "goblin", "kobold", "jackal", "orc", "rat", "bat", "snake", "troll", "dragon",
            "zombie", "skeleton", "gnome", "dwarf", "elf", "giant", "spider", "ant",
            "centipede", "newt", "grid bug", "lichen", "acid blob", "quasit", "imp"
        ]
        monsters = []
        
        lang_lower = lang_obs.lower()
        
        # Exclude pets (tame animals)
        if "tame" in lang_lower:
            return monsters
            
        for monster in monster_words:
            if monster in lang_lower and "tame" not in lang_lower:
                # Make sure it's not a pet reference
                monster_context = self._get_word_context(lang_lower, monster)
                if "tame" not in monster_context:
                    monsters.append(monster)
                
        return monsters
    
    def _get_word_context(self, text: str, word: str, context_size: int = 20) -> str:
        """Get context around a word in text"""
        index = text.find(word)
        if index == -1:
            return ""
        start = max(0, index - context_size)
        end = min(len(text), index + len(word) + context_size)
        return text[start:end]
        
    def _adjacent(self, pos1, pos2):
        """Check if positions are adjacent"""
        return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1
        
    def _navigate_to(self, from_pos, to_pos):
        """Simple navigation toward target"""
        if not from_pos or not to_pos:
            return "wait"
            
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        
        # Move in direction of largest difference
        if abs(dr) > abs(dc):
            return "north" if dr < 0 else "south"
        elif dc != 0:
            return "west" if dc < 0 else "east"
        else:
            return "wait"
            
    def _get_direction_to(self, from_pos, to_pos):
        """Get direction description to target"""
        if not from_pos or not to_pos:
            return "unknown direction"
            
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        
        if abs(dr) > abs(dc):
            return "north" if dr < 0 else "south"
        elif dc != 0:
            return "west" if dc < 0 else "east"
        else:
            return "right here"
            
    def _analyze_map_for_exploration(self, game_map, player_pos):
        """Analyze map to find doors and unexplored areas"""
        result = {
            "doors": [],
            "unexplored_directions": []
        }
        
        if not player_pos or not game_map:
            return result
            
        px, py = player_pos
        
        # Check for doors near player
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                row, col = px + dr, py + dc
                if 0 <= row < len(game_map) and 0 <= col < len(game_map[row]):
                    char = game_map[row][col] if col < len(game_map[row]) else ' '
                    if char == '+':
                        result["doors"].append((row, col))
        
        # Check which directions have unexplored space
        # North
        if px > 0 and py < len(game_map[px-1]) and game_map[px-1][py] == '.':
            result["unexplored_directions"].append("north")
        # South  
        if px < len(game_map)-1 and py < len(game_map[px+1]) and game_map[px+1][py] == '.':
            result["unexplored_directions"].append("south")
        # East
        if py < 79 and py+1 < len(game_map[px]) and game_map[px][py+1] == '.':
            result["unexplored_directions"].append("east")
        # West
        if py > 0 and game_map[px][py-1] == '.':
            result["unexplored_directions"].append("west")
            
        return result
    
    def _explore_direction(self, game_state):
        """Choose exploration direction"""
        # Simple exploration strategy
        directions = ["north", "south", "east", "west"]
        
        # Avoid recent directions if stuck
        if len(set(self.position_history)) <= 2:
            recent_actions = list(self.action_history)[-5:]
            available = [d for d in directions if d not in recent_actions]
            if available:
                return random.choice(available)
                
        return random.choice(directions)
        
    def reset(self):
        """Reset for new episode"""
        self.prompt_builder.reset()
        self.current_plan = None
        self.plan_step = 0
        self.plan_status = PlanStatus.COMPLETED
        self.situation_memories.clear()
        self.position_history.clear()
        self.action_history.clear()
        self.turn_count = 0
        self.last_hp = None
        self.last_score = 0
        self.last_depth = 1
        self.visited_positions.clear()
        self.last_planning_turn = 0
        self.knowledge_cache.clear()  # Clear cache for new episode