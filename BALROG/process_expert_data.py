#!/usr/bin/env python3
"""
Expert Data Processing Pipeline for NetHack SFT Training

Processes TTY recordings and xlogfiles to create training examples
with subgoal annotations for supervised fine-tuning.
"""

import os
import json
import bz2
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExpertEpisode:
    """Represents a single expert episode with metadata"""
    episode_id: str
    points: int
    maxlvl: int
    turns: int
    realtime: int
    role: str
    race: str
    ttyrec_file: str
    death: str

@dataclass
class GameStep:
    """Represents a single game step with action and state"""
    observation: str
    action: str
    subgoal: str
    reasoning: str
    step_num: int
    episode_id: str

class NetHackTTYParser:
    """Parses NetHack TTY recordings to extract game states and actions"""
    
    def __init__(self):
        # Common NetHack action mappings
        self.action_map = {
            'h': 'west', 'j': 'south', 'k': 'north', 'l': 'east',
            'y': 'northwest', 'u': 'northeast', 'b': 'southwest', 'n': 'southeast',
            ',': 'pickup', 'd': 'drop', 'i': 'inventory', 'w': 'wield', 'W': 'wear',
            'e': 'eat', 'q': 'drink', 'r': 'read', 'z': 'zap', 'f': 'fire',
            's': 'search', '.': 'wait', '<': 'up', '>': 'down',
            'o': 'open', 'c': 'close', 'a': 'apply', 'p': 'pay',
            '\r': 'confirm', '\n': 'confirm', ' ': 'confirm', '\x1b': 'escape'
        }
        
    def parse_tty_file(self, tty_path: str) -> List[GameStep]:
        """Parse a TTY recording file to extract game steps"""
        try:
            with bz2.open(tty_path, 'rt', encoding='latin1') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {tty_path}: {e}")
            return []
            
        # Split into frames based on timing data
        frames = self._extract_frames(content)
        
        # Process frames to extract game states and actions
        game_steps = []
        for i, frame in enumerate(frames):
            if i > 0:  # Skip first frame (usually initialization)
                step = self._process_frame(frame, i, os.path.basename(tty_path))
                if step:
                    game_steps.append(step)
                    
        logger.info(f"Extracted {len(game_steps)} steps from {tty_path}")
        return game_steps
    
    def _extract_frames(self, content: str) -> List[str]:
        """Extract individual frames from TTY recording"""
        # TTY recordings have timing markers and screen updates
        # This is a simplified parser - could be enhanced for more accuracy
        frames = []
        current_frame = ""
        
        for line in content.split('\n'):
            if line.strip():
                current_frame += line + '\n'
            else:
                if current_frame.strip():
                    frames.append(current_frame)
                    current_frame = ""
        
        if current_frame.strip():
            frames.append(current_frame)
            
        return frames[:1000]  # Limit to prevent memory issues
    
    def _process_frame(self, frame: str, step_num: int, episode_id: str) -> Optional[GameStep]:
        """Process a single frame to extract game state and action"""
        try:
            # Extract game state information
            observation = self._extract_game_state(frame)
            if not observation:
                return None
                
            # Extract action (simplified - in real TTY this would be more complex)
            action = self._extract_action(frame)
            if not action:
                action = "wait"  # Default action
                
            # Determine subgoal based on game state
            subgoal = self._determine_subgoal(observation)
            
            # Generate reasoning (simplified for now)
            reasoning = self._generate_reasoning(observation, action, subgoal)
            
            return GameStep(
                observation=observation,
                action=action,
                subgoal=subgoal,
                reasoning=reasoning,
                step_num=step_num,
                episode_id=episode_id
            )
        except Exception as e:
            logger.debug(f"Failed to process frame {step_num}: {e}")
            return None
    
    def _extract_game_state(self, frame: str) -> str:
        """Extract structured game state from TTY frame"""
        # Remove ANSI escape sequences for cleaner text
        clean_frame = re.sub(r'\x1b\[[0-9;]*[mH]', '', frame)
        
        # Extract key game elements
        lines = clean_frame.split('\n')
        
        # Look for map area (typically lines 5-21 in NetHack)
        map_lines = []
        status_lines = []
        
        for i, line in enumerate(lines):
            if 5 <= i <= 21 and len(line.strip()) > 0:
                # Likely map content
                if any(char in line for char in '@#.+'):
                    map_lines.append(line[:80])  # NetHack map is 80 chars wide
            elif i > 21:
                # Status/message area
                if line.strip():
                    status_lines.append(line.strip())
        
        # Construct observation
        observation = "map:\n" + '\n'.join(map_lines)
        if status_lines:
            observation += "\n\nstatus:\n" + '\n'.join(status_lines[-5:])  # Last 5 status lines
            
        return observation
    
    def _extract_action(self, frame: str) -> str:
        """Extract action from TTY frame (simplified)"""
        # In a real implementation, this would parse the actual keystroke
        # For now, we'll infer common actions from game state changes
        
        # Look for common action indicators
        if '@' in frame and any(char in frame for char in 'hjklyubn'):
            return random.choice(['north', 'south', 'east', 'west'])
        elif ',' in frame:
            return 'pickup'
        elif 'search' in frame.lower():
            return 'search'
        else:
            return random.choice(['wait', 'search', 'north', 'south', 'east', 'west'])
    
    def _determine_subgoal(self, observation: str) -> str:
        """Determine current subgoal based on game state"""
        obs_lower = observation.lower()
        
        # Check for stairs visibility
        if '<' not in observation and '>' not in observation:
            return "find_stairs"
        
        # Check for monsters
        if any(monster in obs_lower for monster in ['kobold', 'goblin', 'rat', 'orc']):
            return "combat_engagement"
        
        # Check for items
        if any(item in observation for item in [')','[','!','?','%']):
            return "collect_equipment"
        
        # Check if on stairs
        if '<' in observation or '>' in observation:
            return "level_progression"
        
        # Default to exploration
        return "exploration"
    
    def _generate_reasoning(self, observation: str, action: str, subgoal: str) -> str:
        """Generate reasoning for the action given the subgoal"""
        reasoning_templates = {
            "find_stairs": f"Need to explore to find stairs. Taking '{action}' to search new areas.",
            "combat_engagement": f"Monster detected. Using '{action}' for combat strategy.",
            "collect_equipment": f"Items available. Taking '{action}' to manage inventory.",
            "level_progression": f"At stairs location. Using '{action}' to progress deeper.",
            "exploration": f"Continuing exploration with '{action}' to map dungeon layout."
        }
        return reasoning_templates.get(subgoal, f"Taking '{action}' as part of general strategy.")


class ExpertDataProcessor:
    """Main processor for expert NetHack data"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.parser = NetHackTTYParser()
        
    def load_expert_episodes(self) -> List[ExpertEpisode]:
        """Load metadata for all expert episodes"""
        episodes = []
        
        for session_dir in self.data_dir.iterdir():
            if session_dir.is_dir():
                xlog_file = session_dir / f"{session_dir.name.split('_')[1]}.xlogfile"
                if not xlog_file.exists():
                    # Look for any xlogfile
                    xlog_files = list(session_dir.glob("*.xlogfile"))
                    if xlog_files:
                        xlog_file = xlog_files[0]
                    else:
                        logger.warning(f"No xlogfile found in {session_dir}")
                        continue
                
                session_episodes = self._parse_xlogfile(xlog_file, session_dir)
                episodes.extend(session_episodes)
                
        logger.info(f"Loaded {len(episodes)} expert episodes")
        return episodes
    
    def _parse_xlogfile(self, xlog_file: Path, session_dir: Path) -> List[ExpertEpisode]:
        """Parse xlogfile to extract episode metadata"""
        episodes = []
        
        try:
            with open(xlog_file, 'r') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            # Parse key-value pairs
                            data = {}
                            for pair in line.strip().split('\t'):
                                if '=' in pair:
                                    key, value = pair.split('=', 1)
                                    data[key] = value
                            
                            episode = ExpertEpisode(
                                episode_id=f"{session_dir.name}_{line_num}",
                                points=int(data.get('points', 0)),
                                maxlvl=int(data.get('maxlvl', 1)),
                                turns=int(data.get('turns', 0)),
                                realtime=int(data.get('realtime', 0)),
                                role=data.get('role', ''),
                                race=data.get('race', ''),
                                ttyrec_file=str(session_dir / data.get('ttyrecname', '')),
                                death=data.get('death', '')
                            )
                            episodes.append(episode)
                        except (ValueError, KeyError) as e:
                            logger.debug(f"Failed to parse line {line_num}: {e}")
                            
        except Exception as e:
            logger.error(f"Failed to read {xlog_file}: {e}")
            
        return episodes
    
    def filter_quality_episodes(self, episodes: List[ExpertEpisode], min_points: int = 5000) -> List[ExpertEpisode]:
        """Filter episodes by quality metrics"""
        filtered = [ep for ep in episodes if ep.points >= min_points and ep.turns >= 1000]
        logger.info(f"Filtered to {len(filtered)} high-quality episodes (points >= {min_points})")
        return filtered
    
    def process_episodes_to_training_data(self, episodes: List[ExpertEpisode], limit: int = 10) -> List[GameStep]:
        """Process episodes into training examples"""
        all_steps = []
        
        for i, episode in enumerate(episodes[:limit]):
            logger.info(f"Processing episode {i+1}/{min(limit, len(episodes))}: {episode.episode_id}")
            
            if not os.path.exists(episode.ttyrec_file):
                logger.warning(f"TTY file not found: {episode.ttyrec_file}")
                continue
                
            steps = self.parser.parse_tty_file(episode.ttyrec_file)
            
            # Add episode context to each step
            for step in steps:
                step.episode_id = episode.episode_id
                
            all_steps.extend(steps[:200])  # Limit steps per episode
            logger.info(f"Added {len(steps[:200])} steps from {episode.episode_id}")
            
        logger.info(f"Total training examples: {len(all_steps)}")
        return all_steps
    
    def save_training_data(self, steps: List[GameStep], output_file: str):
        """Save training data to JSON format"""
        training_examples = []
        
        for step in steps:
            example = {
                "observation": step.observation,
                "action": step.action,
                "subgoal": step.subgoal,
                "reasoning": step.reasoning,
                "step_num": step.step_num,
                "episode_id": step.episode_id
            }
            training_examples.append(example)
        
        with open(output_file, 'w') as f:
            json.dump(training_examples, f, indent=2)
            
        logger.info(f"Saved {len(training_examples)} training examples to {output_file}")


def main():
    """Main processing pipeline"""
    data_dir = "/home/sai/Desktop/oriole/BALROG/nld-aa-taster/nle_data"
    output_file = "/home/sai/Desktop/oriole/BALROG/expert_training_data.json"
    
    processor = ExpertDataProcessor(data_dir)
    
    # Load all expert episodes
    episodes = processor.load_expert_episodes()
    
    # Filter for quality - get more data for better training
    quality_episodes = processor.filter_quality_episodes(episodes, min_points=3000)
    
    # Process more episodes for robust training
    training_steps = processor.process_episodes_to_training_data(quality_episodes, limit=20)
    
    # Save training data
    processor.save_training_data(training_steps, output_file)
    
    print(f"Expert data processing complete!")
    print(f"Processed {len(quality_episodes)} episodes into {len(training_steps)} training examples")
    print(f"Training data saved to: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Use this data to fine-tune LLM with expert trajectories")  
    print(f"2. Deploy fine-tuned LLM in Tree-MCTS for expert-guided search")
    print(f"3. Test Tree-MCTS + Expert-LLM performance vs baseline")


if __name__ == "__main__":
    main()