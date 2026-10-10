#!/usr/bin/env python3
"""
Improved REBEL extraction for NetHack wiki with game-specific focus
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from typing import List, Dict, Tuple, Set
import re
from collections import defaultdict
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedNetHackKGExtractor:
    def __init__(self):
        # Load REBEL model
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Using GPU for extraction")
        
        # NetHack-specific patterns and relation mappings
        self.init_nethack_patterns()
        
    def init_nethack_patterns(self):
        """Initialize NetHack-specific patterns for better extraction"""
        
        # Map generic relations to game-specific ones
        self.relation_mappings = {
            # Item properties
            'has_property': ['blessed', 'cursed', 'uncursed', 'rustproof', 'fireproof', 'rotproof'],
            'damage': ['d4', 'd6', 'd8', 'd10', 'd12', '2d4', '2d6', '3d4'],
            'armor_class': ['AC', 'armor class', 'protection'],
            'weight': ['weighs', 'weight'],
            'cost': ['zorkmids', 'cost', 'price'],
            'material': ['iron', 'wood', 'leather', 'cloth', 'metal', 'stone', 'glass', 'mithril'],
            
            # Game mechanics
            'grants_ability': ['grants', 'confers', 'provides', 'gives'],
            'resists': ['resistance', 'resist', 'immune'],
            'vulnerability': ['vulnerable', 'weakness', 'susceptible'],
            'required_for': ['needed for', 'required for', 'necessary for'],
            'found_in': ['found in', 'appears in', 'located in', 'generated in'],
            'dropped_by': ['dropped by', 'carried by', 'wielded by'],
            
            # Monster properties
            'alignment': ['lawful', 'neutral', 'chaotic'],
            'difficulty': ['difficulty', 'level', 'HD'],
            'speed': ['speed', 'movement rate'],
            'attacks': ['attacks', 'bite', 'claw', 'sting', 'touch', 'breath'],
            
            # Locations
            'dungeon_level': ['level', 'floor', 'depth'],
            'special_level': ['special level', 'branch', 'quest'],
        }
        
        # Patterns to extract game-specific information
        self.game_patterns = [
            # Item patterns
            (r'(\w+) (?:deals?|does?) (\d+d\d+) damage', 'damage'),
            (r'(\w+) (?:grants?|confers?|provides?) (.+?) resistance', 'grants_resistance'),
            (r'(\w+) (?:weighs?|has weight) (\d+)', 'weight'),
            (r'(\w+) costs? (\d+) zorkmids?', 'cost'),
            (r'(\w+) (?:is|are) made of (\w+)', 'material'),
            (r'(\w+) (?:has|have) (?:AC|armor class) ([+-]?\d+)', 'armor_class'),
            
            # Monster patterns
            (r'(\w+) (?:is|are) (\w+)-aligned?', 'alignment'),
            (r'(\w+) (?:has|have) difficulty (\d+)', 'difficulty'),
            (r'(\w+) (?:attacks?|hits?) with (.+)', 'attack_type'),
            (r'(\w+) drops? (.+)', 'drops'),
            
            # Location patterns
            (r'(\w+) (?:is|are) found (?:in|on) (.+)', 'found_in'),
            (r'(\w+) appears? (?:in|on) (.+)', 'appears_in'),
            (r'(\w+) (?:is|are) located (?:in|on|at) (.+)', 'located_in'),
            
            # Quest/artifact patterns
            (r'(\w+) (?:is|are) the quest artifact of (.+)', 'quest_artifact_of'),
            (r'(\w+) (?:is|are) required for (.+)', 'required_for'),
            (r'(\w+) (?:is|are) needed to (.+)', 'needed_to'),
        ]
        
        # Important NetHack entities to prioritize
        self.important_entities = {
            'artifacts': ['Amulet of Yendor', 'Excalibur', 'Mjollnir', 'Stormbringer', 
                         'Grayswandir', 'Orcrist', 'Sting', 'Magicbane', 'Sunsword'],
            'locations': ['Sokoban', 'Gehennom', 'Castle', 'Astral Plane', 'Dungeons of Doom',
                         'Gnomish Mines', 'Valley of the Dead', 'Medusa level'],
            'monsters': ['Wizard of Yendor', 'Medusa', 'Croesus', 'Vlad the Impaler',
                        'Rodney', 'Death', 'Pestilence', 'Famine'],
            'items': ['wand of wishing', 'bag of holding', 'amulet of life saving',
                     'ring of conflict', 'boots of speed', 'gauntlets of power'],
        }
    
    def extract_from_text(self, text: str, source: str = "") -> List[Dict]:
        """Extract triples with game-specific enhancements"""
        
        # First, apply pattern-based extraction for game-specific info
        pattern_triples = self._extract_pattern_based(text, source)
        
        # Then use REBEL for general extraction
        rebel_triples = self._extract_with_rebel(text, source)
        
        # Combine and enhance triples
        all_triples = self._combine_and_enhance(pattern_triples, rebel_triples, text)
        
        return all_triples
    
    def _extract_pattern_based(self, text: str, source: str) -> List[Dict]:
        """Extract game-specific patterns"""
        triples = []
        
        for pattern, relation_type in self.game_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    subject = match.group(1)
                    object = match.group(2)
                    
                    # Clean up the extracted entities
                    subject = self._clean_entity(subject)
                    object = self._clean_entity(object)
                    
                    if subject and object:
                        triples.append({
                            'head': subject,
                            'type': relation_type,
                            'tail': object,
                            'source': source
                        })
        
        return triples
    
    def _extract_with_rebel(self, text: str, source: str) -> List[Dict]:
        """Extract using REBEL model with game context"""
        triples = []
        
        # Add game context to help REBEL
        context_text = f"In the game NetHack: {text}"
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                context_text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    num_beams=5,
                    max_length=256,
                    num_return_sequences=3,  # Get multiple interpretations
                    do_sample=False
                )
            
            # Decode
            decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            
            for pred in decoded_preds:
                extracted = self._parse_rebel_output(pred)
                for triple in extracted:
                    # Enhance relation names
                    enhanced_relation = self._enhance_relation(triple['type'], text)
                    triple['type'] = enhanced_relation
                    triple['source'] = source
                    triples.append(triple)
        
        except Exception as e:
            logger.error(f"Error in REBEL extraction: {e}")
        
        return triples
    
    def _enhance_relation(self, relation: str, context: str) -> str:
        """Enhance generic relations with game-specific context"""
        
        relation_lower = relation.lower()
        context_lower = context.lower()
        
        # Check for game-specific contexts
        if 'damage' in context_lower and relation in ['has', 'property']:
            return 'damage'
        elif 'resistance' in context_lower and relation in ['has', 'property', 'grants']:
            return 'grants_resistance'
        elif 'armor class' in context_lower or 'AC' in context:
            return 'armor_class'
        elif any(word in context_lower for word in ['weighs', 'weight']):
            return 'weight'
        elif 'zorkmids' in context_lower:
            return 'cost'
        elif any(word in context_lower for word in ['blessed', 'cursed', 'uncursed']):
            return 'beatitude'
        elif 'found in' in context_lower or 'appears in' in context_lower:
            return 'found_in'
        elif 'quest' in context_lower and 'artifact' in context_lower:
            return 'quest_artifact'
        
        # Default enhancement based on relation type
        enhancements = {
            'instance of': 'item_type',
            'subclass of': 'item_category',
            'part of': 'found_in',
            'has part': 'contains',
            'used by': 'wielded_by',
            'uses': 'requires',
            'location': 'found_in',
        }
        
        return enhancements.get(relation_lower, relation)
    
    def _combine_and_enhance(self, pattern_triples: List[Dict], 
                             rebel_triples: List[Dict], 
                             text: str) -> List[Dict]:
        """Combine and enhance all extracted triples"""
        
        # Remove duplicates while preserving game-specific relations
        seen = set()
        combined = []
        
        # Prioritize pattern-based triples (more game-specific)
        for triple in pattern_triples:
            key = (triple['head'].lower(), triple['type'], triple['tail'].lower())
            if key not in seen:
                seen.add(key)
                combined.append(triple)
        
        # Add REBEL triples that aren't duplicates
        for triple in rebel_triples:
            key = (triple['head'].lower(), triple['type'], triple['tail'].lower())
            if key not in seen:
                seen.add(key)
                combined.append(triple)
        
        # Post-process to add game context
        enhanced = []
        for triple in combined:
            # Skip if entities are too generic
            if len(triple['head']) < 2 or len(triple['tail']) < 2:
                continue
            
            # Enhance important entities
            if any(triple['head'] in category for category in self.important_entities.values()):
                triple['importance'] = 'high'
            
            enhanced.append(triple)
        
        return enhanced
    
    def _clean_entity(self, entity: str) -> str:
        """Clean extracted entity"""
        # Remove common articles and clean up
        entity = re.sub(r'^(the|a|an)\s+', '', entity, flags=re.IGNORECASE)
        entity = entity.strip()
        
        # Capitalize properly for NetHack entities
        if entity.lower() in ['nethack', 'sokoban', 'gehennom']:
            return entity.capitalize()
        
        return entity
    
    def _parse_rebel_output(self, text: str) -> List[Dict]:
        """Parse REBEL output with game awareness"""
        triplets = []
        relation = ''
        
        current = {
            'head': '',
            'type': '',
            'tail': ''
        }
        
        for token in text.strip().split():
            if token == "<triplet>":
                current = {'head': '', 'type': '', 'tail': ''}
            elif token == "<subj>":
                current['head'] = ''
            elif token == "<obj>":
                relation = current['type']
                current['tail'] = ''
            elif token.startswith("<") and token.endswith(">"):
                if current['head'] and current['tail']:
                    current['type'] = relation
                    triplets.append(current.copy())
                relation = ''
            else:
                if not current['type']:
                    current['head'] += ' ' + token
                else:
                    current['tail'] += ' ' + token
                current['head'] = current['head'].strip()
                current['tail'] = current['tail'].strip()
                
                if not relation:
                    relation = token.replace('_', ' ')
                else:
                    relation += ' ' + token.replace('_', ' ')
        
        # Final check
        if current['head'] and current['tail'] and relation:
            current['type'] = relation.strip()
            triplets.append(current)
        
        return triplets
    
    def process_nethack_wiki(self, articles_path: str, output_path: str):
        """Process NetHack wiki articles with improved extraction"""
        
        logger.info(f"Loading articles from {articles_path}")
        with open(articles_path, 'r') as f:
            articles = json.load(f)
        
        all_triples = []
        
        logger.info(f"Processing {len(articles)} articles with improved extraction")
        
        for article in tqdm(articles, desc="Extracting knowledge"):
            title = article.get('title', '')
            text = article.get('text', '')
            
            if not text or len(text) < 50:
                continue
            
            # Extract triples
            triples = self.extract_from_text(text, title)
            all_triples.extend(triples)
        
        # Save results
        logger.info(f"Extracted {len(all_triples)} triples")
        
        # Convert to TSV format
        import pandas as pd
        df = pd.DataFrame(all_triples)
        df = df[['head', 'type', 'tail', 'source']]
        df.to_csv(output_path, sep='\t', index=False)
        
        logger.info(f"Saved to {output_path}")
        
        # Save statistics
        self._save_statistics(all_triples, output_path)
    
    def _save_statistics(self, triples: List[Dict], output_path: str):
        """Save extraction statistics"""
        stats = {
            'total_triples': len(triples),
            'unique_entities': len(set([t['head'] for t in triples] + [t['tail'] for t in triples])),
            'unique_relations': len(set([t['type'] for t in triples])),
            'relation_distribution': defaultdict(int),
            'game_specific_relations': defaultdict(int)
        }
        
        game_relations = ['damage', 'armor_class', 'grants_resistance', 'weight', 
                         'cost', 'material', 'found_in', 'quest_artifact', 'drops',
                         'alignment', 'difficulty', 'attack_type']
        
        for triple in triples:
            rel = triple['type']
            stats['relation_distribution'][rel] += 1
            
            if any(gr in rel.lower() for gr in game_relations):
                stats['game_specific_relations'][rel] += 1
        
        # Convert defaultdicts to regular dicts for JSON
        stats['relation_distribution'] = dict(stats['relation_distribution'])
        stats['game_specific_relations'] = dict(stats['game_specific_relations'])
        
        stats_path = output_path.replace('.tsv', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_path}")


def main():
    extractor = ImprovedNetHackKGExtractor()
    
    # Process first 100 pages
    input_path = "data/nethack_wiki_first_100.json"
    output_path = "data/nethack_kg_improved_100.tsv"
    
    if os.path.exists(input_path):
        extractor.process_nethack_wiki(input_path, output_path)
    else:
        logger.error(f"Articles not found at {input_path}")


if __name__ == "__main__":
    main()