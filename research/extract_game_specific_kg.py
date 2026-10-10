#!/usr/bin/env python3
"""
Extract game-specific knowledge from NetHack wiki using domain patterns
This focuses on extracting actionable game information
"""

import json
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameSpecificExtractor:
    def __init__(self):
        self.init_patterns()
        self.stats = defaultdict(int)
    
    def init_patterns(self):
        """Initialize game-specific extraction patterns"""
        
        # Weapon patterns
        self.weapon_patterns = [
            (r'(\w+(?:\s+\w+)*?)\s+(?:deals?|does?|inflicts?)\s+(\d+d\d+(?:\s*[+-]\s*\d+)?)\s+damage', 
             lambda m: (m.group(1), 'damage', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:has|have)\s+(?:a\s+)?([+-]?\d+)\s+(?:to\s+)?(?:hit|accuracy)', 
             lambda m: (m.group(1), 'to_hit', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:is|are)\s+(?:a|an)\s+(two-handed|one-handed)\s+weapon', 
             lambda m: (m.group(1), 'handedness', m.group(2))),
        ]
        
        # Armor patterns
        self.armor_patterns = [
            (r'(\w+(?:\s+\w+)*?)\s+(?:provides?|grants?|gives?)\s+(?:AC|armor\s+class)\s+([+-]?\d+)', 
             lambda m: (m.group(1), 'armor_class', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:provides?|grants?|confers?)\s+(\w+)\s+resistance', 
             lambda m: (m.group(1), 'grants_resistance', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:weighs?|has\s+weight)\s+(\d+)\s*(?:aum)?', 
             lambda m: (m.group(1), 'weight', m.group(2))),
        ]
        
        # Monster patterns
        self.monster_patterns = [
            (r'(\w+(?:\s+\w+)*?)\s+(?:has|have)\s+(?:difficulty|level)\s+(\d+)', 
             lambda m: (m.group(1), 'difficulty', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:is|are)\s+(lawful|neutral|chaotic)', 
             lambda m: (m.group(1), 'alignment', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:can|may|might)\s+drop\s+(\w+(?:\s+\w+)*)', 
             lambda m: (m.group(1), 'can_drop', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:resists?|immune\s+to)\s+(\w+)', 
             lambda m: (m.group(1), 'resists', m.group(2))),
        ]
        
        # Location patterns
        self.location_patterns = [
            (r'(\w+(?:\s+\w+)*?)\s+(?:is|are)\s+(?:found|located)\s+(?:in|on|at)\s+(\w+(?:\s+\w+)*)', 
             lambda m: (m.group(1), 'found_in', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:contains?|has|have)\s+(\w+(?:\s+\w+)*)', 
             lambda m: (m.group(1), 'contains', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:appears?|spawns?)\s+(?:in|on|at)\s+level\s+(\d+(?:-\d+)?)', 
             lambda m: (m.group(1), 'appears_on_level', m.group(2))),
        ]
        
        # Artifact patterns
        self.artifact_patterns = [
            (r'(\w+(?:\s+\w+)*?)\s+is\s+the\s+quest\s+artifact\s+(?:of|for)\s+(?:the\s+)?(\w+)', 
             lambda m: (m.group(1), 'quest_artifact_of', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:grants?|confers?|provides?)\s+(\w+(?:\s+\w+)*)', 
             lambda m: (m.group(1), 'grants_ability', m.group(2))),
            (r'invoking\s+(\w+(?:\s+\w+)*?)\s+(?:will|can|may)\s+(.+)', 
             lambda m: (m.group(1), 'invoke_effect', m.group(2).strip('.'))),
        ]
        
        # Game mechanics patterns
        self.mechanics_patterns = [
            (r'(\w+(?:\s+\w+)*?)\s+(?:cures?|removes?)\s+(\w+(?:\s+\w+)*)', 
             lambda m: (m.group(1), 'cures', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:causes?|inflicts?)\s+(\w+(?:\s+\w+)*)', 
             lambda m: (m.group(1), 'causes', m.group(2))),
            (r'eating\s+(\w+(?:\s+\w+)*?)\s+(?:will|can|may)\s+(.+)', 
             lambda m: (m.group(1), 'eating_effect', m.group(2).strip('.'))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:costs?|priced?\s+at)\s+(\d+)\s+zorkmids?', 
             lambda m: (m.group(1), 'cost', m.group(2))),
        ]
        
        # Special properties
        self.property_patterns = [
            (r'(\w+(?:\s+\w+)*?)\s+(?:is|are)\s+(blessed|cursed|uncursed)', 
             lambda m: (m.group(1), 'beatitude', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:is|are)\s+(?:made\s+of|material:?)\s+(\w+)', 
             lambda m: (m.group(1), 'material', m.group(2))),
            (r'(\w+(?:\s+\w+)*?)\s+(?:is|are)\s+(edible|comestible)', 
             lambda m: (m.group(1), 'property', 'edible')),
        ]
    
    def extract_from_text(self, text: str, title: str) -> List[Tuple[str, str, str]]:
        """Extract game-specific triples from text"""
        triples = []
        
        # Clean text
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # Remove wiki links
        text = re.sub(r'\{\{[^}]+\}\}', '', text)  # Remove templates
        
        # Apply all pattern groups
        pattern_groups = [
            self.weapon_patterns,
            self.armor_patterns,
            self.monster_patterns,
            self.location_patterns,
            self.artifact_patterns,
            self.mechanics_patterns,
            self.property_patterns
        ]
        
        for patterns in pattern_groups:
            for pattern, extractor in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        triple = extractor(match)
                        if triple and all(triple):
                            # Clean entities
                            head = self.clean_entity(triple[0])
                            relation = triple[1]
                            tail = self.clean_entity(triple[2])
                            
                            if head and tail and len(head) > 1 and len(tail) > 1:
                                triples.append((head, relation, tail))
                                self.stats[relation] += 1
                    except:
                        continue
        
        # Extract from infoboxes and structured data
        infobox_triples = self.extract_from_infobox(text, title)
        triples.extend(infobox_triples)
        
        # Extract from tables
        table_triples = self.extract_from_tables(text, title)
        triples.extend(table_triples)
        
        return triples
    
    def extract_from_infobox(self, text: str, title: str) -> List[Tuple[str, str, str]]:
        """Extract from wiki infoboxes"""
        triples = []
        
        # Common infobox patterns
        infobox_patterns = [
            (r'damage[:\s]+(\d+d\d+)', 'damage'),
            (r'to-hit[:\s]+([+-]?\d+)', 'to_hit'),
            (r'AC[:\s]+([+-]?\d+)', 'armor_class'),
            (r'weight[:\s]+(\d+)', 'weight'),
            (r'nutrition[:\s]+(\d+)', 'nutrition'),
            (r'cost[:\s]+(\d+)', 'cost'),
            (r'material[:\s]+(\w+)', 'material'),
            (r'type[:\s]+(\w+(?:\s+\w+)*)', 'item_type'),
        ]
        
        for pattern, relation in infobox_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = self.clean_entity(match.group(1))
                if value:
                    triples.append((title, relation, value))
        
        return triples
    
    def extract_from_tables(self, text: str, title: str) -> List[Tuple[str, str, str]]:
        """Extract from wiki tables"""
        triples = []
        
        # Simple table extraction
        table_pattern = r'\|([^|]+)\|([^|]+)\|'
        matches = re.finditer(table_pattern, text)
        
        for match in matches:
            col1 = self.clean_entity(match.group(1))
            col2 = self.clean_entity(match.group(2))
            
            # Try to determine relation from context
            if col1 and col2:
                if re.match(r'\d+d\d+', col2):
                    triples.append((col1, 'damage', col2))
                elif re.match(r'[+-]?\d+', col2) and 'AC' in text:
                    triples.append((col1, 'armor_class', col2))
                elif col2 in ['lawful', 'neutral', 'chaotic']:
                    triples.append((col1, 'alignment', col2))
        
        return triples
    
    def clean_entity(self, entity: str) -> str:
        """Clean and normalize entity"""
        if not entity:
            return ""
        
        # Remove extra whitespace
        entity = ' '.join(entity.split())
        
        # Remove common prefixes
        entity = re.sub(r'^(the|a|an)\s+', '', entity, flags=re.IGNORECASE)
        
        # Capitalize NetHack entities
        nethack_entities = {
            'nethack': 'NetHack',
            'sokoban': 'Sokoban',
            'gehennom': 'Gehennom',
            'amulet of yendor': 'Amulet of Yendor',
            'excalibur': 'Excalibur',
            'mjollnir': 'Mjollnir',
            'wizard of yendor': 'Wizard of Yendor'
        }
        
        entity_lower = entity.lower()
        if entity_lower in nethack_entities:
            return nethack_entities[entity_lower]
        
        return entity.strip()
    
    def process_articles(self, articles_path: str, output_path: str):
        """Process all articles"""
        logger.info(f"Loading articles from {articles_path}")
        
        with open(articles_path, 'r') as f:
            articles = json.load(f)
        
        all_triples = []
        
        logger.info(f"Processing {len(articles)} articles")
        
        for article in tqdm(articles, desc="Extracting game knowledge"):
            title = article.get('title', '')
            text = article.get('text', '')
            
            if not text:
                continue
            
            # Extract triples
            triples = self.extract_from_text(text, title)
            
            # Add source information
            for head, rel, tail in triples:
                all_triples.append({
                    'head': head,
                    'relation': rel,
                    'tail': tail,
                    'source': title
                })
        
        # Remove duplicates
        seen = set()
        unique_triples = []
        for triple in all_triples:
            key = (triple['head'], triple['relation'], triple['tail'])
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)
        
        logger.info(f"Extracted {len(unique_triples)} unique triples")
        
        # Save to TSV
        df = pd.DataFrame(unique_triples)
        df.to_csv(output_path, sep='\t', index=False)
        
        # Save statistics
        self.save_statistics(unique_triples, output_path)
    
    def save_statistics(self, triples: List[Dict], output_path: str):
        """Save extraction statistics"""
        stats = {
            'total_triples': len(triples),
            'unique_entities': len(set([t['head'] for t in triples] + [t['tail'] for t in triples])),
            'unique_relations': len(set([t['relation'] for t in triples])),
            'relation_counts': dict(self.stats),
            'top_entities': defaultdict(int)
        }
        
        # Count entity occurrences
        for triple in triples:
            stats['top_entities'][triple['head']] += 1
            stats['top_entities'][triple['tail']] += 1
        
        # Get top entities
        stats['top_entities'] = dict(sorted(
            stats['top_entities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20])
        
        stats_path = output_path.replace('.tsv', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_path}")
        logger.info(f"Game-specific relations extracted: {dict(self.stats)}")


def main():
    extractor = GameSpecificExtractor()
    
    # Process first 100 pages
    input_path = "data/nethack_wiki_first_100.json"
    output_path = "data/nethack_kg_game_specific_100.tsv"
    
    extractor.process_articles(input_path, output_path)


if __name__ == "__main__":
    main()