#!/usr/bin/env python3
"""
Combined extraction approach using both REBEL and patterns
with better filtering and validation
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CombinedNetHackExtractor:
    def __init__(self):
        # Load REBEL model
        logger.info("Loading REBEL model...")
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Using GPU")
        
        # Initialize patterns and filters
        self.init_nethack_knowledge()
    
    def init_nethack_knowledge(self):
        """Initialize NetHack-specific knowledge"""
        
        # Known important entities
        self.important_entities = {
            'Amulet of Yendor', 'NetHack', 'Sokoban', 'Gehennom', 'Excalibur',
            'Mjollnir', 'Stormbringer', 'Wizard of Yendor', 'Medusa', 'Castle',
            'Gnomish Mines', 'Valley of the Dead', 'Astral Plane', 'Dungeons of Doom',
            'wand of wishing', 'bag of holding', 'blessed', 'cursed', 'uncursed'
        }
        
        # Game-specific patterns with high precision
        self.high_precision_patterns = [
            # Damage patterns
            (r'(?:deals?|does?|inflicts?)\s+(\d+d\d+(?:\s*[+-]\s*\d+)?)\s+(?:points?\s+of\s+)?damage',
             lambda text, m: self._extract_damage(text, m)),
            
            # AC patterns
            (r'(?:AC|armor\s+class)\s+(?:of\s+)?([+-]?\d+)',
             lambda text, m: self._extract_ac(text, m)),
            
            # Weight patterns
            (r'weighs?\s+(\d+)\s*(?:aum|zorkmids?)?',
             lambda text, m: self._extract_weight(text, m)),
            
            # Cost patterns
            (r'costs?\s+(\d+)\s+zorkmids?',
             lambda text, m: self._extract_cost(text, m)),
            
            # Level patterns
            (r'found\s+(?:from\s+)?(?:dungeon\s+)?levels?\s+(\d+)(?:\s*[-–]\s*(\d+))?',
             lambda text, m: self._extract_level(text, m)),
            
            # Resistance patterns
            (r'(?:grants?|confers?|provides?)\s+(\w+)\s+resistance',
             lambda text, m: self._extract_resistance(text, m)),
        ]
        
        # Valid relation types for NetHack
        self.valid_relations = {
            'damage', 'armor_class', 'weight', 'cost', 'material',
            'grants_resistance', 'grants_ability', 'found_in', 'appears_on_level',
            'dropped_by', 'wielded_by', 'quest_artifact_of', 'alignment',
            'difficulty', 'speed', 'attack_type', 'beatitude', 'edible',
            'invoke_effect', 'reading_effect', 'zapping_effect'
        }
    
    def extract_from_article(self, title: str, text: str) -> List[Dict]:
        """Extract triples from a single article"""
        all_triples = []
        
        # Skip if too short or redirect
        if len(text) < 100 or text.startswith('#REDIRECT'):
            return []
        
        # Clean wiki markup
        cleaned_text = self._clean_wiki_text(text)
        
        # 1. Extract with high-precision patterns
        pattern_triples = self._extract_with_patterns(title, cleaned_text)
        
        # 2. Extract with REBEL on key sections
        rebel_triples = self._extract_with_rebel_smart(title, cleaned_text)
        
        # 3. Combine and validate
        combined = self._combine_triples(pattern_triples, rebel_triples, title)
        
        return combined
    
    def _clean_wiki_text(self, text: str) -> str:
        """Clean wiki markup"""
        # Remove templates
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        # Remove wiki links but keep text
        text = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove references
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text)
        
        return text
    
    def _extract_with_patterns(self, title: str, text: str) -> List[Dict]:
        """Extract using high-precision patterns"""
        triples = []
        
        for pattern, extractor in self.high_precision_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                result = extractor(text, match)
                if result:
                    head, relation, tail = result
                    if self._validate_triple(head, relation, tail):
                        triples.append({
                            'head': head,
                            'type': relation,
                            'tail': tail,
                            'confidence': 0.9
                        })
        
        return triples
    
    def _extract_damage(self, text: str, match) -> Optional[Tuple[str, str, str]]:
        """Extract damage information"""
        damage = match.group(1)
        
        # Look for subject before the match
        before_text = text[:match.start()].split('.')[-1]
        
        # Common damage patterns
        weapon_match = re.search(r'(\w+(?:\s+\w+)*?)\s+(?:deals?|does?)', before_text)
        if weapon_match:
            weapon = weapon_match.group(1).strip()
            return (weapon, 'damage', damage)
        
        return None
    
    def _extract_ac(self, text: str, match) -> Optional[Tuple[str, str, str]]:
        """Extract armor class"""
        ac = match.group(1)
        
        # Look for armor/item before
        before_text = text[:match.start()].split('.')[-1]
        
        armor_match = re.search(r'(\w+(?:\s+\w+)*?)\s+(?:has|have|provides?)', before_text)
        if armor_match:
            armor = armor_match.group(1).strip()
            return (armor, 'armor_class', ac)
        
        return None
    
    def _extract_weight(self, text: str, match) -> Optional[Tuple[str, str, str]]:
        """Extract weight information"""
        weight = match.group(1)
        
        before_text = text[:match.start()].split('.')[-1]
        item_match = re.search(r'(\w+(?:\s+\w+)*?)\s+weighs?', before_text)
        if item_match:
            item = item_match.group(1).strip()
            return (item, 'weight', weight)
        
        return None
    
    def _extract_cost(self, text: str, match) -> Optional[Tuple[str, str, str]]:
        """Extract cost information"""
        cost = match.group(1)
        
        before_text = text[:match.start()].split('.')[-1]
        item_match = re.search(r'(\w+(?:\s+\w+)*?)\s+costs?', before_text)
        if item_match:
            item = item_match.group(1).strip()
            return (item, 'cost', f"{cost} zorkmids")
        
        return None
    
    def _extract_level(self, text: str, match) -> Optional[Tuple[str, str, str]]:
        """Extract level information"""
        start_level = match.group(1)
        end_level = match.group(2) if match.group(2) else start_level
        
        before_text = text[:match.start()].split('.')[-1]
        entity_match = re.search(r'(\w+(?:\s+\w+)*?)\s+(?:is|are|can\s+be)\s+found', before_text)
        if entity_match:
            entity = entity_match.group(1).strip()
            return (entity, 'appears_on_level', f"{start_level}-{end_level}")
        
        return None
    
    def _extract_resistance(self, text: str, match) -> Optional[Tuple[str, str, str]]:
        """Extract resistance information"""
        resistance = match.group(1)
        
        before_text = text[:match.start()].split('.')[-1]
        item_match = re.search(r'(\w+(?:\s+\w+)*?)\s+(?:grants?|confers?)', before_text)
        if item_match:
            item = item_match.group(1).strip()
            return (item, 'grants_resistance', f"{resistance} resistance")
        
        return None
    
    def _extract_with_rebel_smart(self, title: str, text: str) -> List[Dict]:
        """Extract using REBEL on key sections"""
        triples = []
        
        # Extract first paragraph (usually definition)
        paragraphs = text.split('\n\n')
        if paragraphs:
            first_para = paragraphs[0][:500]  # Limit length
            
            # Add context
            if any(entity in title for entity in self.important_entities):
                context = f"{title} is an important element in NetHack. {first_para}"
            else:
                context = f"In NetHack, {first_para}"
            
            rebel_triples = self._run_rebel(context)
            triples.extend(rebel_triples)
        
        # Look for specific sections
        section_patterns = [
            (r'==\s*Properties?\s*==([^=]+)', 'property'),
            (r'==\s*Effects?\s*==([^=]+)', 'effect'),
            (r'==\s*Strategy\s*==([^=]+)', 'strategy'),
            (r'==\s*Generation\s*==([^=]+)', 'generation')
        ]
        
        for pattern, section_type in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1)[:500]
                context = f"About {title} in NetHack: {section_text}"
                section_triples = self._run_rebel(context)
                triples.extend(section_triples)
        
        return triples
    
    def _run_rebel(self, text: str) -> List[Dict]:
        """Run REBEL model on text"""
        triples = []
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                max_length=256,
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
                    num_beams=3,
                    max_length=200,
                    do_sample=False
                )
            
            # Decode
            decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            
            for pred in decoded_preds:
                extracted = self._parse_rebel_output(pred)
                for triple in extracted:
                    if self._validate_triple(triple['head'], triple['type'], triple['tail']):
                        triple['confidence'] = 0.7
                        triples.append(triple)
        
        except Exception as e:
            logger.error(f"REBEL error: {e}")
        
        return triples
    
    def _parse_rebel_output(self, text: str) -> List[Dict]:
        """Parse REBEL output"""
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
        
        return triplets
    
    def _validate_triple(self, head: str, relation: str, tail: str) -> bool:
        """Validate triple quality"""
        # Check length
        if len(head) < 2 or len(tail) < 2:
            return False
        
        # Skip generic entities
        generic_entities = {'it', 'they', 'you', 'we', 'that', 'which', 'who', 'what'}
        if head.lower() in generic_entities or tail.lower() in generic_entities:
            return False
        
        # Skip if too long (likely sentence fragments)
        if len(head.split()) > 10 or len(tail.split()) > 10:
            return False
        
        return True
    
    def _combine_triples(self, pattern_triples: List[Dict], 
                        rebel_triples: List[Dict], 
                        source: str) -> List[Dict]:
        """Combine and deduplicate triples"""
        all_triples = []
        seen = set()
        
        # Add high-confidence pattern triples first
        for triple in pattern_triples:
            key = (triple['head'].lower(), triple['type'], triple['tail'].lower())
            if key not in seen:
                seen.add(key)
                triple['source'] = source
                all_triples.append(triple)
        
        # Add REBEL triples if not duplicates
        for triple in rebel_triples:
            key = (triple['head'].lower(), triple['type'], triple['tail'].lower())
            if key not in seen:
                seen.add(key)
                triple['source'] = source
                all_triples.append(triple)
        
        return all_triples
    
    def process_articles(self, articles_path: str, output_path: str):
        """Process all articles"""
        logger.info(f"Loading articles from {articles_path}")
        
        with open(articles_path, 'r') as f:
            articles = json.load(f)
        
        all_triples = []
        
        logger.info(f"Processing {len(articles)} articles")
        
        for article in tqdm(articles, desc="Extracting knowledge"):
            title = article.get('title', '')
            text = article.get('text', '')
            
            triples = self.extract_from_article(title, text)
            all_triples.extend(triples)
        
        logger.info(f"Extracted {len(all_triples)} triples")
        
        if all_triples:
            # Convert to dataframe
            df = pd.DataFrame(all_triples)
            if 'confidence' in df.columns:
                df = df[['head', 'type', 'tail', 'source', 'confidence']]
            else:
                df = df[['head', 'type', 'tail', 'source']]
            
            df.to_csv(output_path, sep='\t', index=False)
            logger.info(f"Saved to {output_path}")
        else:
            logger.warning("No triples extracted!")
            # Create empty file
            pd.DataFrame(columns=['head', 'type', 'tail', 'source']).to_csv(output_path, sep='\t', index=False)
        
        # Save statistics
        self._save_stats(all_triples, output_path)
    
    def _save_stats(self, triples: List[Dict], output_path: str):
        """Save extraction statistics"""
        stats = {
            'total_triples': len(triples),
            'unique_entities': len(set([t['head'] for t in triples] + [t['tail'] for t in triples])),
            'unique_relations': len(set([t['type'] for t in triples])),
            'relation_distribution': defaultdict(int),
            'high_confidence_triples': sum(1 for t in triples if t.get('confidence', 0) >= 0.8)
        }
        
        for triple in triples:
            stats['relation_distribution'][triple['type']] += 1
        
        stats['relation_distribution'] = dict(stats['relation_distribution'])
        
        stats_path = output_path.replace('.tsv', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_path}")


def main():
    extractor = CombinedNetHackExtractor()
    
    # Process first 100 pages
    input_path = "data/nethack_wiki_first_100.json"
    output_path = "data/nethack_kg_combined_100.tsv"
    
    extractor.process_articles(input_path, output_path)


if __name__ == "__main__":
    main()