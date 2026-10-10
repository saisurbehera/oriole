#!/usr/bin/env python3
"""
Final NetHack KG extraction combining template parsing, patterns, and REBEL
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinalNetHackExtractor:
    def __init__(self, use_rebel: bool = True):
        self.use_rebel = use_rebel
        
        if use_rebel:
            logger.info("Loading REBEL model...")
            self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Using GPU")
    
    def extract_from_article(self, title: str, text: str) -> List[Dict]:
        """Extract triples from article using all methods"""
        all_triples = []
        
        # 1. Parse wiki templates
        template_triples = self._extract_from_templates(title, text)
        all_triples.extend(template_triples)
        
        # 2. Extract from article prose  
        prose_triples = self._extract_from_prose(title, text)
        all_triples.extend(prose_triples)
        
        # 3. Use REBEL on key sections
        if self.use_rebel and len(text) > 200:
            rebel_triples = self._extract_with_rebel(title, text)
            all_triples.extend(rebel_triples)
        
        # Deduplicate
        seen = set()
        unique_triples = []
        for triple in all_triples:
            key = (triple['head'].lower(), triple['type'], triple['tail'].lower())
            if key not in seen:
                seen.add(key)
                triple['source'] = title
                unique_triples.append(triple)
        
        return unique_triples
    
    def _extract_from_templates(self, title: str, text: str) -> List[Dict]:
        """Extract from wiki templates"""
        triples = []
        
        # Monster template
        monster_data = self._parse_monster_template(text)
        if monster_data:
            name = monster_data.get('name', title)
            
            # Basic stats
            if 'AC' in monster_data:
                triples.append({'head': name, 'type': 'armor_class', 'tail': monster_data['AC']})
            if 'difficulty' in monster_data:
                triples.append({'head': name, 'type': 'difficulty', 'tail': monster_data['difficulty']})
            if 'level' in monster_data:
                triples.append({'head': name, 'type': 'base_level', 'tail': monster_data['level']})
            if 'speed' in monster_data:
                triples.append({'head': name, 'type': 'speed', 'tail': monster_data['speed']})
            if 'weight' in monster_data:
                triples.append({'head': name, 'type': 'weight', 'tail': monster_data['weight']})
            
            # Alignment
            if 'align' in monster_data:
                align_val = monster_data['align']
                try:
                    if int(align_val) == 0:
                        triples.append({'head': name, 'type': 'alignment', 'tail': 'neutral'})
                    elif int(align_val) > 0:
                        triples.append({'head': name, 'type': 'alignment', 'tail': 'lawful'})
                    else:
                        triples.append({'head': name, 'type': 'alignment', 'tail': 'chaotic'})
                except:
                    pass
            
            # Parse attacks
            if 'attacks' in monster_data:
                self._parse_attacks(name, monster_data['attacks'], triples)
            
            # Parse resistances
            if 'resistances' in monster_data:
                self._parse_resistances(name, monster_data['resistances'], triples, 'has_resistance')
            if 'resistances conveyed' in monster_data:
                self._parse_resistances(name, monster_data['resistances conveyed'], triples, 'grants_resistance')
        
        # Item template
        item_data = self._parse_item_template(text)
        if item_data:
            self._extract_item_data(title, item_data, triples)
        
        return triples
    
    def _parse_monster_template(self, text: str) -> Dict[str, str]:
        """Parse monster template"""
        data = {}
        
        template_match = re.search(r'\{\{monster\s*\n([^}]+)\}\}', text, re.IGNORECASE | re.DOTALL)
        if not template_match:
            return data
        
        template_content = template_match.group(1)
        
        for line in template_content.split('\n'):
            if '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip().strip('|')
                    value = parts[1].strip()
                    
                    # Clean wiki markup
                    value = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', value)
                    value = re.sub(r'\{\{[^}]+\}\}', '', value)
                    
                    data[key] = value
        
        return data
    
    def _parse_item_template(self, text: str) -> Dict[str, str]:
        """Parse item/weapon/armor templates"""
        data = {}
        
        # Try different template types
        for template_name in ['weapon', 'armor', 'item', 'amulet', 'wand', 'scroll', 'potion']:
            pattern = rf'\{{\{{{template_name}\s*\n([^}}]+)\}}\}}'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                template_content = match.group(1)
                
                for line in template_content.split('\n'):
                    if '=' in line:
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            key = parts[0].strip().strip('|')
                            value = parts[1].strip()
                            value = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', value)
                            data[key] = value
                break
        
        return data
    
    def _parse_attacks(self, name: str, attacks_str: str, triples: List[Dict]):
        """Parse attack string"""
        # Pattern for attacks like "Bite 2d4 physical, Sting 3d4 poison"
        attack_pattern = r'(\w+)\s+(\d+d\d+)\s+(\w+)'
        
        for match in re.finditer(attack_pattern, attacks_str):
            attack_type = match.group(1).lower()
            damage = match.group(2)
            damage_type = match.group(3).lower()
            
            triples.append({'head': name, 'type': 'attack_type', 'tail': attack_type})
            triples.append({'head': name, 'type': 'attack_damage', 'tail': damage})
            if damage_type not in ['physical', 'damage']:
                triples.append({'head': name, 'type': 'damage_type', 'tail': damage_type})
    
    def _parse_resistances(self, name: str, res_str: str, triples: List[Dict], relation: str):
        """Parse resistance string"""
        # Extract resistance types
        res_pattern = r'(\w+)\s+resistance'
        for match in re.finditer(res_pattern, res_str, re.IGNORECASE):
            res_type = match.group(1).lower()
            triples.append({'head': name, 'type': relation, 'tail': f"{res_type} resistance"})
    
    def _extract_item_data(self, name: str, data: Dict[str, str], triples: List[Dict]):
        """Extract item properties"""
        if 'damage' in data:
            triples.append({'head': name, 'type': 'damage', 'tail': data['damage']})
        if 'tohit' in data:
            triples.append({'head': name, 'type': 'to_hit', 'tail': data['tohit']})
        if 'ac' in data:
            triples.append({'head': name, 'type': 'armor_class', 'tail': data['ac']})
        if 'weight' in data:
            triples.append({'head': name, 'type': 'weight', 'tail': data['weight']})
        if 'cost' in data:
            triples.append({'head': name, 'type': 'cost', 'tail': data['cost']})
        if 'material' in data:
            triples.append({'head': name, 'type': 'material', 'tail': data['material']})
    
    def _extract_from_prose(self, title: str, text: str) -> List[Dict]:
        """Extract from article prose using patterns"""
        triples = []
        
        # Clean text
        cleaned_text = re.sub(r'\{\{[^}]+\}\}', '', text)
        cleaned_text = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', cleaned_text)
        
        # High-precision patterns
        patterns = [
            # Level ranges
            (r'found (?:from |on |between )?(?:dungeon )?levels? (\d+)(?:\s*[-–to]\s*(\d+))?',
             lambda m: (title, 'found_on_level', f"{m.group(1)}-{m.group(2) or m.group(1)}")),
            
            # Eating effects
            (r'eating (?:a |an |the )?(.+?) (?:corpse |tin )(?:will |can |may )?(.+?)(?:\.|,)',
             lambda m: (m.group(1), 'eating_effect', m.group(2))),
            
            # Resistance granting
            (r'(?:grants?|confers?|provides?) (\w+) resistance',
             lambda m: (title, 'grants_resistance', f"{m.group(1).lower()} resistance")),
            
            # Cost in zorkmids
            (r'costs? (\d+) zorkmids?',
             lambda m: (title, 'cost', f"{m.group(1)} zorkmids")),
            
            # Weight
            (r'weighs? (\d+)',
             lambda m: (title, 'weight', m.group(1))),
        ]
        
        for pattern, extractor in patterns:
            for match in re.finditer(pattern, cleaned_text, re.IGNORECASE):
                try:
                    result = extractor(match)
                    if result and all(result):
                        head, rel, tail = result
                        if len(head) > 1 and len(tail) > 1:
                            triples.append({'head': head, 'type': rel, 'tail': tail})
                except:
                    continue
        
        return triples
    
    def _extract_with_rebel(self, title: str, text: str) -> List[Dict]:
        """Extract using REBEL on key sections"""
        triples = []
        
        # Clean text
        cleaned_text = re.sub(r'\{\{[^}]+\}\}', '', text)
        cleaned_text = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', cleaned_text)
        
        # Extract first meaningful paragraph
        paragraphs = cleaned_text.split('\n\n')
        for para in paragraphs:
            if len(para) > 100 and not para.startswith('|'):
                # Add context
                context = f"In NetHack, {para[:300]}"
                
                try:
                    inputs = self.tokenizer(
                        context,
                        max_length=256,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            num_beams=3,
                            max_length=200,
                            do_sample=False
                        )
                    
                    decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
                    
                    for pred in decoded_preds:
                        extracted = self._parse_rebel_output(pred)
                        for triple in extracted:
                            if self._validate_triple(triple['head'], triple['type'], triple['tail']):
                                triples.append(triple)
                    
                    break  # Only process first good paragraph
                
                except Exception as e:
                    logger.error(f"REBEL error: {e}")
        
        return triples
    
    def _parse_rebel_output(self, text: str) -> List[Dict]:
        """Parse REBEL output"""
        triplets = []
        relation = ''
        
        current = {'head': '', 'type': '', 'tail': ''}
        
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
                    current['type'] = relation.strip()
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
        # Length check
        if len(head) < 2 or len(tail) < 2:
            return False
        
        # Skip generic entities
        generic = {'it', 'they', 'you', 'we', 'that', 'which', 'this', 'these', 'those'}
        if head.lower() in generic or tail.lower() in generic:
            return False
        
        # Skip overly long entities
        if len(head.split()) > 8 or len(tail.split()) > 8:
            return False
        
        return True
    
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
            
            # Skip meta pages
            if any(skip in title for skip in [':', 'MediaWiki', 'Template']):
                continue
            
            if len(text) > 50:
                triples = self.extract_from_article(title, text)
                all_triples.extend(triples)
        
        logger.info(f"Extracted {len(all_triples)} triples")
        
        # Save to TSV
        df = pd.DataFrame(all_triples)
        df = df[['head', 'type', 'tail', 'source']]
        df.to_csv(output_path, sep='\t', index=False)
        
        # Save statistics
        self._save_statistics(all_triples, output_path)
    
    def _save_statistics(self, triples: List[Dict], output_path: str):
        """Save extraction statistics"""
        stats = {
            'total_triples': len(triples),
            'unique_entities': len(set([t['head'] for t in triples] + [t['tail'] for t in triples])),
            'unique_relations': len(set([t['type'] for t in triples])),
            'relation_distribution': defaultdict(int),
            'top_entities': defaultdict(int)
        }
        
        for triple in triples:
            stats['relation_distribution'][triple['type']] += 1
            stats['top_entities'][triple['head']] += 1
            stats['top_entities'][triple['tail']] += 1
        
        # Convert to regular dicts
        stats['relation_distribution'] = dict(sorted(
            stats['relation_distribution'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        stats['top_entities'] = dict(sorted(
            stats['top_entities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20])
        
        stats_path = output_path.replace('.tsv', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_path}")


def main():
    # Use REBEL for more comprehensive extraction
    extractor = FinalNetHackExtractor(use_rebel=True)
    
    # Process first 100 pages
    input_path = "data/nethack_wiki_first_100.json"
    output_path = "data/nethack_kg_final_100.tsv"
    
    extractor.process_articles(input_path, output_path)


if __name__ == "__main__":
    main()