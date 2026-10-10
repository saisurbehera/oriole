#!/usr/bin/env python3
"""
Balanced extraction - faster but still comprehensive
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


class BalancedNetHackExtractor:
    def __init__(self):
        logger.info("Loading REBEL model...")
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Using GPU")
    
    def extract_from_article(self, title: str, text: str) -> List[Dict]:
        """Extract triples from article"""
        all_triples = []
        
        # 1. Template extraction (fast and accurate)
        template_triples = self._extract_from_templates(title, text)
        all_triples.extend(template_triples)
        
        # 2. Pattern extraction (fast)
        pattern_triples = self._extract_patterns(title, text)
        all_triples.extend(pattern_triples)
        
        # 3. REBEL on key sentences only
        if len(text) > 100:
            rebel_triples = self._extract_with_rebel_selective(title, text)
            all_triples.extend(rebel_triples)
        
        # Deduplicate
        seen = set()
        unique_triples = []
        for triple in all_triples:
            key = (str(triple.get('head', '')).lower(), 
                   triple.get('type', ''), 
                   str(triple.get('tail', '')).lower())
            if all(key) and key not in seen and len(triple['head']) > 1 and len(triple['tail']) > 1:
                seen.add(key)
                triple['source'] = title
                unique_triples.append(triple)
        
        return unique_triples
    
    def _extract_from_templates(self, title: str, text: str) -> List[Dict]:
        """Extract from wiki templates"""
        triples = []
        
        # All template types
        template_patterns = [
            (r'\{\{monster\s*\n([^}]+)\}\}', 'monster'),
            (r'\{\{item\s*\n([^}]+)\}\}', 'item'),
            (r'\{\{weapon\s*\n([^}]+)\}\}', 'weapon'),
            (r'\{\{armor\s*\n([^}]+)\}\}', 'armor'),
            (r'\{\{spell\s*\n([^}]+)\}\}', 'spell'),
        ]
        
        for pattern, template_type in template_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                template_content = match.group(1)
                
                # Parse key-value pairs
                for line in template_content.split('\n'):
                    if '=' in line:
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            key = parts[0].strip().strip('|')
                            value = parts[1].strip()
                            
                            # Clean value
                            value = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', value)
                            value = re.sub(r'\{\{[^}]+\}\}', '', value).strip()
                            
                            if value and len(value) < 100:
                                # Map common fields
                                if key in ['name', 'title']:
                                    name = value
                                    if name != title:
                                        triples.append({'head': name, 'type': 'also_known_as', 'tail': title})
                                elif key in ['AC', 'ac', 'armor']:
                                    triples.append({'head': title, 'type': 'armor_class', 'tail': value})
                                elif key == 'difficulty':
                                    triples.append({'head': title, 'type': 'difficulty', 'tail': value})
                                elif key == 'level':
                                    triples.append({'head': title, 'type': 'base_level', 'tail': value})
                                elif key == 'speed':
                                    triples.append({'head': title, 'type': 'speed', 'tail': value})
                                elif key == 'weight':
                                    triples.append({'head': title, 'type': 'weight', 'tail': value})
                                elif key == 'cost':
                                    triples.append({'head': title, 'type': 'cost', 'tail': value})
                                elif key == 'damage':
                                    triples.append({'head': title, 'type': 'damage', 'tail': value})
                                elif key == 'material':
                                    triples.append({'head': title, 'type': 'material', 'tail': value})
                                elif key == 'color':
                                    triples.append({'head': title, 'type': 'color', 'tail': value})
                                elif key in ['genocidable', 'unique']:
                                    if value.lower() in ['yes', 'true', '1']:
                                        triples.append({'head': title, 'type': 'property', 'tail': key})
                                elif key == 'size':
                                    triples.append({'head': title, 'type': 'size', 'tail': value})
                                elif key == 'nutr':
                                    triples.append({'head': title, 'type': 'nutrition', 'tail': value})
                                elif key == 'align':
                                    align_map = {'0': 'neutral', '-': 'chaotic', '+': 'lawful'}
                                    alignment = align_map.get(value[0], value)
                                    triples.append({'head': title, 'type': 'alignment', 'tail': alignment})
                                elif key == 'attacks':
                                    # Parse attack string
                                    attack_pattern = r'(\w+)\s+(\d+d\d+)\s*(\w*)'
                                    for match in re.finditer(attack_pattern, value):
                                        attack_type = match.group(1).lower()
                                        damage = match.group(2)
                                        damage_type = match.group(3).lower() if match.group(3) else 'physical'
                                        
                                        triples.append({'head': title, 'type': 'attack_type', 'tail': attack_type})
                                        triples.append({'head': title, 'type': 'attack_damage', 'tail': damage})
                                        if damage_type != 'physical':
                                            triples.append({'head': title, 'type': 'damage_type', 'tail': damage_type})
                                elif key in ['resistances', 'resists']:
                                    res_pattern = r'(\w+)\s*(?:resistance)?'
                                    for match in re.finditer(res_pattern, value, re.IGNORECASE):
                                        res = match.group(1).lower()
                                        if res:
                                            triples.append({'head': title, 'type': 'has_resistance', 'tail': f"{res} resistance"})
                                elif key == 'resistances conveyed':
                                    res_pattern = r'(\w+)\s*(?:resistance)?'
                                    for match in re.finditer(res_pattern, value, re.IGNORECASE):
                                        res = match.group(1).lower()
                                        if res:
                                            triples.append({'head': title, 'type': 'grants_resistance', 'tail': f"{res} resistance"})
        
        return triples
    
    def _extract_patterns(self, title: str, text: str) -> List[Dict]:
        """Fast pattern-based extraction"""
        triples = []
        
        # Clean text
        cleaned = re.sub(r'\{\{[^}]+\}\}', ' ', text)
        cleaned = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', cleaned)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Key patterns
        patterns = [
            # Type patterns
            (r'(?:is|are)\s+(?:a|an)\s+type\s+of\s+(\w+(?:\s+\w+)*)', 
             lambda m: {'head': title, 'type': 'type_of', 'tail': m.group(1).strip()}),
            
            # Location patterns
            (r'(?:found|appears?|generated)\s+(?:in|on|at)\s+(.+?)(?:[,.]|$)',
             lambda m: {'head': title, 'type': 'found_in', 'tail': m.group(1).strip()[:50]}),
            
            # Level range patterns
            (r'(?:from\s+)?(?:dungeon\s+)?levels?\s+(\d+)(?:\s*[-–to]\s*(\d+))?',
             lambda m: {'head': title, 'type': 'appears_on_level', 'tail': f"{m.group(1)}-{m.group(2) or m.group(1)}"}),
            
            # Effect patterns
            (r'(?:grants?|confers?|provides?)\s+(.+?)(?:\s+(?:when|if|to)|[,.]|$)',
             lambda m: {'head': title, 'type': 'grants', 'tail': m.group(1).strip()[:50]}),
            
            # Drop patterns
            (r'(?:drops?|leaves?|may\s+drop)\s+(.+?)(?:[,.]|$)',
             lambda m: {'head': title, 'type': 'drops', 'tail': m.group(1).strip()[:50]}),
            
            # Eating effects
            (r'eating\s+(?:a|an|the)?\s*(?:corpse|tin)?\s*(?:will|can|may)?\s*(.+?)(?:[,.]|$)',
             lambda m: {'head': title, 'type': 'eating_effect', 'tail': m.group(1).strip()[:50]}),
            
            # Cost patterns
            (r'costs?\s+(\d+)\s+zorkmids?',
             lambda m: {'head': title, 'type': 'cost', 'tail': f"{m.group(1)} zorkmids"}),
            
            # Weight patterns
            (r'weighs?\s+(\d+)(?:\s+aum)?',
             lambda m: {'head': title, 'type': 'weight', 'tail': m.group(1)}),
            
            # Part of patterns
            (r'(?:is|are)\s+(?:part\s+of|found\s+in|located\s+in)\s+(.+?)(?:[,.]|$)',
             lambda m: {'head': title, 'type': 'part_of', 'tail': m.group(1).strip()[:50]}),
        ]
        
        for pattern, extractor in patterns:
            for match in re.finditer(pattern, cleaned, re.IGNORECASE):
                try:
                    triple = extractor(match)
                    if triple:
                        # Basic validation
                        if len(triple['tail']) > 2 and not triple['tail'].isspace():
                            triples.append(triple)
                except:
                    continue
        
        return triples
    
    def _extract_with_rebel_selective(self, title: str, text: str) -> List[Dict]:
        """Use REBEL selectively on key sentences"""
        triples = []
        
        # Clean text
        cleaned = re.sub(r'\{\{[^}]+\}\}', ' ', text)
        cleaned = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', cleaned)
        
        # Find key sentences (first paragraph + sentences with title)
        sentences = re.split(r'[.!?]\s+', cleaned)
        
        key_sentences = []
        
        # First 2 sentences
        for i, sent in enumerate(sentences[:2]):
            if 20 < len(sent) < 200:
                key_sentences.append(sent)
        
        # Sentences mentioning the title
        for sent in sentences[2:]:
            if title.lower() in sent.lower() and 20 < len(sent) < 200:
                key_sentences.append(sent)
                if len(key_sentences) >= 5:  # Limit
                    break
        
        # Process with REBEL
        for sentence in key_sentences:
            # Add minimal context
            context = f"In NetHack, {sentence}"
            
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
                
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
                extracted = self._parse_rebel_output(decoded)
                
                # Filter results
                for triple in extracted:
                    # Replace pronouns with title
                    if triple['head'].lower() in ['it', 'this', 'that', 'they']:
                        triple['head'] = title
                    
                    # Skip if still has pronouns or too generic
                    if triple['head'].lower() in ['it', 'they', 'he', 'she', 'you', 'we']:
                        continue
                    if triple['tail'].lower() in ['it', 'they', 'he', 'she', 'you', 'we']:
                        continue
                    
                    # Skip very long tails
                    if len(triple['tail']) > 50:
                        continue
                    
                    triples.append(triple)
                    
            except Exception as e:
                logger.debug(f"REBEL error: {e}")
                continue
        
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
    
    def process_articles(self, articles_path: str, output_path: str):
        """Process articles"""
        logger.info(f"Loading articles from {articles_path}")
        
        with open(articles_path, 'r') as f:
            articles = json.load(f)
        
        all_triples = []
        
        logger.info(f"Processing {len(articles)} articles")
        
        for article in tqdm(articles, desc="Extracting"):
            title = article.get('title', '')
            text = article.get('text', '')
            
            # Skip meta
            if any(skip in title for skip in [':', 'MediaWiki', 'Template', 'Talk:', 'User:']):
                continue
            
            if len(text) > 50:
                triples = self.extract_from_article(title, text)
                all_triples.extend(triples)
        
        logger.info(f"Extracted {len(all_triples)} triples")
        
        # Save
        df = pd.DataFrame(all_triples)
        df = df[['head', 'type', 'tail', 'source']]
        df.to_csv(output_path, sep='\t', index=False)
        
        # Stats
        stats = {
            'total_triples': len(all_triples),
            'unique_entities': len(set([t['head'] for t in all_triples] + [t['tail'] for t in all_triples])),
            'unique_relations': len(set([t['type'] for t in all_triples])),
            'relation_distribution': defaultdict(int),
            'source_distribution': defaultdict(int)
        }
        
        for triple in all_triples:
            stats['relation_distribution'][triple['type']] += 1
            stats['source_distribution'][triple['source']] += 1
        
        stats['relation_distribution'] = dict(sorted(
            stats['relation_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:20])
        
        stats['source_distribution'] = dict(sorted(
            stats['source_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        stats_path = output_path.replace('.tsv', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Stats saved to {stats_path}")


def main():
    extractor = BalancedNetHackExtractor()
    
    input_path = "data/nethack_wiki_first_100.json"
    output_path = "data/nethack_kg_balanced_100.tsv"
    
    extractor.process_articles(input_path, output_path)


if __name__ == "__main__":
    main()