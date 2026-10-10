#!/usr/bin/env python3
"""
Enhanced REBEL extraction to find more triples
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


class EnhancedNetHackExtractor:
    def __init__(self):
        logger.info("Loading REBEL model...")
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Using GPU")
        
        # NetHack-specific context phrases to help REBEL
        self.context_templates = [
            "In the game NetHack, {}",
            "In NetHack, {}",
            "NetHack information: {}",
            "Game mechanics: {}",
            "{}"  # Also try without context
        ]
    
    def extract_from_article(self, title: str, text: str) -> List[Dict]:
        """Extract triples from article using multiple strategies"""
        all_triples = []
        
        # 1. Parse wiki templates (same as before)
        template_triples = self._extract_from_templates(title, text)
        all_triples.extend(template_triples)
        
        # 2. Extract from prose with REBEL - more aggressive
        prose_triples = self._extract_from_prose_enhanced(title, text)
        all_triples.extend(prose_triples)
        
        # 3. Extract from structured sections
        section_triples = self._extract_from_sections(title, text)
        all_triples.extend(section_triples)
        
        # 4. Pattern-based extraction for missed facts
        pattern_triples = self._extract_with_patterns(title, text)
        all_triples.extend(pattern_triples)
        
        # Deduplicate and validate
        seen = set()
        unique_triples = []
        for triple in all_triples:
            key = (str(triple.get('head', '')).lower(), 
                   triple.get('type', ''), 
                   str(triple.get('tail', '')).lower())
            if all(key) and key not in seen:
                seen.add(key)
                triple['source'] = title
                unique_triples.append(triple)
        
        return unique_triples
    
    def _extract_from_prose_enhanced(self, title: str, text: str) -> List[Dict]:
        """Enhanced REBEL extraction from prose"""
        triples = []
        
        # Clean text
        cleaned_text = re.sub(r'\{\{[^}]+\}\}', '', text)
        cleaned_text = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', cleaned_text)
        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', cleaned_text)
        
        # Process more sentences (not just first paragraph)
        processed = 0
        for sentence in sentences:
            if processed >= 10:  # Process up to 10 sentences
                break
                
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 300:
                continue
            
            # Skip meta sentences
            if any(skip in sentence.lower() for skip in ['this page', 'this article', 'category:', 'see also']):
                continue
            
            # Try multiple context templates
            for template in self.context_templates:
                context = template.format(sentence)
                
                try:
                    # Extract with REBEL
                    rebel_triples = self._run_rebel(context)
                    
                    # Filter and enhance
                    for triple in rebel_triples:
                        # Replace pronouns with title entity if appropriate
                        if triple['head'].lower() in ['it', 'this', 'that'] and title:
                            triple['head'] = title
                        
                        # Skip if still has pronouns
                        if triple['head'].lower() in ['it', 'they', 'he', 'she', 'this', 'that']:
                            continue
                        
                        triples.append(triple)
                    
                    # If we got results, don't try other templates
                    if rebel_triples:
                        break
                        
                except Exception as e:
                    logger.debug(f"REBEL error on sentence: {e}")
                    continue
            
            processed += 1
        
        return triples
    
    def _run_rebel(self, text: str) -> List[Dict]:
        """Run REBEL with better parameters"""
        triples = []
        
        try:
            inputs = self.tokenizer(
                text,
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
                    num_beams=5,  # Increased beam search
                    num_return_sequences=3,  # Get multiple interpretations
                    max_length=256,
                    length_penalty=0.8,  # Prefer longer outputs
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            
            for pred in decoded_preds:
                extracted = self._parse_rebel_output(pred)
                triples.extend(extracted)
        
        except Exception as e:
            logger.error(f"REBEL error: {e}")
        
        return triples
    
    def _extract_from_sections(self, title: str, text: str) -> List[Dict]:
        """Extract from specific wiki sections"""
        triples = []
        
        # Common section patterns in NetHack wiki
        sections = {
            r'==\s*Generation\s*==([^=]+)': 'generation_info',
            r'==\s*Strategy\s*==([^=]+)': 'strategy',
            r'==\s*Properties?\s*==([^=]+)': 'properties',
            r'==\s*Variants?\s*==([^=]+)': 'variants',
            r'==\s*Messages?\s*==([^=]+)': 'messages',
            r'==\s*Encyclopedia\s*==([^=]+)': 'lore'
        }
        
        for pattern, section_type in sections.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section_text = match[:500]  # Limit length
                
                # Add context based on section type
                if section_type == 'generation_info':
                    context = f"{title} appears in NetHack. {section_text}"
                elif section_type == 'properties':
                    context = f"{title} has these properties: {section_text}"
                elif section_type == 'strategy':
                    context = f"Strategy for {title}: {section_text}"
                else:
                    context = f"About {title}: {section_text}"
                
                section_triples = self._run_rebel(context)
                triples.extend(section_triples)
        
        return triples
    
    def _extract_with_patterns(self, title: str, text: str) -> List[Dict]:
        """Pattern-based extraction for common facts"""
        triples = []
        
        # Clean text first
        cleaned = re.sub(r'\{\{[^}]+\}\}', '', text)
        cleaned = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', cleaned)
        
        patterns = [
            # Generation patterns
            (r'(?:is|are) (?:randomly )?generated (?:on|in|at) (.+?)(?:\.|,)', 
             lambda m: {'head': title, 'type': 'generated_in', 'tail': m.group(1).strip()}),
            
            # Comparison patterns  
            (r'(?:is|are) (?:similar to|like|comparable to) (.+?)(?:\.|,)',
             lambda m: {'head': title, 'type': 'similar_to', 'tail': m.group(1).strip()}),
            
            # Property patterns
            (r'(?:is|are) (?:a|an) (.+?) (?:that|which)',
             lambda m: {'head': title, 'type': 'type_of', 'tail': m.group(1).strip()}),
            
            # Effect patterns
            (r'(?:grants?|gives?|provides?) (.+?) (?:when|if)',
             lambda m: {'head': title, 'type': 'grants', 'tail': m.group(1).strip()}),
            
            # Location patterns
            (r'(?:found|located|appears?) (?:in|on|at) (?:the )?(.+?)(?:\.|,)',
             lambda m: {'head': title, 'type': 'found_in', 'tail': m.group(1).strip()}),
            
            # Usage patterns
            (r'(?:can be )?(?:used|useful) (?:to|for) (.+?)(?:\.|,)',
             lambda m: {'head': title, 'type': 'used_for', 'tail': m.group(1).strip()}),
            
            # Contains patterns
            (r'(?:contains?|has|have) (.+?)(?:\.|,)',
             lambda m: {'head': title, 'type': 'contains', 'tail': m.group(1).strip()}),
        ]
        
        for pattern, extractor in patterns:
            for match in re.finditer(pattern, cleaned, re.IGNORECASE):
                try:
                    triple = extractor(match)
                    if triple and len(triple['tail']) < 50:  # Avoid too long tails
                        triples.append(triple)
                except:
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
    
    def _extract_from_templates(self, title: str, text: str) -> List[Dict]:
        """Extract from wiki templates (same as before)"""
        triples = []
        
        # Monster template
        monster_match = re.search(r'\{\{monster\s*\n([^}]+)\}\}', text, re.IGNORECASE | re.DOTALL)
        if monster_match:
            template_content = monster_match.group(1)
            
            for line in template_content.split('\n'):
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().strip('|')
                        value = parts[1].strip()
                        
                        # Clean value
                        value = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', value)
                        value = re.sub(r'\{\{[^}]+\}\}', '', value)
                        
                        # Map to triples
                        if key in ['AC', 'armor_class']:
                            triples.append({'head': title, 'type': 'armor_class', 'tail': value})
                        elif key == 'difficulty':
                            triples.append({'head': title, 'type': 'difficulty', 'tail': value})
                        elif key == 'level':
                            triples.append({'head': title, 'type': 'base_level', 'tail': value})
                        elif key == 'speed':
                            triples.append({'head': title, 'type': 'speed', 'tail': value})
                        elif key == 'weight':
                            triples.append({'head': title, 'type': 'weight', 'tail': value})
                        elif key == 'attacks' and value:
                            # Parse attacks
                            attack_pattern = r'(\w+)\s+(\d+d\d+)'
                            for match in re.finditer(attack_pattern, value):
                                triples.append({
                                    'head': title,
                                    'type': 'attack_type',
                                    'tail': match.group(1).lower()
                                })
                                triples.append({
                                    'head': title,
                                    'type': 'attack_damage',
                                    'tail': match.group(2)
                                })
        
        return triples
    
    def process_articles(self, articles_path: str, output_path: str):
        """Process articles"""
        logger.info(f"Loading articles from {articles_path}")
        
        with open(articles_path, 'r') as f:
            articles = json.load(f)
        
        all_triples = []
        
        logger.info(f"Processing {len(articles)} articles with enhanced extraction")
        
        for article in tqdm(articles, desc="Extracting knowledge"):
            title = article.get('title', '')
            text = article.get('text', '')
            
            # Skip meta pages
            if any(skip in title for skip in [':', 'MediaWiki', 'Template', 'Talk:']):
                continue
            
            if len(text) > 50:
                triples = self.extract_from_article(title, text)
                all_triples.extend(triples)
        
        logger.info(f"Extracted {len(all_triples)} triples")
        
        # Save
        df = pd.DataFrame(all_triples)
        df = df[['head', 'type', 'tail', 'source']]
        df.to_csv(output_path, sep='\t', index=False)
        
        # Statistics
        stats = {
            'total_triples': len(all_triples),
            'unique_entities': len(set([t['head'] for t in all_triples] + [t['tail'] for t in all_triples])),
            'unique_relations': len(set([t['type'] for t in all_triples])),
            'relation_distribution': defaultdict(int)
        }
        
        for triple in all_triples:
            stats['relation_distribution'][triple['type']] += 1
        
        stats['relation_distribution'] = dict(sorted(
            stats['relation_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        stats_path = output_path.replace('.tsv', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Top relations: {list(stats['relation_distribution'].items())[:10]}")


def main():
    extractor = EnhancedNetHackExtractor()
    
    input_path = "data/nethack_wiki_first_100.json"
    output_path = "data/nethack_kg_enhanced_100.tsv"
    
    extractor.process_articles(input_path, output_path)


if __name__ == "__main__":
    main()