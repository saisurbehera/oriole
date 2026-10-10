#!/usr/bin/env python3
"""
Parse wiki templates to extract game information
"""

import re
from typing import Dict, List, Tuple

def parse_monster_template(text: str) -> Dict[str, str]:
    """Parse NetHack wiki monster template"""
    data = {}
    
    # Extract template content
    template_match = re.search(r'\{\{monster\s*\n([^}]+)\}\}', text, re.IGNORECASE | re.DOTALL)
    if not template_match:
        return data
    
    template_content = template_match.group(1)
    
    # Parse key-value pairs
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

def extract_triples_from_template(name: str, template_data: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """Convert template data to triples"""
    triples = []
    
    # Basic properties
    if 'AC' in template_data:
        triples.append((name, 'armor_class', template_data['AC']))
    
    if 'difficulty' in template_data:
        triples.append((name, 'difficulty', template_data['difficulty']))
    
    if 'level' in template_data:
        triples.append((name, 'base_level', template_data['level']))
    
    if 'speed' in template_data:
        triples.append((name, 'speed', template_data['speed']))
    
    if 'weight' in template_data:
        triples.append((name, 'weight', template_data['weight']))
    
    if 'align' in template_data:
        align_val = template_data['align']
        if align_val == '0':
            triples.append((name, 'alignment', 'neutral'))
        elif int(align_val) > 0:
            triples.append((name, 'alignment', 'lawful'))
        else:
            triples.append((name, 'alignment', 'chaotic'))
    
    # Parse attacks
    if 'attacks' in template_data:
        attacks = template_data['attacks']
        # Parse attack patterns like "Bite 2d4 physical, Sting 3d4 poison"
        attack_pattern = r'(\w+)\s+(\d+d\d+)\s+(\w+)'
        for match in re.finditer(attack_pattern, attacks):
            attack_type = match.group(1)
            damage = match.group(2)
            damage_type = match.group(3)
            
            triples.append((name, 'attack_type', attack_type.lower()))
            triples.append((name, 'attack_damage', damage))
            triples.append((name, 'damage_type', damage_type))
    
    # Parse resistances
    if 'resistances' in template_data:
        resistances = template_data['resistances']
        # Extract resistance types
        res_pattern = r'(\w+)\s+resistance'
        for match in re.finditer(res_pattern, resistances, re.IGNORECASE):
            res_type = match.group(1)
            triples.append((name, 'has_resistance', f"{res_type.lower()} resistance"))
    
    if 'resistances conveyed' in template_data:
        conveyed = template_data['resistances conveyed']
        res_pattern = r'(\w+)\s+resistance'
        for match in re.finditer(res_pattern, conveyed, re.IGNORECASE):
            res_type = match.group(1)
            triples.append((name, 'grants_resistance', f"{res_type.lower()} resistance"))
    
    return triples

# Test
if __name__ == "__main__":
    import json
    
    # Load articles
    with open('data/nethack_wiki_first_100.json', 'r') as f:
        articles = json.load(f)
    
    # Test on soldier ant
    for article in articles:
        if article['title'] == 'Soldier ant':
            print(f"Parsing: {article['title']}")
            
            template_data = parse_monster_template(article['text'])
            print("\nTemplate data:")
            for k, v in template_data.items():
                print(f"  {k}: {v}")
            
            print("\nExtracted triples:")
            triples = extract_triples_from_template('Soldier ant', template_data)
            for h, r, t in triples:
                print(f"  {h} --[{r}]--> {t}")
            
            break