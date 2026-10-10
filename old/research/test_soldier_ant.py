#!/usr/bin/env python3
"""
Test extraction specifically on Soldier ant article
"""

import json
from combined_extraction import CombinedNetHackExtractor

# Load articles
with open('data/nethack_wiki_first_100.json', 'r') as f:
    articles = json.load(f)

# Find soldier ant
soldier_ant = None
for article in articles:
    if article['title'] == 'Soldier ant':
        soldier_ant = article
        break

if soldier_ant:
    print(f"Article: {soldier_ant['title']}")
    print(f"Text length: {len(soldier_ant['text'])}")
    print("\nRaw text preview:")
    print("-" * 50)
    print(soldier_ant['text'][:500])
    print("-" * 50)
    
    # Test extraction
    extractor = CombinedNetHackExtractor()
    
    # Show cleaned text
    cleaned = extractor._clean_wiki_text(soldier_ant['text'])
    print("\nCleaned text preview:")
    print("-" * 50)
    print(cleaned[:500])
    print("-" * 50)
    
    # Test pattern extraction
    print("\nTesting pattern extraction...")
    pattern_triples = extractor._extract_with_patterns(soldier_ant['title'], cleaned)
    print(f"Pattern triples: {len(pattern_triples)}")
    for triple in pattern_triples:
        print(f"  {triple['head']} --[{triple['type']}]--> {triple['tail']}")
    
    # Test specific patterns
    print("\nTesting specific patterns on text:")
    import re
    
    # Look for damage patterns
    damage_pattern = r'(?:deals?|does?|inflicts?)\s+(\d+d\d+(?:\s*[+-]\s*\d+)?)\s+(?:points?\s+of\s+)?damage'
    matches = list(re.finditer(damage_pattern, cleaned, re.IGNORECASE))
    print(f"Damage pattern matches: {len(matches)}")
    for m in matches:
        print(f"  Found: {m.group(0)}")
    
    # Look for bite/sting patterns
    attack_pattern = r'(?:bite|sting|claw)\s+(\d+d\d+)'
    matches = list(re.finditer(attack_pattern, cleaned, re.IGNORECASE))
    print(f"Attack pattern matches: {len(matches)}")
    for m in matches:
        print(f"  Found: {m.group(0)}")