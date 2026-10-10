#!/usr/bin/env python3
"""
Test extraction on a single article to debug
"""

import json
from combined_extraction import CombinedNetHackExtractor

# Load first 100 articles
with open('data/nethack_wiki_first_100.json', 'r') as f:
    articles = json.load(f)

# Find a good test article
test_articles = []
for article in articles:
    if 'soldier ant' in article['title'].lower() or 'amulet' in article['title'].lower():
        test_articles.append(article)
        print(f"Found: {article['title']}")

# Test extraction
extractor = CombinedNetHackExtractor()

# Test on multiple articles
test_count = 0
for article in articles:
    if test_count >= 5:
        break
    
    if len(article['text']) > 500 and not article['text'].startswith('#REDIRECT'):
        print(f"\nTesting on: {article['title']}")
        print(f"Text length: {len(article['text'])}")
        
        # Show cleaned text
        cleaned = extractor._clean_wiki_text(article['text'])
        print(f"Cleaned text preview: {cleaned[:200]}...")
        
        triples = extractor.extract_from_article(article['title'], article['text'])
        
        print(f"\nExtracted {len(triples)} triples:")
        for triple in triples[:5]:  # Show first 5
            print(f"  {triple['head']} --[{triple['type']}]--> {triple['tail']}")
        
        test_count += 1