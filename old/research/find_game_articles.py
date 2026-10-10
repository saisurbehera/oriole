#!/usr/bin/env python3
"""
Find actual game content articles in the first 100
"""

import json

with open('data/nethack_wiki_first_100.json', 'r') as f:
    articles = json.load(f)

print("Game content articles:")
print("-" * 50)

game_keywords = ['damage', 'AC', 'armor', 'weapon', 'monster', 'item', 'level', 
                 'dungeon', 'HP', 'hit points', 'speed', 'alignment']

for i, article in enumerate(articles):
    title = article['title']
    text = article['text'].lower()
    
    # Skip meta pages
    if any(skip in title for skip in ['NetHackWiki:', 'Template:', 'User:', 'File:', 'Category:']):
        continue
    
    # Check for game content
    if any(keyword in text for keyword in game_keywords):
        print(f"\n{i}. {title}")
        # Show context
        for keyword in game_keywords:
            if keyword in text:
                idx = text.find(keyword)
                context = text[max(0, idx-50):idx+50]
                print(f"   ...{context}...")
                break