"""
Process more NetHack content to build a comprehensive knowledge graph
"""

import xml.etree.ElementTree as ET
import re
import json
import os
from transformers import pipeline

def clean_wiki_text(text):
    """Clean wiki markup from text"""
    if not text:
        return ""
    
    # Remove wiki templates {{...}}
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    
    # Remove wiki links [[...]]
    text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)  # [[link|text]] -> text
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[link]] -> link
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove wiki formatting
    text = re.sub(r"'''([^']+)'''", r'\1', text)  # bold
    text = re.sub(r"''([^']+)''", r'\1', text)    # italic
    
    # Remove references [1], [2], etc
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove categories
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    return text

def extract_more_articles(xml_path, limit=50):
    """Extract more diverse articles from NetHack wiki"""
    
    # Target articles that would give good game knowledge
    priority_titles = [
        "Amulet of Yendor", "Gehennom", "Astral Plane", "Quest",
        "Elbereth", "Oracle", "Sokoban", "Castle", "Medusa",
        "Wizard of Yendor", "Vlad the Impaler", "High priest",
        "Wand of wishing", "Bag of holding", "Artifact",
        "Excalibur", "Mjollnir", "Stormbringer",
        "Valkyrie", "Wizard", "Samurai", "Monk", "Knight",
        "Dragon", "Demon", "Lich", "Vampire", "Mind flayer",
        "Polymorph", "Teleportation", "Invisibility",
        "Armor class", "Hit points", "Experience",
        "Alignment", "Luck", "Prayer", "Sacrifice",
        "Shop", "Vault", "Throne room", "Zoo",
        "Wand", "Scroll", "Potion", "Ring", "Spellbook",
        "Food", "Corpse", "Intrinsic", "Resistance"
    ]
    
    articles = []
    context = ET.iterparse(xml_path, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    current_page = {}
    found_priority = set()
    
    for event, elem in context:
        if event == 'start' and elem.tag.endswith('page'):
            current_page = {}
            
        elif event == 'end':
            if elem.tag.endswith('title'):
                if elem.text:
                    current_page['title'] = elem.text
                    
            elif elem.tag.endswith('text'):
                if elem.text:
                    current_page['text'] = elem.text
                    
            elif elem.tag.endswith('page'):
                if current_page.get('title') and current_page.get('text'):
                    title = current_page['title']
                    
                    # Skip meta pages
                    if not any(prefix in title for prefix in ['File:', 'Template:', 'Category:', 'User:', 'Talk:']):
                        # Prioritize important articles
                        if title in priority_titles and title not in found_priority:
                            # Take more content from priority articles
                            text = current_page['text'][:3000]
                            articles.append({
                                'page_title': title,
                                'chunk_text': clean_wiki_text(text)
                            })
                            found_priority.add(title)
                        elif len(articles) < limit:
                            # Add other articles too
                            text = current_page['text'][:1500]
                            articles.append({
                                'page_title': title,
                                'chunk_text': clean_wiki_text(text)
                            })
                
                elem.clear()
                root.clear()
                
                if len(articles) >= limit:
                    break
    
    print(f"Extracted {len(articles)} articles")
    print(f"Priority articles found: {len(found_priority)}")
    return articles

# Extract more content
print("Extracting more NetHack articles...")
articles = extract_more_articles("data/nethackwiki_current.xml", limit=100)

# Save the expanded chunks
os.makedirs("data/nethack_samples", exist_ok=True)
with open("data/nethack_samples/expanded_chunks.json", 'w') as f:
    json.dump(articles, f, indent=2)

print(f"Saved {len(articles)} articles to expanded_chunks.json")

# Process with REBEL
print("\nProcessing with REBEL model...")
from rebel_extract_kg import process_nethack_chunks
process_nethack_chunks("data/nethack_samples/expanded_chunks.json", "data/nethack_kg_expanded")