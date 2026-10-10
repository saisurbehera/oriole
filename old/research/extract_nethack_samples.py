"""
Extract sample text from NetHack wiki XML for testing REBEL model
"""

import xml.etree.ElementTree as ET
import re
import json
import os

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

def extract_key_articles(xml_path, target_titles):
    """Extract specific articles from wiki XML"""
    articles = {}
    
    print(f"Parsing XML file: {xml_path}")
    
    # Use iterparse for memory efficiency
    context = ET.iterparse(xml_path, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    current_page = {}
    found_count = 0
    
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
                if current_page.get('title') in target_titles and current_page.get('text'):
                    title = current_page['title']
                    # Extract first few paragraphs
                    text = current_page['text']
                    # Get first 2000 characters for sample
                    text = text[:2000] if len(text) > 2000 else text
                    articles[title] = clean_wiki_text(text)
                    found_count += 1
                    print(f"Found: {title}")
                    
                    if found_count >= len(target_titles):
                        break
                
                # Clear memory
                elem.clear()
                root.clear()
    
    return articles

# Key NetHack concepts to extract
target_articles = [
    "Amulet of Yendor",
    "Elbereth",
    "Oracle",
    "Sokoban",
    "Castle",
    "Wizard of Yendor",
    "Medusa",
    "Gehennom",
    "Ascension",
    "Quest",
    "Dungeon",
    "Monster",
    "Prayer",
    "Luck",
    "Altar"
]

# Extract articles
articles = extract_key_articles("data/nethackwiki_current.xml", target_articles)

# Create sample chunks for REBEL processing
os.makedirs("data/nethack_samples", exist_ok=True)

# Save individual article samples
sample_chunks = []

for title, text in articles.items():
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Take first 5 meaningful sentences
    chunk_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 20:  # Skip very short sentences
            chunk_sentences.append(sent)
        if len(chunk_sentences) >= 5:
            break
    
    chunk_text = '. '.join(chunk_sentences) + '.'
    
    sample_chunks.append({
        'page_title': title,
        'chunk_text': chunk_text
    })

# Add some manually created NetHack-specific samples
manual_samples = [
    {
        'page_title': 'Game_Basics',
        'chunk_text': 'NetHack is a single-player roguelike dungeon exploration game. The player controls an adventurer who must retrieve the Amulet of Yendor from the deepest level of the dungeon. The game features permadeath, meaning when your character dies, you must start over. Each game randomly generates the dungeon layout, making every playthrough unique.'
    },
    {
        'page_title': 'Combat_System',
        'chunk_text': 'Combat in NetHack is turn-based. The player can attack monsters by moving into them or using ranged weapons. Weapons include swords, spears, bows, and magical wands. Armor provides protection from attacks. Different monsters have different resistances and attacks.'
    },
    {
        'page_title': 'Magic_System',
        'chunk_text': 'Magic in NetHack includes spells, wands, scrolls, and potions. Wizards start with spellbooks and can learn new spells. Wands can be zapped to produce magical effects. Scrolls provide one-time magical effects when read. Potions can be quaffed for various effects.'
    },
    {
        'page_title': 'Character_Classes',
        'chunk_text': 'NetHack offers multiple character classes including Archeologist, Barbarian, Caveman, Healer, Knight, Monk, Priest, Ranger, Rogue, Samurai, Tourist, Valkyrie, and Wizard. Each class has unique starting equipment, abilities, and quest. The Valkyrie starts with good armor and weapons. The Wizard begins with spellbooks and magical items.'
    }
]

sample_chunks.extend(manual_samples)

# Save all samples
with open('data/nethack_samples/sample_chunks.json', 'w') as f:
    json.dump(sample_chunks, f, indent=2)

print(f"\nSaved {len(sample_chunks)} sample chunks to data/nethack_samples/sample_chunks.json")

# Also save as simple text for quick testing
with open('data/nethack_samples/sample_texts.txt', 'w') as f:
    for chunk in sample_chunks:
        f.write(f"=== {chunk['page_title']} ===\n")
        f.write(f"{chunk['chunk_text']}\n\n")

print("Also saved as sample_texts.txt for easy viewing")