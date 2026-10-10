"""
Test NetHack knowledge graph with question answering
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_kg_simple import SimpleKGWrapper

def test_nethack_kg():
    """Test the NetHack knowledge graph"""
    print("=== NetHack Knowledge Graph Q&A ===\n")
    
    # Load the knowledge graph
    kg = SimpleKGWrapper()
    kg.parse_kg_from_file("data/nethack_kg/nethack_kg.tsv")
    
    # Show statistics
    stats = kg.get_stats()
    print(f"Loaded NetHack KG with:")
    print(f"  - {stats['num_entities']} entities")
    print(f"  - {stats['num_relations']} relations") 
    print(f"  - {stats['num_triples']} triples\n")
    
    print("Entities:", sorted(kg.entity2id.keys()))
    print("\nRelations:", sorted(kg.relation2id.keys()))
    
    # Test queries
    print("\n--- Question Answering ---\n")
    
    # Q1: What is the Amulet of Yendor?
    print("Q: What is the Amulet of Yendor?")
    answers = kg.answer_simple_query("Amulet of Yendor", "is")
    print(f"A: The Amulet of Yendor is: {answers}\n")
    
    # Q2: Where is the Amulet hidden?
    print("Q: Where is the Amulet of Yendor hidden?")
    answers = kg.answer_simple_query("Amulet of Yendor", "hidden in")
    print(f"A: The Amulet is hidden in: {answers}\n")
    
    # Q3: What must the player retrieve?
    print("Q: What must the player retrieve?")
    answers = kg.answer_simple_query("player", "must retrieve")
    print(f"A: The player must retrieve: {answers}\n")
    
    # Q4: What is Elbereth?
    print("Q: What is Elbereth?")
    answers = kg.answer_simple_query("Elbereth", "is")
    print(f"A: Elbereth is: {answers}\n")
    
    # Q5: What are monsters scared by?
    print("Q: What are monsters scared by?")
    # Need to search backwards
    scared_by = []
    for entity in kg.entity2id:
        if "monsters" in kg.answer_simple_query(entity, "scared by"):
            scared_by.extend(kg.answer_simple_query("monsters", "scared by"))
            break
    print(f"A: Monsters are scared by: {scared_by}\n")
    
    # Show all facts about NetHack
    print("--- All facts about NetHack ---")
    nethack_relations = kg.get_entity_relations("NetHack")
    for rel in nethack_relations:
        facts = kg.answer_simple_query("NetHack", rel)
        print(f"NetHack {rel}: {facts}")
    
    # Show knowledge graph connections
    print("\n--- Knowledge Graph Structure ---")
    print("Key entities and their connections:")
    
    key_entities = ["Amulet of Yendor", "player", "Moloch", "Elbereth", "NetHack"]
    for entity in key_entities:
        if entity in kg.entity2id:
            relations = kg.get_entity_relations(entity)
            if relations:
                print(f"\n{entity}:")
                for rel in relations:
                    targets = kg.answer_simple_query(entity, rel)
                    for target in targets:
                        print(f"  --[{rel}]--> {target}")


def create_extended_nethack_kg():
    """Create an extended NetHack KG with more game knowledge"""
    print("\n\n=== Creating Extended NetHack KG ===\n")
    
    # Additional triples based on NetHack game mechanics
    additional_triples = [
        # Game structure
        ("NetHack", "has branch", "Sokoban"),
        ("NetHack", "has branch", "Mines"),
        ("NetHack", "has branch", "Quest"),
        ("NetHack", "has location", "Castle"),
        ("NetHack", "has location", "Valley of the Dead"),
        
        # Character classes
        ("Wizard", "is", "character class"),
        ("Valkyrie", "is", "character class"), 
        ("Wizard", "starts with", "spellbooks"),
        ("Valkyrie", "starts with", "good armor"),
        
        # Items and mechanics
        ("wand of wishing", "found in", "Castle"),
        ("bag of holding", "reward from", "Sokoban"),
        ("amulet of reflection", "reward from", "Sokoban"),
        
        # Monsters and NPCs
        ("Oracle", "provides", "consultations"),
        ("Oracle", "found on", "level 5-9"),
        ("Medusa", "can turn to stone", "player"),
        ("Wizard of Yendor", "guards", "tower"),
        
        # Game objectives
        ("player", "must reach", "Astral Plane"),
        ("player", "must offer to", "god"),
        ("ascension", "requires", "Amulet of Yendor"),
        
        # Combat and magic
        ("spells", "learned from", "spellbooks"),
        ("potions", "provide", "temporary effects"),
        ("scrolls", "provide", "one-time effects"),
        ("armor", "provides", "protection"),
        
        # Dungeon features
        ("altar", "used for", "prayer"),
        ("altar", "used for", "sacrifice"),
        ("fountain", "can grant", "wishes"),
        ("throne", "can grant", "wishes")
    ]
    
    # Combine with existing triples
    with open("data/nethack_kg/nethack_kg.tsv", 'r') as f:
        existing = [line.strip() for line in f if line.strip()]
    
    # Write extended KG
    with open("data/nethack_kg/nethack_kg_extended.tsv", 'w') as f:
        # Write existing
        for line in existing:
            f.write(line + '\n')
        
        # Write additional
        for h, r, t in additional_triples:
            f.write(f"{h}\t{r}\t{t}\n")
    
    print(f"Created extended KG with {len(existing) + len(additional_triples)} triples")
    print("Saved to: data/nethack_kg/nethack_kg_extended.tsv")


if __name__ == "__main__":
    # Test the basic extracted KG
    test_nethack_kg()
    
    # Create and test extended KG
    create_extended_nethack_kg()
    
    print("\n" + "="*50)
    print("NetHack KG Q&A testing completed!")
    print("\nThis KG can be used to:")
    print("1. Answer questions about NetHack game mechanics")
    print("2. Train an AI assistant to understand the game")
    print("3. Power a NetHack-specific chatbot or tool")
    print("4. Build more complex reasoning about game strategies")