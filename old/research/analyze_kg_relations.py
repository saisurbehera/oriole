#!/usr/bin/env python3
"""
Analyze the quality of relations in the NetHack knowledge graph
"""

import pandas as pd
from collections import defaultdict, Counter

def analyze_kg_relations(kg_path):
    """Analyze relations in the KG"""
    print(f"Loading KG from {kg_path}...")
    
    # Load KG
    df = pd.read_csv(kg_path, sep='\t', header=0)
    if len(df.columns) == 3:
        df.columns = ['head', 'relation', 'tail']
    else:
        df.columns = ['head', 'relation', 'tail', 'source']
    
    print(f"Total triples: {len(df)}")
    
    # Analyze key entities
    key_entities = [
        'Amulet of Yendor',
        'NetHack',
        'Excalibur',
        'Sokoban',
        'Gehennom',
        'Wizard of Yendor'
    ]
    
    print("\n" + "="*60)
    print("KEY ENTITY ANALYSIS")
    print("="*60)
    
    for entity in key_entities:
        print(f"\n{entity}:")
        
        # Find all triples with this entity
        as_head = df[df['head'] == entity]
        as_tail = df[df['tail'] == entity]
        
        print(f"  As head: {len(as_head)} triples")
        print(f"  As tail: {len(as_tail)} triples")
        
        if len(as_head) > 0:
            print("\n  Relations when as head:")
            rel_counts = as_head['relation'].value_counts()
            for rel, count in rel_counts.head(10).items():
                # Show some example tails
                examples = as_head[as_head['relation'] == rel]['tail'].head(3).tolist()
                examples_str = ', '.join([f'"{e}"' for e in examples[:2]])
                if len(examples) > 2:
                    examples_str += f", ... ({count} total)"
                print(f"    - {rel}: {examples_str}")
        
        if len(as_tail) > 0:
            print("\n  Relations when as tail:")
            rel_counts = as_tail['relation'].value_counts()
            for rel, count in rel_counts.head(10).items():
                # Show some example heads
                examples = as_tail[as_tail['relation'] == rel]['head'].head(3).tolist()
                examples_str = ', '.join([f'"{e}"' for e in examples[:2]])
                if len(examples) > 2:
                    examples_str += f", ... ({count} total)"
                print(f"    - {rel}: {examples_str}")
    
    # Analyze relation quality
    print("\n" + "="*60)
    print("RELATION QUALITY ANALYSIS")
    print("="*60)
    
    # Get all unique relations
    all_relations = df['relation'].value_counts()
    
    # Categorize relations
    good_relations = []
    game_relations = []
    structural_relations = []
    temporal_relations = []
    other_relations = []
    
    for rel, count in all_relations.items():
        rel_lower = rel.lower()
        
        # Game-specific relations
        if any(word in rel_lower for word in ['damage', 'armor', 'weight', 'cost', 'material', 
                                               'spell', 'potion', 'scroll', 'wand', 'weapon']):
            game_relations.append((rel, count))
        # Good descriptive relations
        elif any(word in rel_lower for word in ['instance_of', 'subclass_of', 'type', 'category',
                                                 'description', 'property', 'effect', 'grants']):
            good_relations.append((rel, count))
        # Structural relations
        elif any(word in rel_lower for word in ['part_of', 'has_part', 'contains', 'location',
                                                 'found_in', 'appears_in']):
            structural_relations.append((rel, count))
        # Temporal relations
        elif any(word in rel_lower for word in ['follows', 'followed_by', 'point_in_time',
                                                 'date', 'year', 'time']):
            temporal_relations.append((rel, count))
        else:
            other_relations.append((rel, count))
    
    print("\nGame-specific relations:")
    for rel, count in sorted(game_relations, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {rel}: {count}")
    
    print("\nGood descriptive relations:")
    for rel, count in sorted(good_relations, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {rel}: {count}")
    
    print("\nStructural relations:")
    for rel, count in sorted(structural_relations, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {rel}: {count}")
    
    # Find entities with the most interesting relations
    print("\n" + "="*60)
    print("ENTITIES WITH RICH INFORMATION")
    print("="*60)
    
    entity_info_score = defaultdict(float)
    entity_relations = defaultdict(set)
    
    for _, row in df.iterrows():
        head, rel, tail = row['head'], row['relation'], row['tail']
        
        # Score based on relation quality
        score = 1.0
        if any(word in rel.lower() for word in ['instance_of', 'subclass_of', 'type', 'description']):
            score = 5.0
        elif any(word in rel.lower() for word in ['damage', 'armor', 'effect', 'grants']):
            score = 3.0
        elif any(word in rel.lower() for word in ['part_of', 'location', 'found_in']):
            score = 2.0
        
        entity_info_score[head] += score
        entity_info_score[tail] += score * 0.5  # Being a tail is less informative
        entity_relations[head].add(rel)
        entity_relations[tail].add(rel)
    
    # Top entities by information richness
    top_entities = sorted(entity_info_score.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 entities by information richness:")
    for entity, score in top_entities[:20]:
        if len(entity) > 3:  # Skip short entities
            num_rels = len(entity_relations[entity])
            print(f"  - {entity}: score={score:.1f}, relations={num_rels}")


if __name__ == "__main__":
    analyze_kg_relations('data/nethack_kg_full/nethack_kg_intermediate.tsv')