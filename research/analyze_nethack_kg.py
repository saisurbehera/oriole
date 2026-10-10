"""
Analyze NetHack KG structure without requiring ULTRA
"""

import pandas as pd
from collections import defaultdict, Counter
import json

def analyze_kg(tsv_path):
    """Analyze the NetHack knowledge graph"""
    
    print(f"Loading KG from: {tsv_path}")
    
    # Read the TSV file
    df = pd.read_csv(tsv_path, sep='\t', header=0, names=['head', 'relation', 'tail'])
    
    # Basic statistics
    print(f"\n=== KG Statistics ===")
    print(f"Total triples: {len(df)}")
    print(f"Unique entities: {len(set(df['head'].unique()) | set(df['tail'].unique()))}")
    print(f"Unique relations: {len(df['relation'].unique())}")
    
    # Most common relations
    print(f"\n=== Top 20 Relations ===")
    relation_counts = df['relation'].value_counts()
    for rel, count in relation_counts.head(20).items():
        print(f"{rel}: {count}")
    
    # Entity analysis
    entity_counts = defaultdict(int)
    for _, row in df.iterrows():
        entity_counts[row['head']] += 1
        entity_counts[row['tail']] += 1
    
    # Most connected entities
    print(f"\n=== Top 20 Most Connected Entities ===")
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
    for entity, count in sorted_entities[:20]:
        print(f"{entity}: {count} connections")
    
    # NetHack-specific queries
    print(f"\n=== NetHack-Specific Analysis ===")
    
    # Find all facts about key game elements
    key_entities = ['Amulet of Yendor', 'Wizard of Yendor', 'Gehennom', 'Sokoban', 
                    'Excalibur', 'Valkyrie', 'ascension', 'quest']
    
    for entity in key_entities:
        # Find as head
        head_facts = df[df['head'] == entity]
        # Find as tail
        tail_facts = df[df['tail'] == entity]
        
        if len(head_facts) > 0 or len(tail_facts) > 0:
            print(f"\n{entity}:")
            for _, row in head_facts.iterrows():
                print(f"  {row['head']} --[{row['relation']}]--> {row['tail']}")
            for _, row in tail_facts.iterrows():
                print(f"  {row['head']} --[{row['relation']}]--> {row['tail']}")
    
    # Character classes
    print(f"\n=== Character Classes ===")
    class_facts = df[df['tail'].str.contains('character class', case=False, na=False) | 
                     df['tail'].str.contains('roles available', case=False, na=False)]
    for _, row in class_facts.iterrows():
        print(f"  {row['head']} --[{row['relation']}]--> {row['tail']}")
    
    # Artifacts and weapons
    print(f"\n=== Artifacts ===")
    artifact_facts = df[(df['relation'] == 'instance of') & 
                       (df['tail'].str.contains('artifact', case=False, na=False))]
    for _, row in artifact_facts.iterrows():
        print(f"  {row['head']} is an {row['tail']}")
    
    # Game structure (parts of NetHack)
    print(f"\n=== Game Structure (parts of NetHack) ===")
    parts = df[(df['relation'] == 'part of') & (df['tail'] == 'NetHack')]
    for _, row in parts.head(20).iterrows():
        print(f"  {row['head']}")
    
    return df

# Load statistics file
def load_stats(json_path):
    """Load and display statistics"""
    with open(json_path, 'r') as f:
        stats = json.load(f)
    
    print(f"\n=== Statistics Summary ===")
    print(f"Total triplets: {stats['total_triplets']}")
    print(f"Unique entities: {stats['unique_entities']}")
    print(f"Unique relations: {stats['unique_relations']}")
    print(f"Unique sources: {stats['unique_sources']}")
    print(f"Avg triplets per source: {stats['avg_triplets_per_source']:.2f}")

if __name__ == "__main__":
    # Analyze the KG
    kg_path = "data/nethack_kg_full/nethack_kg_intermediate.tsv"
    df = analyze_kg(kg_path)
    
    # Load statistics
    stats_path = "data/nethack_kg_full/statistics_intermediate.json"
    load_stats(stats_path)