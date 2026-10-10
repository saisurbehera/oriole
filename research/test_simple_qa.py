#!/usr/bin/env python3
"""
Simple QA test with the final KG
"""

import pandas as pd

# Load the KG
print("Loading KG...")
df = pd.read_csv('data/nethack_kg_final_100.tsv', sep='\t')
print(f"Loaded {len(df)} triples")

# Show some statistics
print("\nRelation types:")
print(df['type'].value_counts())

print("\nTop entities:")
entity_counts = pd.concat([df['head'], df['tail']]).value_counts()
print(entity_counts.head(10))

# Simple QA function
def simple_qa(question, df):
    """Simple pattern-based QA"""
    question = question.lower()
    
    # Find mentioned entities
    entities = []
    for entity in pd.concat([df['head'], df['tail']]).unique():
        if str(entity).lower() in question:
            entities.append(entity)
    
    if not entities:
        return "No entities found in question"
    
    # Get facts about first entity
    entity = entities[0]
    facts = []
    
    # As head
    head_facts = df[df['head'] == entity]
    for _, row in head_facts.iterrows():
        facts.append(f"{entity} {row['type']} {row['tail']}")
    
    # As tail
    tail_facts = df[df['tail'] == entity]
    for _, row in tail_facts.iterrows():
        facts.append(f"{row['head']} {row['type']} {entity}")
    
    if facts:
        return ". ".join(facts[:3]) + "."
    else:
        return f"No facts found about {entity}"

# Test questions
print("\n" + "="*60)
print("Testing Simple QA")
print("="*60)

questions = [
    "What is soldier ant?",
    "Tell me about Medusa",
    "What about Wumpus?"
]

for q in questions:
    print(f"\nQ: {q}")
    answer = simple_qa(q, df)
    print(f"A: {answer}")