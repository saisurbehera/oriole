#!/usr/bin/env python3
"""
Test the A+ quality KG with QA system
"""

import pandas as pd

# Load the A+ KG
df = pd.read_csv('data/nethack_kg_final_a_plus.tsv', sep='\t')

print("A+ QUALITY KNOWLEDGE GRAPH TEST")
print("="*60)
print(f"Total facts: {len(df)}")
print(f"Unique entities: {df['head'].nunique() + df['tail'].nunique()}")
print(f"Unique relations: {df['type'].nunique()}")

# Test specific queries
print("\n\nTEST QUERIES:")
print("-"*60)

test_entities = ['Medusa', 'Soldier ant', 'Dragon', 'Excalibur', 'Amulet of Yendor']

for entity in test_entities:
    print(f"\n{entity}:")
    entity_facts = df[df['head'] == entity].sort_values('priority', ascending=False)
    
    # Show top priority facts
    shown = 0
    for _, fact in entity_facts.iterrows():
        if shown < 5:
            print(f"  [{fact['priority']}] {fact['type']}: {fact['tail']}")
            shown += 1
    
    # Show computed facts if any
    computed = entity_facts[entity_facts['source'] == 'computed']
    if len(computed) > 0:
        print("  Computed facts:")
        for _, fact in computed.iterrows():
            print(f"    - {fact['type']}: {fact['tail']}")
    
    # Show interpreted facts if any
    interpreted = entity_facts[entity_facts['source'] == 'interpreted']
    if len(interpreted) > 0:
        print("  Interpretations:")
        for _, fact in interpreted.iterrows():
            print(f"    - {fact['type']}: {fact['tail']}")

# Test the improved QA system
print("\n\n" + "="*60)
print("TESTING QA WITH A+ KNOWLEDGE GRAPH")
print("="*60)

from improved_retriever_qa import ImprovedRetrieverQA

# Update to use A+ KG
qa = ImprovedRetrieverQA('data/nethack_kg_final_a_plus.tsv')

test_questions = [
    "How dangerous is Medusa?",
    "What makes Soldier ant tough?",
    "Which monsters are extremely fast?",
    "What resistances does Dragon have?",
    "How good is Excalibur?",
    "What is the combat rating of Medusa?"
]

for question in test_questions:
    print(f"\nQ: {question}")
    answer = qa.answer_question(question, verbose=False)
    print(f"A: {answer}")

# Check relation quality
print("\n\n" + "="*60)
print("RELATION QUALITY CHECK")
print("="*60)

# Show distribution of relation clusters
print("\nRelation clusters:")
cluster_counts = df['relation_cluster'].value_counts()
for cluster, count in cluster_counts.items():
    print(f"  {cluster}: {count} facts")
    # Show example relations in this cluster
    cluster_rels = df[df['relation_cluster'] == cluster]['type'].value_counts().head(3)
    for rel, rel_count in cluster_rels.items():
        print(f"    - {rel}: {rel_count}")

# Priority distribution
print("\nPriority distribution:")
priority_counts = df['priority'].value_counts().sort_index(ascending=False)
for priority, count in priority_counts.items():
    print(f"  Priority {priority}: {count} facts")