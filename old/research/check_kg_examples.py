#!/usr/bin/env python3
"""
Check specific examples from KG for quality assessment
"""

import pandas as pd
import random

def examine_entity_facts(df, entity_name):
    """Show all facts about a specific entity"""
    print(f"\nFacts about '{entity_name}':")
    print("-" * 60)
    
    # As head
    head_facts = df[df['head'] == entity_name]
    for _, row in head_facts.iterrows():
        print(f"  {row['head']} --[{row['type']}]--> {row['tail']}")
    
    # As tail
    tail_facts = df[df['tail'] == entity_name]
    for _, row in tail_facts.iterrows():
        print(f"  {row['head']} --[{row['type']}]--> {row['tail']}")
    
    if len(head_facts) + len(tail_facts) == 0:
        print("  No facts found")

# Load the larger KG
df = pd.read_csv('data/nethack_kg_partial.tsv', sep='\t')

print("DETAILED QUALITY EXAMINATION")
print("="*80)

# 1. Check specific game entities
print("\n1. EXAMINING KEY GAME ENTITIES:")
key_entities = ['Medusa', 'Amulet of Yendor', 'Excalibur', 'Dragon', 'Valkyrie']

for entity in key_entities:
    examine_entity_facts(df, entity)

# 2. Check quality of special_attack relation (most common)
print("\n\n2. EXAMINING 'special_attack' RELATION (sample of 20):")
print("-" * 80)
special_attacks = df[df['type'] == 'special_attack'].sample(min(20, len(df[df['type'] == 'special_attack'])))
for _, row in special_attacks.iterrows():
    print(f"{row['head']} --> {row['tail'][:80]}{'...' if len(str(row['tail'])) > 80 else ''}")

# 3. Check quality of game-specific relations
print("\n\n3. EXAMINING GAME-SPECIFIC RELATIONS:")

# Attack damage
print("\nAttack Damage:")
damage_facts = df[df['type'] == 'attack_damage'].head(10)
for _, row in damage_facts.iterrows():
    print(f"  {row['head']} does {row['tail']} damage")

# Resistances
print("\nResistances:")
resistance_facts = df[df['type'] == 'has_resistance'].head(10)
for _, row in resistance_facts.iterrows():
    print(f"  {row['head']} has {row['tail']}")

# Item properties
print("\nItem Properties (found_in):")
item_facts = df[df['type'] == 'found_in'].head(10)
for _, row in item_facts.iterrows():
    print(f"  {row['head']} found in {row['tail']}")

# 4. Check for extraction errors
print("\n\n4. POTENTIAL EXTRACTION ERRORS:")

# Very long tails (might be incorrectly extracted text)
long_tails = df[df['tail'].astype(str).str.len() > 50]
print(f"\nTriples with long tails (>50 chars): {len(long_tails)}")
if len(long_tails) > 0:
    print("Examples:")
    for _, row in long_tails.head(5).iterrows():
        print(f"  {row['head']} --[{row['type']}]--> {str(row['tail'])[:60]}...")

# Check for weird patterns
print("\n\n5. CHECKING FOR COMMON ISSUES:")

# Entities that might be errors
weird_entities = df[df['head'].astype(str).str.contains('\\{\\{|\\}\\}|\\[\\[|\\]\\]', regex=True)]
print(f"Entities with wiki markup: {len(weird_entities)}")

# Relations that are too generic
generic_relations = ['type_of', 'found_in', 'grants']
for rel in generic_relations:
    rel_df = df[df['type'] == rel]
    print(f"\n'{rel}' relation examples:")
    for _, row in rel_df.sample(min(5, len(rel_df))).iterrows():
        print(f"  {row['head']} --> {row['tail']}")

# 6. Statistical quality metrics
print("\n\n6. QUALITY METRICS:")
print("-" * 60)

# Average entities per relation type
rel_coverage = df.groupby('type')['head'].nunique().sort_values(ascending=False)
print("Unique entities per relation (top 10):")
for rel, count in rel_coverage.head(10).items():
    print(f"  {rel}: {count} unique entities")

# Fact density for top entities
top_entities = df['head'].value_counts().head(10)
print("\nFact density for top entities:")
for entity, count in top_entities.items():
    print(f"  {entity}: {count} facts")