#!/usr/bin/env python3
"""
Test custom analysis with A+ KG
"""

import pandas as pd

# Load the A+ KG
kg = pd.read_csv('data/nethack_kg_final_a_plus.tsv', sep='\t')

print("A+ KNOWLEDGE GRAPH ANALYSIS")
print("="*60)
print(f"Total triples: {len(kg):,}")
print(f"Unique entities: {len(set(kg['head'].tolist() + kg['tail'].tolist())):,}")
print(f"Unique relations: {kg['type'].nunique()}")

# Use priority column to get most important facts first
print("\nHIGH PRIORITY FACTS (Priority >= 9):")
print("-"*40)
high_priority = kg[kg['priority'] >= 9].sort_values('priority', ascending=False)
for _, row in high_priority.head(10).iterrows():
    print(f"[{row['priority']}] {row['head']} --{row['type']}--> {row['tail']}")

# Show computed facts
print("\nCOMPUTED FACTS:")
print("-"*40)
computed = kg[kg['source'] == 'computed']
print(f"Total computed facts: {len(computed)}")
for _, row in computed.head(5).iterrows():
    print(f"  {row['head']} --{row['type']}--> {row['tail']}")

# Show interpreted facts
print("\nINTERPRETED FACTS:")
print("-"*40)
interpreted = kg[kg['source'] == 'interpreted']
print(f"Total interpreted facts: {len(interpreted)}")
for _, row in interpreted.head(5).iterrows():
    print(f"  {row['head']} --{row['type']}--> {row['tail']}")

# Semantic clusters
print("\nSEMANTIC CLUSTERS:")
print("-"*40)
clusters = kg['relation_cluster'].value_counts()
for cluster, count in clusters.items():
    print(f"  {cluster}: {count:,} facts")

print("\nAnalysis complete! The A+ KG is ready for use.")