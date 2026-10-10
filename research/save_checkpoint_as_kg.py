#!/usr/bin/env python3
"""
Save checkpoint data as KG for testing
"""

import json
import pandas as pd
from collections import defaultdict

# Load checkpoint
with open('data/extraction_checkpoint.json', 'r') as f:
    checkpoint = json.load(f)

triples = checkpoint['triples']
print(f"Loaded {len(triples)} triples from checkpoint")

# Save as TSV
df = pd.DataFrame(triples)
df = df[['head', 'type', 'tail', 'source']]
df.to_csv('data/nethack_kg_partial.tsv', sep='\t', index=False)

# Generate statistics
stats = {
    'total_triples': len(triples),
    'unique_entities': len(set([t['head'] for t in triples] + [t['tail'] for t in triples])),
    'unique_relations': len(set([t['type'] for t in triples])),
    'relation_distribution': defaultdict(int),
    'source_distribution': defaultdict(int)
}

for triple in triples:
    stats['relation_distribution'][triple['type']] += 1
    stats['source_distribution'][triple['source']] += 1

# Sort distributions
stats['relation_distribution'] = dict(sorted(
    stats['relation_distribution'].items(),
    key=lambda x: x[1],
    reverse=True
)[:20])

stats['source_distribution'] = dict(sorted(
    stats['source_distribution'].items(),
    key=lambda x: x[1],
    reverse=True
)[:20])

# Save stats
with open('data/nethack_kg_partial_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\nSaved to data/nethack_kg_partial.tsv")
print(f"Unique entities: {stats['unique_entities']}")
print(f"Unique relations: {stats['unique_relations']}")
print(f"\nTop relations:")
for rel, count in list(stats['relation_distribution'].items())[:10]:
    print(f"  {rel}: {count}")
print(f"\nTop sources:")
for src, count in list(stats['source_distribution'].items())[:10]:
    print(f"  {src}: {count}")