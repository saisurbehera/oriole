#!/usr/bin/env python3
"""
Analyze why so many unseen entities in "easy" test
"""

import pandas as pd

# Load train and test sets
train_kg = pd.read_csv("data/nethack_kg_train.tsv", sep='\t')
test_easy = pd.read_csv("data/nethack_kg_test_easy.tsv", sep='\t')
test_zero = pd.read_csv("data/nethack_kg_test_zero_shot.tsv", sep='\t')

# Get training entities
train_entities = set()
for _, row in train_kg.iterrows():
    train_entities.add(row['head'])
    train_entities.add(row['tail'])

print(f"Training entities: {len(train_entities)}")
print(f"Training triples: {len(train_kg)}")

# Check easy test
unseen_in_easy = []
for _, row in test_easy.iterrows():
    if row['head'] not in train_entities or row['tail'] not in train_entities:
        unseen_in_easy.append((row['head'], row['type'], row['tail']))

print(f"\nEasy test: {len(test_easy)} triples")
print(f"Unseen entity triples: {len(unseen_in_easy)} ({len(unseen_in_easy)/len(test_easy)*100:.1f}%)")

# Check what entities are in zero-shot
zero_shot_entities = set()
for _, row in test_zero.iterrows():
    zero_shot_entities.add(row['head'])
    zero_shot_entities.add(row['tail'])

print(f"\nZero-shot test: {len(test_zero)} triples")
print(f"Zero-shot entities: {len(zero_shot_entities)}")

# See overlap
overlap = 0
for head, rel, tail in unseen_in_easy[:10]:
    if head in zero_shot_entities or tail in zero_shot_entities:
        overlap += 1
        print(f"Example: {head} {rel} {tail}")
        if head not in train_entities:
            print(f"  - '{head}' removed from training (in zero-shot)")
        if tail not in train_entities:
            print(f"  - '{tail}' removed from training (in zero-shot)")

print(f"\nThe bug: Entities were removed from training for zero-shot,")
print(f"but other test sets still contain those entities!")