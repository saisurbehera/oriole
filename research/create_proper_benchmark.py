#!/usr/bin/env python3
"""
Create a PROPER harder benchmark without the entity removal bug
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import random

def create_proper_benchmark(kg_path="data/nethack_kg_final_a_plus.tsv"):
    """Create train/test split properly"""
    
    print("Creating PROPER Benchmark")
    print("="*50)
    
    # Load full KG
    kg = pd.read_csv(kg_path, sep='\t')
    print(f"Total triples: {len(kg)}")
    
    # First, do a simple 80/20 split
    all_indices = list(range(len(kg)))
    random.shuffle(all_indices)
    
    split_point = int(len(kg) * 0.8)
    train_indices = all_indices[:split_point]
    test_indices = all_indices[split_point:]
    
    train_kg = kg.iloc[train_indices]
    test_kg = kg.iloc[test_indices]
    
    print(f"\nInitial split:")
    print(f"  Train: {len(train_kg)} triples")
    print(f"  Test: {len(test_kg)} triples")
    
    # Now categorize test triples by difficulty
    # Get entity/relation frequencies from TRAINING data
    entity_counts = defaultdict(int)
    relation_counts = defaultdict(int)
    
    for _, row in train_kg.iterrows():
        entity_counts[row['head']] += 1
        entity_counts[row['tail']] += 1
        relation_counts[row['type']] += 1
    
    # Categorize test triples
    test_sets = {
        'easy': [],      # Both entities common (>20 occurrences)
        'medium': [],    # At least one entity medium (5-20)
        'hard': [],      # At least one entity rare (1-5)
        'zero_shot': []  # At least one entity NOT in training
    }
    
    for _, row in test_kg.iterrows():
        head, rel, tail = row['head'], row['type'], row['tail']
        
        head_count = entity_counts.get(head, 0)
        tail_count = entity_counts.get(tail, 0)
        
        # Zero-shot: entity not in training at all
        if head_count == 0 or tail_count == 0:
            test_sets['zero_shot'].append((head, rel, tail))
        # Hard: at least one rare entity
        elif min(head_count, tail_count) <= 5:
            test_sets['hard'].append((head, rel, tail))
        # Medium: at least one medium entity
        elif min(head_count, tail_count) <= 20:
            test_sets['medium'].append((head, rel, tail))
        # Easy: both entities are common
        else:
            test_sets['easy'].append((head, rel, tail))
    
    # Balance test sets
    min_size = min(len(v) for v in test_sets.values() if v)
    max_size = min(1000, min_size)
    
    print(f"\nTest set distribution:")
    for name, triples in test_sets.items():
        print(f"  {name}: {len(triples)} triples")
    
    # Save files
    train_kg.to_csv("data/nethack_kg_train_proper.tsv", sep='\t', index=False)
    
    for name, triples in test_sets.items():
        if triples:
            # Sample to balance sizes
            sampled = triples[:max_size] if len(triples) > max_size else triples
            test_df = pd.DataFrame(sampled, columns=['head', 'type', 'tail'])
            test_df.to_csv(f"data/nethack_kg_test_{name}_proper.tsv", sep='\t', index=False)
    
    # Verify no bugs
    train_entities = set()
    for _, row in train_kg.iterrows():
        train_entities.add(row['head'])
        train_entities.add(row['tail'])
    
    print(f"\nVerification:")
    print(f"Training entities: {len(train_entities)}")
    
    # Check each test set
    for name in ['easy', 'medium', 'hard']:
        test_df = pd.read_csv(f"data/nethack_kg_test_{name}_proper.tsv", sep='\t')
        unseen = 0
        for _, row in test_df.iterrows():
            if row['head'] not in train_entities or row['tail'] not in train_entities:
                unseen += 1
        print(f"{name} test - unseen entities: {unseen}/{len(test_df)} ({unseen/len(test_df)*100:.1f}%)")
    
    return train_kg, test_sets

def main():
    random.seed(42)
    create_proper_benchmark()
    
    print("\n" + "="*50)
    print("✅ PROPER benchmark created!")
    print("Now all models can be fairly evaluated.")

if __name__ == "__main__":
    main()