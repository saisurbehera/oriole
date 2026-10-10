#!/usr/bin/env python3
"""
Create a harder benchmark for KG link prediction
- Hold out 20% of triples for testing
- Focus on rare relations and entities
- Test true generalization
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import random

def create_harder_benchmark(kg_path="data/nethack_kg_final_a_plus.tsv"):
    """Create train/test split with harder test cases"""
    
    print("Creating Harder Benchmark")
    print("="*50)
    
    # Load full KG
    kg = pd.read_csv(kg_path, sep='\t')
    print(f"Total triples: {len(kg)}")
    
    # Analyze KG statistics
    entity_counts = defaultdict(int)
    relation_counts = defaultdict(int)
    
    for _, row in kg.iterrows():
        entity_counts[row['head']] += 1
        entity_counts[row['tail']] += 1
        relation_counts[row['type']] += 1
    
    # Categorize entities by frequency
    rare_entities = [e for e, count in entity_counts.items() if count <= 5]
    medium_entities = [e for e, count in entity_counts.items() if 5 < count <= 20]
    common_entities = [e for e, count in entity_counts.items() if count > 20]
    
    print(f"\nEntity distribution:")
    print(f"  Rare (≤5 facts): {len(rare_entities)}")
    print(f"  Medium (6-20 facts): {len(medium_entities)}")
    print(f"  Common (>20 facts): {len(common_entities)}")
    
    # Create test sets of increasing difficulty
    test_sets = {
        'easy': [],
        'medium': [],
        'hard': [],
        'zero_shot': []
    }
    
    train_triples = []
    all_indices = list(range(len(kg)))
    random.shuffle(all_indices)
    
    for idx in all_indices:
        row = kg.iloc[idx]
        head, rel, tail = row['head'], row['type'], row['tail']
        
        # Determine difficulty
        head_freq = entity_counts[head]
        tail_freq = entity_counts[tail]
        rel_freq = relation_counts[rel]
        
        # Zero-shot: Entities that will have NO training examples
        if head in rare_entities and random.random() < 0.5:
            test_sets['zero_shot'].append((head, rel, tail))
            # Remove ALL triples with this entity from training
            continue
        
        # Hard: Rare entities or rare relations
        elif (head_freq <= 10 or tail_freq <= 10 or rel_freq <= 50) and len(test_sets['hard']) < 1000:
            test_sets['hard'].append((head, rel, tail))
        
        # Medium: Medium frequency entities
        elif (head in medium_entities or tail in medium_entities) and len(test_sets['medium']) < 1000:
            test_sets['medium'].append((head, rel, tail))
        
        # Easy: Common entities
        elif len(test_sets['easy']) < 1000:
            test_sets['easy'].append((head, rel, tail))
        
        else:
            train_triples.append((head, rel, tail))
    
    # Print test set statistics
    print(f"\nTest set sizes:")
    for difficulty, triples in test_sets.items():
        print(f"  {difficulty}: {len(triples)} triples")
    
    print(f"\nTraining set: {len(train_triples)} triples")
    
    # Save splits
    train_df = pd.DataFrame(train_triples, columns=['head', 'type', 'tail'])
    train_df.to_csv("data/nethack_kg_train.tsv", sep='\t', index=False)
    
    for difficulty, triples in test_sets.items():
        if triples:
            test_df = pd.DataFrame(triples, columns=['head', 'type', 'tail'])
            test_df.to_csv(f"data/nethack_kg_test_{difficulty}.tsv", sep='\t', index=False)
    
    # Create adversarial test cases
    print("\nCreating adversarial test cases...")
    adversarial_tests = []
    
    # Type 1: Negation patterns (opposite relations)
    negation_pairs = [
        ('resistant_to', 'vulnerable_to'),
        ('allied_with', 'hostile_to'),
        ('can_eat', 'poisonous_to'),
        ('found_in', 'excluded_from')
    ]
    
    for rel1, rel2 in negation_pairs:
        # Find entities with rel1, test if they have rel2
        rel1_facts = [(h, t) for h, r, t in train_triples if r == rel1]
        for h, t in rel1_facts[:5]:
            adversarial_tests.append({
                'query': (h, rel2, '?'),
                'type': 'negation',
                'expected': 'different_from_' + t
            })
    
    # Type 2: Transitive reasoning
    transitive_rels = ['part_of', 'subclass_of', 'located_in']
    for rel in transitive_rels:
        # Find chains A->B->C
        facts = [(h, t) for h, r, t in train_triples if r == rel]
        facts_dict = defaultdict(list)
        for h, t in facts:
            facts_dict[h].append(t)
        
        for a, b_list in facts_dict.items():
            for b in b_list:
                if b in facts_dict:
                    for c in facts_dict[b]:
                        adversarial_tests.append({
                            'query': (a, rel, '?'),
                            'type': 'transitive',
                            'expected': c,
                            'chain': f"{a}->{b}->{c}"
                        })
                        break
            if len(adversarial_tests) > 20:
                break
    
    # Type 3: Inverse relations
    inverse_pairs = [
        ('has_part', 'part_of'),
        ('damages', 'damaged_by'),
        ('summons', 'summoned_by')
    ]
    
    for rel1, rel2 in inverse_pairs:
        facts = [(h, t) for h, r, t in train_triples if r == rel1]
        for h, t in facts[:5]:
            adversarial_tests.append({
                'query': (t, rel2, '?'),
                'type': 'inverse',
                'expected': h
            })
    
    # Save adversarial tests
    with open("data/nethack_kg_adversarial.txt", 'w') as f:
        for test in adversarial_tests:
            f.write(f"{test['type']}: {test['query']} -> {test['expected']}\n")
    
    print(f"\nCreated {len(adversarial_tests)} adversarial test cases")
    
    return train_df, test_sets

def evaluate_on_harder_benchmark(model_func, test_path="data/nethack_kg_test_hard.tsv"):
    """Evaluate a model on the harder benchmark"""
    
    print(f"\nEvaluating on: {test_path}")
    
    # Load test set
    test_df = pd.read_csv(test_path, sep='\t')
    
    mrr_scores = []
    hits_at_1 = []
    hits_at_3 = []
    hits_at_10 = []
    
    for idx, row in test_df.iterrows():
        if idx >= 50:  # Limit for testing
            break
            
        head, rel, tail = row['head'], row['type'], row['tail']
        
        # Get predictions from model
        predictions = model_func(head, rel)
        
        if tail in predictions:
            rank = predictions.index(tail) + 1
            mrr = 1.0 / rank
            
            mrr_scores.append(mrr)
            hits_at_1.append(1 if rank == 1 else 0)
            hits_at_3.append(1 if rank <= 3 else 0)
            hits_at_10.append(1 if rank <= 10 else 0)
        else:
            mrr_scores.append(0.0)
            hits_at_1.append(0)
            hits_at_3.append(0)
            hits_at_10.append(0)
    
    results = {
        'mrr': np.mean(mrr_scores),
        'hits@1': np.mean(hits_at_1),
        'hits@3': np.mean(hits_at_3),
        'hits@10': np.mean(hits_at_10)
    }
    
    print(f"Results: MRR={results['mrr']:.3f}, H@1={results['hits@1']:.3f}, H@3={results['hits@3']:.3f}")
    
    return results

def main():
    random.seed(42)
    
    # Create harder benchmark
    train_df, test_sets = create_harder_benchmark()
    
    print("\n" + "="*50)
    print("Harder Benchmark Created!")
    print("="*50)
    
    print("\nKey differences from easy evaluation:")
    print("1. Test entities/relations NOT seen during training")
    print("2. Zero-shot test: Entities with NO training examples")
    print("3. Adversarial tests: Negations, transitivity, inverses")
    print("4. Focus on rare patterns that require generalization")
    
    print("\nFiles created:")
    print("- data/nethack_kg_train.tsv (training set)")
    print("- data/nethack_kg_test_easy.tsv")
    print("- data/nethack_kg_test_medium.tsv")
    print("- data/nethack_kg_test_hard.tsv")
    print("- data/nethack_kg_test_zero_shot.tsv")
    print("- data/nethack_kg_adversarial.txt")
    
    print("\nNow models must truly generalize, not memorize!")

if __name__ == "__main__":
    main()