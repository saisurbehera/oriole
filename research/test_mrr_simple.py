#!/usr/bin/env python3
"""
Simple MRR test for NetHack A+ KG
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def test_mrr_simple(kg):
    """Simple MRR test using frequency baseline"""
    print("Testing MRR with simple baseline...")
    
    # Get entities with multiple facts
    entity_facts = defaultdict(list)
    for h, r, t in kg.triples:
        entity_facts[h].append((r, t))
    
    # Filter entities with at least 3 facts
    test_entities = {e: facts for e, facts in entity_facts.items() if len(facts) >= 3}
    print(f"Testing on {len(test_entities)} entities with 3+ facts")
    
    # Create test cases: hide one fact, try to predict it
    test_cases = []
    for entity, facts in list(test_entities.items())[:20]:  # Test on 20 entities
        for i, (hidden_r, hidden_t) in enumerate(facts[:3]):  # Test first 3 facts
            test_cases.append({
                'entity': entity,
                'hidden_relation': hidden_r,
                'hidden_tail': hidden_t,
                'other_facts': [f for j, f in enumerate(facts) if j != i]
            })
    
    print(f"Created {len(test_cases)} test cases")
    
    # Simple frequency baseline
    relation_counts = defaultdict(int)
    for h, r, t in kg.triples:
        relation_counts[r] += 1
    
    # Sort relations by frequency
    sorted_relations = sorted(relation_counts.keys(), key=lambda x: relation_counts[x], reverse=True)
    
    # Calculate MRR
    reciprocal_ranks = []
    for case in test_cases:
        target_relation = case['hidden_relation']
        if target_relation in sorted_relations:
            rank = sorted_relations.index(target_relation) + 1
            reciprocal_ranks.append(1.0 / rank)
    
    mrr = np.mean(reciprocal_ranks)
    print(f"Simple baseline MRR: {mrr:.4f}")
    
    # Show some example predictions
    print("\nExample predictions:")
    for i, case in enumerate(test_cases[:5]):
        entity = case['entity']
        target_rel = case['hidden_relation']
        rank = sorted_relations.index(target_rel) + 1 if target_rel in sorted_relations else len(sorted_relations)
        print(f"  {entity} --?--> : predicted {sorted_relations[0]} (target: {target_rel}, rank: {rank})")
    
    return mrr

def main():
    print("NETHACK A+ KG - SIMPLE MRR TEST")
    print("="*50)
    
    # Load KG
    df = pd.read_csv("data/nethack_kg_final_a_plus.tsv", sep='\t')
    print(f"Loaded {len(df)} triples from A+ KG")
    
    # Convert to simple format
    triples = []
    for _, row in df.iterrows():
        triples.append((str(row['head']), str(row['type']), str(row['tail'])))
    
    # Create simple KG structure
    kg = type('SimpleKG', (), {})()
    kg.triples = triples[:5000]  # Use subset for speed
    
    # Test MRR
    mrr = test_mrr_simple(kg)
    
    print(f"\nFinal MRR: {mrr:.4f}")
    print("\nNote: This is a simple baseline. ULTRA would achieve much higher MRR!")

if __name__ == "__main__":
    main()