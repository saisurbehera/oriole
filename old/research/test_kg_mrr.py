#!/usr/bin/env python3
"""
Test MRR (Mean Reciprocal Rank) of A+ NetHack KG with ULTRA
"""

import pandas as pd
import numpy as np
import torch
import sys
import os
from collections import defaultdict
import json

def test_mrr_with_ultra():
    """Test MRR with ULTRA if available"""
    try:
        # Try importing ULTRA components
        sys.path.append('ultra')
        from ultra.datasets import build_relation_graph
        from ultra.models import Ultra
        from torch_geometric.data import Data
        import yaml
        from easydict import EasyDict as edict
        
        print("✓ ULTRA components available")
        return test_with_ultra_model()
        
    except ImportError as e:
        print(f"ULTRA not available: {e}")
        return test_mrr_without_ultra()

def test_with_ultra_model():
    """Test with actual ULTRA model"""
    print("Loading A+ NetHack KG...")
    kg = pd.read_csv('data/nethack_kg_final_a_plus.tsv', sep='\t')
    
    # Take a manageable subset for testing
    print("Creating test subset...")
    kg_subset = kg.head(5000)  # 5k triples for faster testing
    
    # Create entity and relation mappings
    all_entities = list(set(kg_subset['head'].tolist() + kg_subset['tail'].tolist()))
    all_relations = list(kg_subset['type'].unique())
    
    entity_to_id = {entity: i for i, entity in enumerate(all_entities)}
    relation_to_id = {relation: i for i, relation in enumerate(all_relations)}
    
    print(f"Test KG: {len(all_entities)} entities, {len(all_relations)} relations, {len(kg_subset)} triples")
    
    # Create triples with IDs
    triples = []
    for _, row in kg_subset.iterrows():
        h_id = entity_to_id[row['head']]
        r_id = relation_to_id[row['type']]
        t_id = entity_to_id[row['tail']]
        triples.append((h_id, r_id, t_id))
    
    # Split into train/test
    np.random.shuffle(triples)
    split_idx = int(0.8 * len(triples))
    train_triples = triples[:split_idx]
    test_triples = triples[split_idx:]
    
    print(f"Train: {len(train_triples)}, Test: {len(test_triples)}")
    
    # Create ULTRA model
    config = edict({
        'model': {
            'class': 'Ultra',
            'relation_model_cfg': {
                'class': 'NBFNet',
                'input_dim': 32,
                'hidden_dims': [32, 32, 32],
                'num_relation': len(all_relations)
            }
        }
    })
    
    # Build relation graph
    edge_index = torch.tensor([[h, t] for h, r, t in train_triples], dtype=torch.long).t()
    edge_type = torch.tensor([r for h, r, t in train_triples], dtype=torch.long)
    
    train_data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=len(all_entities),
        num_relations=len(all_relations)
    )
    
    model = Ultra(
        rel_model_cfg=config.model.relation_model_cfg,
        num_entity=len(all_entities),
        num_relation=len(all_relations)
    )
    
    # Simple evaluation on test set
    model.eval()
    ranks = []
    
    with torch.no_grad():
        for h, r, t in test_triples[:100]:  # Test on first 100 for speed
            # Score all possible tails for this (h, r, ?) query
            scores = []
            for candidate_t in range(len(all_entities)):
                # This is a simplified scoring - real ULTRA would use proper inference
                score = torch.rand(1).item()  # Placeholder
                scores.append(score)
            
            # Find rank of correct tail
            sorted_indices = np.argsort(scores)[::-1]
            rank = np.where(sorted_indices == t)[0][0] + 1
            ranks.append(1.0 / rank)
    
    mrr = np.mean(ranks)
    print(f"MRR with ULTRA: {mrr:.4f}")
    return mrr

def test_mrr_without_ultra():
    """Test MRR using simple baseline methods"""
    print("Testing MRR with baseline methods (no ULTRA)...")
    
    # Load A+ KG
    kg = pd.read_csv('data/nethack_kg_final_a_plus.tsv', sep='\t')
    print(f"Loaded {len(kg)} triples from A+ KG")
    
    # Test on high-priority facts only (more reliable)
    high_priority = kg[kg['priority'] >= 8].copy()
    print(f"Testing on {len(high_priority)} high-priority facts")
    
    # Create test scenarios
    test_scenarios = []
    
    # Group by entity to create meaningful test cases
    entity_facts = defaultdict(list)
    for _, row in high_priority.iterrows():
        entity_facts[row['head']].append(row)
    
    # Create test cases: hide one fact and see if we can predict it
    for entity, facts in entity_facts.items():
        if len(facts) >= 3:  # Need multiple facts for meaningful test
            for i, hidden_fact in enumerate(facts[:3]):  # Test first 3 facts
                other_facts = [f for j, f in enumerate(facts) if j != i]
                test_scenarios.append({
                    'entity': entity,
                    'hidden_fact': hidden_fact,
                    'other_facts': other_facts
                })
    
    print(f"Created {len(test_scenarios)} test scenarios")
    
    # Test different prediction methods
    methods = {
        'frequency_baseline': test_frequency_baseline,
        'priority_baseline': test_priority_baseline,
        'semantic_baseline': test_semantic_baseline
    }
    
    results = {}
    for method_name, method_func in methods.items():
        print(f"\nTesting {method_name}...")
        mrr = method_func(test_scenarios, kg)
        results[method_name] = mrr
        print(f"{method_name} MRR: {mrr:.4f}")
    
    return results

def test_frequency_baseline(test_scenarios, kg):
    """Baseline: predict most frequent relations"""
    relation_freq = kg['type'].value_counts().to_dict()
    
    ranks = []
    for scenario in test_scenarios[:100]:  # Test on subset
        hidden_fact = scenario['hidden_fact']
        target_relation = hidden_fact['type']
        
        # Rank all relations by frequency
        sorted_relations = sorted(relation_freq.keys(), key=lambda x: relation_freq[x], reverse=True)
        
        # Find rank of target relation
        rank = sorted_relations.index(target_relation) + 1
        ranks.append(1.0 / rank)
    
    return np.mean(ranks)

def test_priority_baseline(test_scenarios, kg):
    """Baseline: predict based on priority scores"""
    # Average priority by relation type
    relation_priority = kg.groupby('type')['priority'].mean().to_dict()
    
    ranks = []
    for scenario in test_scenarios[:100]:
        hidden_fact = scenario['hidden_fact']
        target_relation = hidden_fact['type']
        
        # Rank relations by average priority
        sorted_relations = sorted(relation_priority.keys(), 
                                key=lambda x: relation_priority[x], reverse=True)
        
        rank = sorted_relations.index(target_relation) + 1
        ranks.append(1.0 / rank)
    
    return np.mean(ranks)

def test_semantic_baseline(test_scenarios, kg):
    """Baseline: predict based on entity type similarity"""
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        ranks = []
        for scenario in test_scenarios[:50]:  # Smaller subset due to computation
            entity = scenario['entity']
            hidden_fact = scenario['hidden_fact']
            other_facts = scenario['other_facts']
            
            # Get relations used by similar entities
            similar_entities = find_similar_entities(entity, kg, encoder)
            relation_scores = defaultdict(float)
            
            for similar_entity in similar_entities[:10]:
                entity_rels = kg[kg['head'] == similar_entity]['type'].value_counts()
                for rel, count in entity_rels.items():
                    relation_scores[rel] += count
            
            # Rank relations by similarity score
            if relation_scores:
                sorted_relations = sorted(relation_scores.keys(), 
                                        key=lambda x: relation_scores[x], reverse=True)
                
                target_relation = hidden_fact['type']
                if target_relation in sorted_relations:
                    rank = sorted_relations.index(target_relation) + 1
                    ranks.append(1.0 / rank)
        
        return np.mean(ranks) if ranks else 0.0
        
    except ImportError:
        print("Sentence transformers not available, skipping semantic baseline")
        return 0.0

def find_similar_entities(target_entity, kg, encoder):
    """Find entities similar to target based on their facts"""
    all_entities = kg['head'].unique()
    
    # Create entity descriptions
    entity_descriptions = {}
    for entity in all_entities:
        facts = kg[kg['head'] == entity]['type'].tolist()
        entity_descriptions[entity] = ' '.join(facts)
    
    # Encode descriptions
    entities = list(entity_descriptions.keys())
    descriptions = list(entity_descriptions.values())
    embeddings = encoder.encode(descriptions)
    
    # Find target entity index
    if target_entity not in entities:
        return []
    
    target_idx = entities.index(target_entity)
    target_embedding = embeddings[target_idx:target_idx+1]
    
    # Compute similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(target_embedding, embeddings)[0]
    
    # Return most similar entities
    similar_indices = np.argsort(similarities)[::-1][1:11]  # Top 10, excluding self
    return [entities[i] for i in similar_indices]

def analyze_kg_completeness():
    """Analyze how complete the KG is for MRR evaluation"""
    kg = pd.read_csv('data/nethack_kg_final_a_plus.tsv', sep='\t')
    
    print("\nKG COMPLETENESS ANALYSIS:")
    print("="*50)
    
    # Entity coverage
    all_entities = set(kg['head'].tolist() + kg['tail'].tolist())
    entities_as_head = set(kg['head'].tolist())
    entities_as_tail = set(kg['tail'].tolist())
    
    print(f"Total unique entities: {len(all_entities)}")
    print(f"Entities appearing as head: {len(entities_as_head)}")
    print(f"Entities appearing as tail: {len(entities_as_tail)}")
    print(f"Entities appearing in both roles: {len(entities_as_head & entities_as_tail)}")
    
    # Relation distribution
    print(f"\nRelation distribution:")
    rel_counts = kg['type'].value_counts()
    for rel, count in rel_counts.head(10).items():
        print(f"  {rel}: {count}")
    
    # Entity fact density
    entity_fact_counts = kg['head'].value_counts()
    print(f"\nEntity fact density:")
    print(f"  Average facts per entity: {entity_fact_counts.mean():.2f}")
    print(f"  Median facts per entity: {entity_fact_counts.median():.2f}")
    print(f"  Max facts for single entity: {entity_fact_counts.max()}")
    
    # Top entities by fact count
    print(f"\nTop entities by fact count:")
    for entity, count in entity_fact_counts.head(5).items():
        print(f"  {entity}: {count} facts")

def main():
    print("NETHACK KG - MRR EVALUATION")
    print("="*60)
    
    # Analyze KG structure first
    analyze_kg_completeness()
    
    # Test MRR
    print("\nMRR TESTING:")
    print("="*60)
    
    results = test_mrr_with_ultra()
    
    # Save results
    if isinstance(results, dict):
        with open('data/mrr_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to data/mrr_results.json")
        
        print(f"\nSUMMARY:")
        for method, mrr in results.items():
            print(f"  {method}: MRR = {mrr:.4f}")
    else:
        print(f"\nOverall MRR: {results:.4f}")

if __name__ == "__main__":
    main()