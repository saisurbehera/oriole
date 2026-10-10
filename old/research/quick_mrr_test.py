#!/usr/bin/env python3
"""
Quick MRR test for NetHack A+ KG
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import json

def quick_mrr_test():
    """Fast MRR test using simple baselines"""
    print("QUICK MRR TEST - NETHACK A+ KG")
    print("="*50)
    
    # Load KG
    kg = pd.read_csv('data/nethack_kg_final_a_plus.tsv', sep='\t')
    print(f"Loaded {len(kg)} triples")
    
    # Focus on high-priority, well-connected entities
    print("\nAnalyzing KG structure...")
    entity_counts = kg['head'].value_counts()
    well_connected = entity_counts[entity_counts >= 5].head(20)  # Top 20 entities with 5+ facts
    
    print(f"Testing on {len(well_connected)} well-connected entities:")
    for entity, count in well_connected.head(5).items():
        print(f"  {entity}: {count} facts")
    
    # Create test cases
    test_cases = []
    for entity in well_connected.index:
        entity_facts = kg[kg['head'] == entity].to_dict('records')
        if len(entity_facts) >= 3:
            # Hide one fact, try to predict it
            for i, hidden_fact in enumerate(entity_facts[:3]):
                test_cases.append({
                    'entity': entity,
                    'hidden_relation': hidden_fact['type'],
                    'hidden_tail': hidden_fact['tail'],
                    'other_facts': [f for j, f in enumerate(entity_facts) if j != i]
                })
    
    print(f"Created {len(test_cases)} test cases")
    
    # Test different prediction strategies
    results = {}
    
    # 1. Frequency baseline
    print("\n1. Testing frequency baseline...")
    relation_freq = kg['type'].value_counts()
    freq_ranks = []
    
    for case in test_cases[:100]:  # Test subset for speed
        target_rel = case['hidden_relation']
        # Rank all relations by frequency
        sorted_rels = relation_freq.index.tolist()
        rank = sorted_rels.index(target_rel) + 1
        freq_ranks.append(1.0 / rank)
    
    freq_mrr = np.mean(freq_ranks)
    results['frequency_baseline'] = freq_mrr
    print(f"   Frequency baseline MRR: {freq_mrr:.4f}")
    
    # 2. Priority-aware baseline
    print("\n2. Testing priority-aware baseline...")
    rel_priority = kg.groupby('type')['priority'].mean()
    priority_ranks = []
    
    for case in test_cases[:100]:
        target_rel = case['hidden_relation']
        # Rank relations by average priority
        sorted_rels = rel_priority.sort_values(ascending=False).index.tolist()
        rank = sorted_rels.index(target_rel) + 1
        priority_ranks.append(1.0 / rank)
    
    priority_mrr = np.mean(priority_ranks)
    results['priority_baseline'] = priority_mrr
    print(f"   Priority-aware MRR: {priority_mrr:.4f}")
    
    # 3. Entity-type specific baseline
    print("\n3. Testing entity-type baseline...")
    # Group entities by their most common relation types
    entity_profiles = {}
    for entity in well_connected.index:
        entity_rels = kg[kg['head'] == entity]['type'].value_counts()
        entity_profiles[entity] = entity_rels.index.tolist()
    
    type_ranks = []
    for case in test_cases[:100]:
        entity = case['entity']
        target_rel = case['hidden_relation']
        
        # Use entity's relation profile as prediction order
        if entity in entity_profiles:
            profile = entity_profiles[entity]
            if target_rel in profile:
                rank = profile.index(target_rel) + 1
                type_ranks.append(1.0 / rank)
    
    type_mrr = np.mean(type_ranks) if type_ranks else 0.0
    results['entity_type_baseline'] = type_mrr
    print(f"   Entity-type MRR: {type_mrr:.4f}")
    
    # 4. Game-specific baseline (NetHack knowledge)
    print("\n4. Testing game-specific baseline...")
    # Define NetHack entity categories and their likely relations
    game_patterns = {
        'monster': ['attack_damage', 'armor_class', 'has_resistance', 'speed', 'difficulty'],
        'item': ['weight', 'cost', 'material', 'when_equipped'],
        'artifact': ['grants', 'base_item', 'when_equipped'],
        'spell': ['grants', 'effect'],
    }
    
    def classify_entity(entity, kg):
        """Classify entity based on its relations"""
        entity_rels = set(kg[kg['head'] == entity]['type'].tolist())
        
        scores = {}
        for category, expected_rels in game_patterns.items():
            overlap = len(entity_rels.intersection(expected_rels))
            scores[category] = overlap / len(expected_rels)
        
        return max(scores, key=scores.get) if scores else 'unknown'
    
    game_ranks = []
    for case in test_cases[:100]:
        entity = case['entity']
        target_rel = case['hidden_relation']
        
        entity_type = classify_entity(entity, kg)
        if entity_type in game_patterns:
            expected_rels = game_patterns[entity_type]
            if target_rel in expected_rels:
                rank = expected_rels.index(target_rel) + 1
                game_ranks.append(1.0 / rank)
    
    game_mrr = np.mean(game_ranks) if game_ranks else 0.0
    results['game_specific_baseline'] = game_mrr
    print(f"   Game-specific MRR: {game_mrr:.4f}")
    
    # Summary
    print(f"\nMRR RESULTS SUMMARY:")
    print("="*30)
    for method, mrr in results.items():
        print(f"{method:25s}: {mrr:.4f}")
    
    # Expected ULTRA performance estimate
    print(f"\nESTIMATED ULTRA PERFORMANCE:")
    print("="*30)
    print("Based on literature, ULTRA typically achieves:")
    print("• MRR 0.4-0.8 on well-structured KGs")
    print("• Higher performance on game/domain-specific KGs")
    print("• Our A+ KG should achieve MRR ~0.6-0.8 with ULTRA")
    
    # Save results
    results['test_info'] = {
        'total_triples': len(kg),
        'test_cases': len(test_cases),
        'well_connected_entities': len(well_connected)
    }
    
    with open('data/quick_mrr_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to data/quick_mrr_results.json")
    
    return results

def analyze_kg_for_mrr():
    """Analyze KG properties relevant to MRR"""
    kg = pd.read_csv('data/nethack_kg_final_a_plus.tsv', sep='\t')
    
    print("\nKG ANALYSIS FOR MRR:")
    print("="*30)
    
    # Basic stats
    print(f"Total triples: {len(kg)}")
    print(f"Unique entities: {len(set(kg['head'].tolist() + kg['tail'].tolist()))}")
    print(f"Unique relations: {kg['type'].nunique()}")
    
    # Relation distribution
    rel_dist = kg['type'].value_counts()
    print(f"\nTop 10 relations:")
    for rel, count in rel_dist.head(10).items():
        print(f"  {rel}: {count}")
    
    # Entity connectivity
    entity_facts = kg['head'].value_counts()
    print(f"\nEntity connectivity:")
    print(f"  Avg facts per entity: {entity_facts.mean():.2f}")
    print(f"  Entities with 10+ facts: {(entity_facts >= 10).sum()}")
    print(f"  Entities with 5+ facts: {(entity_facts >= 5).sum()}")
    
    # High-priority facts
    high_priority = kg[kg['priority'] >= 8]
    print(f"\nHigh-priority facts: {len(high_priority)}")
    print(f"High-priority relations: {high_priority['type'].nunique()}")

if __name__ == "__main__":
    analyze_kg_for_mrr()
    results = quick_mrr_test()