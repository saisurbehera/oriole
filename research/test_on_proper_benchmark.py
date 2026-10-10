#!/usr/bin/env python3
"""
Test ULTRA and heuristics on the PROPER benchmark
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
from easydict import EasyDict as edict

# Add ULTRA path
sys.path.append('ultra')
from ultra.models import Ultra
from ultra.tasks import build_relation_graph

class TestModelsOnProperBenchmark:
    def __init__(self, train_path="data/nethack_kg_train_proper.tsv"):
        print("Loading proper benchmark training data...")
        self.train_kg = pd.read_csv(train_path, sep='\t')
        print(f"Training set: {len(self.train_kg)} triples")
        
        # Build structures
        self.entity_facts = defaultdict(list)
        self.relation_facts = defaultdict(list)
        entities = set()
        relations = set()
        
        for _, row in self.train_kg.iterrows():
            head, rel, tail = row['head'], row['type'], row['tail']
            self.entity_facts[head].append((rel, tail))
            self.relation_facts[rel].append((head, tail))
            entities.add(head)
            entities.add(tail)
            relations.add(rel)
        
        self.entities = sorted(list(entities))
        self.relations = sorted(list(relations))
        self.entity2id = {e: i for i, e in enumerate(self.entities)}
        self.relation2id = {r: i for i, r in enumerate(self.relations)}
        
        print(f"Entities: {len(self.entities)}, Relations: {len(self.relations)}")
    
    def test_heuristics(self, head, relation, max_candidates=10):
        """Simple heuristic baseline"""
        candidates = list(set([t for h, t in self.relation_facts.get(relation, [])]))
        
        if not candidates:
            return []
        
        scores = {}
        for candidate in candidates:
            score = 0.0
            
            # Frequency
            relation_tails = [t for h, t in self.relation_facts.get(relation, [])]
            freq = relation_tails.count(candidate)
            score += freq * 2.0
            
            # Co-occurrence
            for rel, tail in self.entity_facts.get(head, []):
                if tail == candidate:
                    score += 1.0
            
            scores[candidate] = score
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cand for cand, score in ranked[:max_candidates]]
    
    def evaluate_model(self, model_func, test_path, test_name):
        """Evaluate any model on test set"""
        print(f"\n{'='*50}")
        print(f"Evaluating {test_name}")
        print(f"{'='*50}")
        
        test_kg = pd.read_csv(test_path, sep='\t')
        print(f"Test set: {len(test_kg)} triples")
        
        # Sample for evaluation
        sample_size = min(100, len(test_kg))
        test_sample = test_kg.sample(n=sample_size, random_state=42)
        
        mrr_scores = []
        hits_at_1 = []
        hits_at_3 = []
        hits_at_10 = []
        
        for _, row in test_sample.iterrows():
            head, rel, tail = row['head'], row['type'], row['tail']
            
            predictions = model_func(head, rel)
            
            if not predictions:
                mrr_scores.append(0.0)
                hits_at_1.append(0)
                hits_at_3.append(0)
                hits_at_10.append(0)
                continue
            
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
        
        print(f"\nResults:")
        print(f"  MRR: {results['mrr']:.3f}")
        print(f"  Hits@1: {results['hits@1']:.3f}")
        print(f"  Hits@3: {results['hits@3']:.3f}")
        print(f"  Hits@10: {results['hits@10']:.3f}")
        
        return results

def main():
    print("🔥 Testing on PROPER Benchmark")
    print("="*50)
    
    # Initialize
    tester = TestModelsOnProperBenchmark()
    
    # Test heuristics
    print("\n1. Testing Heuristics:")
    heuristic_results = {}
    for difficulty in ['easy', 'medium', 'hard']:
        test_path = f"data/nethack_kg_test_{difficulty}_proper.tsv"
        if os.path.exists(test_path):
            results = tester.evaluate_model(
                tester.test_heuristics, 
                test_path, 
                f"Heuristics on {difficulty.upper()}"
            )
            heuristic_results[difficulty] = results['mrr']
    
    print("\n" + "="*50)
    print("SUMMARY - PROPER BENCHMARK:")
    print("\nHeuristics Performance:")
    for diff, mrr in heuristic_results.items():
        print(f"  {diff}: {mrr:.3f}")
    print(f"  Average: {np.mean(list(heuristic_results.values())):.3f}")
    
    print("\nComparison to buggy benchmark:")
    print("  Buggy benchmark: 0.034 MRR (unfair - entities removed)")
    print(f"  Proper benchmark: {heuristic_results.get('hard', 0):.3f} MRR (fair evaluation)")
    
    print("\nNote: Now you can run ULTRA on this proper benchmark")
    print("and it should perform much better since entities are in vocabulary!")

if __name__ == "__main__":
    main()