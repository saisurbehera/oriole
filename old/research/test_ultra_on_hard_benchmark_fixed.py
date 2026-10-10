#!/usr/bin/env python3
"""
Test real ULTRA on the harder benchmark
This will show ULTRA's true generalization ability
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

class ULTRAHardBenchmark:
    def __init__(self, train_path="data/nethack_kg_train.tsv"):
        print("🚀 Initializing ULTRA for hard benchmark")
        print("="*50)
        
        # Load training data only
        print("Loading training data...")
        self.train_kg = pd.read_csv(train_path, sep='\t')
        print(f"Training set: {len(self.train_kg)} triples")
        
        # Build entity and relation mappings from TRAINING data
        entities = set()
        relations = set()
        
        for _, row in self.train_kg.iterrows():
            entities.add(row['head'])
            entities.add(row['tail'])
            relations.add(row['type'])
        
        self.entities = sorted(list(entities))
        self.relations = sorted(list(relations))
        
        self.entity2id = {e: i for i, e in enumerate(self.entities)}
        self.relation2id = {r: i for i, r in enumerate(self.relations)}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.id2relation = {i: r for r, i in self.relation2id.items()}
        
        print(f"Training entities: {len(self.entities)}, relations: {len(self.relations)}")
        
        # Build PyG data
        self.kg_data = self._build_pyg_data()
        
        # Initialize ULTRA
        self.model = self._initialize_ultra()
        
    def _build_pyg_data(self):
        """Build PyG data from training set only"""
        edge_list = []
        edge_type = []
        
        for _, row in self.train_kg.iterrows():
            if row['head'] in self.entity2id and row['tail'] in self.entity2id:
                h_id = self.entity2id[row['head']]
                r_id = self.relation2id[row['type']]
                t_id = self.entity2id[row['tail']]
                
                edge_list.append([h_id, t_id])
                edge_type.append(r_id)
        
        # Add self-loops
        self_loop_rel_id = len(self.relations)
        for i in range(len(self.entities)):
            edge_list.append([i, i])
            edge_type.append(self_loop_rel_id)
        
        # Add inverse relations
        num_direct_relations = len(self.relations) + 1
        inverse_edge_list = []
        inverse_edge_type = []
        
        for i, (h_id, t_id) in enumerate(edge_list):
            if edge_type[i] != self_loop_rel_id:
                inverse_edge_list.append([t_id, h_id])
                inverse_edge_type.append(edge_type[i] + num_direct_relations)
        
        # Combine
        all_edges = edge_list + inverse_edge_list
        all_edge_types = edge_type + inverse_edge_type
        
        edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        edge_type_tensor = torch.tensor(all_edge_types, dtype=torch.long)
        
        data = Data(
            edge_index=edge_index,
            edge_type=edge_type_tensor,
            num_nodes=len(self.entities)
        )
        data.num_relations = num_direct_relations * 2
        
        return data
    
    def _initialize_ultra(self):
        """Initialize ULTRA model"""
        print("\nInitializing ULTRA model...")
        
        # Build relation graph
        kg_data = build_relation_graph(self.kg_data)
        
        # Config
        config = edict({
            'model': {
                'relation_graph': {
                    'class': 'RelNBFNet',
                    'input_dim': 64,
                    'hidden_dims': [64, 64, 64, 64, 64, 64],
                    'message_func': 'distmult',
                    'aggregate_func': 'sum',
                    'short_cut': True,
                    'layer_norm': True
                },
                'entity_graph': {
                    'class': 'EntityNBFNet',
                    'input_dim': 64,
                    'hidden_dims': [64, 64, 64, 64, 64, 64],
                    'message_func': 'distmult',
                    'aggregate_func': 'sum',
                    'short_cut': True,
                    'layer_norm': True
                }
            }
        })
        
        # Initialize
        model = Ultra(
            rel_model_cfg=config.model.relation_graph,
            entity_model_cfg=config.model.entity_graph
        )
        
        # Load checkpoint
        checkpoint_path = "ultra/ckpts/ultra_4g.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            print("✓ Checkpoint loaded")
        
        model.eval()
        return model
    
    def predict(self, head, relation, max_candidates=20):
        """ULTRA prediction for unseen test queries"""
        
        # Check if entities/relations are in training vocabulary
        if head not in self.entity2id:
            # Unknown entity - ULTRA can't handle this directly
            return []
        
        if relation not in self.relation2id:
            # Unknown relation - ULTRA can't handle this
            return []
        
        head_id = self.entity2id[head]
        rel_id = self.relation2id[relation]
        
        # Get all possible candidates from training vocab
        candidate_ids = list(range(len(self.entities)))
        
        # Score all candidates
        queries = torch.tensor([[head_id, rel_id, t_id] for t_id in candidate_ids], dtype=torch.long)
        
        with torch.no_grad():
            try:
                scores = self.model(self.kg_data, queries)
                scores = scores.cpu().numpy()
            except Exception as e:
                print(f"ULTRA error: {e}")
                return []
        
        # Sort by score
        scored_candidates = [(candidate_ids[i], scores[i]) for i in range(len(candidate_ids))]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        top_candidates = [self.id2entity[cid] for cid, _ in scored_candidates[:max_candidates]]
        return top_candidates
    
    def evaluate_on_test_set(self, test_path, test_name):
        """Evaluate ULTRA on a test set"""
        
        print(f"\n{'='*50}")
        print(f"Evaluating ULTRA on {test_name}")
        print(f"{'='*50}")
        
        test_kg = pd.read_csv(test_path, sep='\t')
        print(f"Test set: {len(test_kg)} triples")
        
        # Count unseen entities/relations
        unseen_entities = 0
        unseen_relations = 0
        
        for _, row in test_kg.iterrows():
            if row['head'] not in self.entity2id or row['tail'] not in self.entity2id:
                unseen_entities += 1
            if row['type'] not in self.relation2id:
                unseen_relations += 1
        
        print(f"Unseen entities: {unseen_entities}/{len(test_kg)} ({unseen_entities/len(test_kg)*100:.1f}%)")
        print(f"Unseen relations: {unseen_relations}/{len(test_kg)} ({unseen_relations/len(test_kg)*100:.1f}%)")
        
        # Sample for evaluation
        sample_size = min(100, len(test_kg))
        test_sample = test_kg.sample(n=sample_size, random_state=42)
        
        mrr_scores = []
        hits_at_1 = []
        hits_at_3 = []
        hits_at_10 = []
        skipped = 0
        
        for idx, row in test_sample.iterrows():
            head, rel, tail = row['head'], row['type'], row['tail']
            
            # Skip if entities not in training vocab
            if head not in self.entity2id or tail not in self.entity2id or rel not in self.relation2id:
                skipped += 1
                continue
            
            predictions = self.predict(head, rel)
            
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
        
        if mrr_scores:
            print(f"\nResults (on {len(mrr_scores)} valid queries):")
            print(f"  MRR: {np.mean(mrr_scores):.3f}")
            print(f"  Hits@1: {np.mean(hits_at_1):.3f}")
            print(f"  Hits@3: {np.mean(hits_at_3):.3f}")
            print(f"  Hits@10: {np.mean(hits_at_10):.3f}")
            print(f"  Skipped: {skipped}/{sample_size} (unseen entities/relations)")
        else:
            print("No valid predictions!")
        
        return np.mean(mrr_scores) if mrr_scores else 0.0

def main():
    print("🔥 Testing Real ULTRA on Harder Benchmark")
    print("="*50)
    
    # Initialize ULTRA
    ultra = ULTRAHardBenchmark()
    
    # Test on different difficulty levels
    easy_mrr = ultra.evaluate_on_test_set("data/nethack_kg_test_easy.tsv", "EASY Test Set")
    medium_mrr = ultra.evaluate_on_test_set("data/nethack_kg_test_medium.tsv", "MEDIUM Test Set")
    hard_mrr = ultra.evaluate_on_test_set("data/nethack_kg_test_hard.tsv", "HARD Test Set")
    zero_mrr = ultra.evaluate_on_test_set("data/nethack_kg_test_zero_shot.tsv", "ZERO-SHOT Test Set")
    
    print("\n" + "="*50)
    print("ULTRA PERFORMANCE SUMMARY:")
    print(f"  Easy: {easy_mrr:.3f}")
    print(f"  Medium: {medium_mrr:.3f}")
    print(f"  Hard: {hard_mrr:.3f}")
    print(f"  Zero-shot: {zero_mrr:.3f}")
    print(f"  Average (excl. zero-shot): {np.mean([easy_mrr, medium_mrr, hard_mrr]):.3f}")
    
    print("\n" + "="*50)
    print("COMPARISON:")
    print("  Heuristics on 'easy' eval: 0.685 (memorization)")
    print("  Heuristics on hard: 0.034")
    print(f"  ULTRA on hard: {hard_mrr:.3f}")
    print("  Original ULTRA: 0.603 (on different test set)")

if __name__ == "__main__":
    main()