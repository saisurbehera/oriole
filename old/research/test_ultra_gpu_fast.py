#!/usr/bin/env python3
"""
Fast GPU-enabled ULTRA test on proper benchmark
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
from easydict import EasyDict as edict
import time

# Add ULTRA path
sys.path.append('ultra')
from ultra.models import Ultra
from ultra.tasks import build_relation_graph

class ULTRAGPUFast:
    def __init__(self, train_path="data/nethack_kg_train_proper.tsv"):
        print("🚀 Initializing ULTRA with GPU acceleration")
        print("="*50)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load training data
        print("\nLoading training data...")
        self.train_kg = pd.read_csv(train_path, sep='\t')
        print(f"Training set: {len(self.train_kg)} triples")
        
        # Build mappings
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
        
        print(f"Entities: {len(self.entities)}, Relations: {len(self.relations)}")
        
        # Build PyG data
        self.kg_data = self._build_pyg_data()
        
        # Initialize ULTRA
        self.model = self._initialize_ultra()
        
        # Pre-compute relation-specific candidates for faster inference
        print("\nPre-computing relation candidates...")
        self.relation_candidates = defaultdict(set)
        for _, row in self.train_kg.iterrows():
            self.relation_candidates[row['type']].add(self.entity2id[row['tail']])
        
    def _build_pyg_data(self):
        """Build PyG data"""
        edge_list = []
        edge_type = []
        
        for _, row in self.train_kg.iterrows():
            h_id = self.entity2id[row['head']]
            r_id = self.relation2id[row['type']]
            t_id = self.entity2id[row['tail']]
            
            edge_list.append([h_id, t_id])
            edge_type.append(r_id)
        
        # Self-loops
        self_loop_rel_id = len(self.relations)
        for i in range(len(self.entities)):
            edge_list.append([i, i])
            edge_type.append(self_loop_rel_id)
        
        # Inverse relations
        num_direct_relations = len(self.relations) + 1
        inverse_edge_list = []
        inverse_edge_type = []
        
        for i, (h_id, t_id) in enumerate(edge_list):
            if edge_type[i] != self_loop_rel_id:
                inverse_edge_list.append([t_id, h_id])
                inverse_edge_type.append(edge_type[i] + num_direct_relations)
        
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
        
        # Build relation graph and move to GPU
        self.kg_data = build_relation_graph(self.kg_data)
        self.kg_data = self.kg_data.to(self.device)
        
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
        
        # Initialize and move to GPU
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
        
        model = model.to(self.device)
        model.eval()
        print(f"✓ Model on {self.device}")
        
        return model
    
    def predict_ultra_fast(self, head, relation, max_candidates=50):
        """Fast ULTRA prediction using relation-specific candidates"""
        
        if head not in self.entity2id or relation not in self.relation2id:
            return []
        
        head_id = self.entity2id[head]
        rel_id = self.relation2id[relation]
        
        # Use relation-specific candidates instead of all entities
        if relation in self.relation_candidates:
            candidate_ids = list(self.relation_candidates[relation])[:200]  # Limit candidates
        else:
            candidate_ids = list(range(min(len(self.entities), 200)))
        
        if len(candidate_ids) == 0:
            return []
        
        # Create batch
        batch_triples = []
        for tail_id in candidate_ids:
            batch_triples.append([head_id, rel_id, tail_id])
        
        batch = torch.tensor([batch_triples], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            try:
                scores = self.model(self.kg_data, batch)
                scores = scores.cpu().numpy().flatten()
            except Exception as e:
                print(f"Error: {e}")
                return []
        
        # Sort by score
        scored_candidates = [(candidate_ids[i], scores[i]) for i in range(len(scores))]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        top_candidates = [self.id2entity[cid] for cid, _ in scored_candidates[:max_candidates]]
        return top_candidates
    
    def evaluate_fast(self, test_path, test_name, max_queries=30):
        """Fast evaluation on limited queries"""
        
        print(f"\n{'='*50}")
        print(f"Evaluating ULTRA on {test_name}")
        print(f"{'='*50}")
        
        test_kg = pd.read_csv(test_path, sep='\t')
        print(f"Test set: {len(test_kg)} triples")
        print(f"Evaluating on {max_queries} queries for speed")
        
        # Sample queries
        test_sample = test_kg.sample(n=min(max_queries, len(test_kg)), random_state=42)
        
        mrr_scores = []
        hits_at_1 = []
        hits_at_3 = []
        hits_at_10 = []
        
        start_time = time.time()
        
        for idx, row in test_sample.iterrows():
            head, rel, tail = row['head'], row['type'], row['tail']
            
            if head not in self.entity2id or tail not in self.entity2id:
                continue
            
            predictions = self.predict_ultra_fast(head, rel)
            
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
        
        elapsed = time.time() - start_time
        
        if mrr_scores:
            print(f"\nResults (on {len(mrr_scores)} queries):")
            print(f"  MRR: {np.mean(mrr_scores):.3f}")
            print(f"  Hits@1: {np.mean(hits_at_1):.3f}")
            print(f"  Hits@3: {np.mean(hits_at_3):.3f}")
            print(f"  Hits@10: {np.mean(hits_at_10):.3f}")
            print(f"  Time: {elapsed:.1f}s ({elapsed/len(mrr_scores):.2f}s per query)")
        
        return np.mean(mrr_scores) if mrr_scores else 0.0

def main():
    print("🔥 Fast GPU Test of ULTRA on Proper Benchmark")
    print("="*50)
    
    # Initialize ULTRA with GPU
    ultra = ULTRAGPUFast()
    
    # Test on limited queries for speed
    results = {}
    for difficulty in ['easy', 'medium', 'hard']:
        test_path = f"data/nethack_kg_test_{difficulty}_proper.tsv"
        if os.path.exists(test_path):
            mrr = ultra.evaluate_fast(test_path, f"{difficulty.upper()} Test", max_queries=30)
            results[difficulty] = mrr
    
    print("\n" + "="*50)
    print("ULTRA vs HEURISTICS (Limited Test):")
    print("\nULTRA Performance:")
    for diff, mrr in results.items():
        print(f"  {diff}: {mrr:.3f}")
    print(f"  Average: {np.mean(list(results.values())):.3f}")
    
    print("\nHeuristics (full test):")
    print("  easy: 0.555")
    print("  medium: 0.404") 
    print("  hard: 0.115")
    print("  Average: 0.358")
    
    print("\nNote: This is a limited test (30 queries) for speed.")
    print("Full evaluation would take longer but be more accurate.")

if __name__ == "__main__":
    main()