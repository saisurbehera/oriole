#!/usr/bin/env python3
"""
Train ULTRA from scratch on NetHack KG - no pre-trained checkpoint
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

class ULTRAFromScratch:
    def __init__(self, train_path="data/nethack_kg_train_proper.tsv"):
        print("🚀 Initializing ULTRA from SCRATCH (no checkpoint)")
        print("="*50)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
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
        
        # Initialize ULTRA WITHOUT checkpoint
        self.model = self._initialize_ultra_no_checkpoint()
        
        # Pre-compute for faster inference
        self.relation_candidates = defaultdict(set)
        for _, row in self.train_kg.iterrows():
            self.relation_candidates[row['type']].add(self.entity2id[row['tail']])
    
    def _build_pyg_data(self):
        """Build PyG data with correct structure"""
        edge_list = []
        edge_type = []
        
        # Add KG edges
        for _, row in self.train_kg.iterrows():
            h_id = self.entity2id[row['head']]
            r_id = self.relation2id[row['type']]
            t_id = self.entity2id[row['tail']]
            
            edge_list.append([h_id, t_id])
            edge_type.append(r_id)
        
        # CRITICAL: Build simple structure first
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
        
        data = Data(
            edge_index=edge_index,
            edge_type=edge_type_tensor,
            num_nodes=len(self.entities)
        )
        
        # Set num_relations to actual number
        data.num_relations = len(self.relations)
        
        print(f"\nGraph construction:")
        print(f"  Direct relations: {data.num_relations}")
        print(f"  Edges: {edge_index.shape[1]}")
        print(f"  Max relation ID: {edge_type_tensor.max().item()}")
        
        return data
    
    def _initialize_ultra_no_checkpoint(self):
        """Initialize ULTRA without pre-trained weights"""
        print("\nInitializing ULTRA model from scratch...")
        
        # Build relation graph
        print("Building relation graph...")
        self.kg_data = build_relation_graph(self.kg_data)
        
        print(f"Relation graph nodes: {self.kg_data.relation_graph.num_nodes}")
        print(f"Expected (num_relations): {self.kg_data.num_relations}")
        
        # Move to device
        self.kg_data = self.kg_data.to(self.device)
        
        # Smaller config for testing
        config = edict({
            'model': {
                'relation_graph': {
                    'class': 'RelNBFNet',
                    'input_dim': 32,  # Smaller for faster testing
                    'hidden_dims': [32, 32, 32],  # Fewer layers
                    'message_func': 'distmult',
                    'aggregate_func': 'sum',
                    'short_cut': True,
                    'layer_norm': True
                },
                'entity_graph': {
                    'class': 'EntityNBFNet',
                    'input_dim': 32,
                    'hidden_dims': [32, 32, 32],
                    'message_func': 'distmult',
                    'aggregate_func': 'sum',
                    'short_cut': True,
                    'layer_norm': True
                }
            }
        })
        
        # Initialize model
        model = Ultra(
            rel_model_cfg=config.model.relation_graph,
            entity_model_cfg=config.model.entity_graph
        )
        
        print("✓ Model initialized with random weights")
        
        model = model.to(self.device)
        model.eval()  # Just for inference
        print(f"✓ Model on {self.device}")
        
        return model
    
    def predict_simple(self, head, relation, max_candidates=20):
        """Simple prediction without batching issues"""
        
        if head not in self.entity2id or relation not in self.relation2id:
            return []
        
        head_id = self.entity2id[head]
        rel_id = self.relation2id[relation]
        
        # Get candidates
        if relation in self.relation_candidates:
            candidate_ids = list(self.relation_candidates[relation])[:50]
        else:
            candidate_ids = list(range(min(len(self.entities), 50)))
        
        if not candidate_ids:
            return []
        
        scores = []
        
        # Score each candidate individually to avoid batch issues
        for tail_id in candidate_ids:
            # Create single query
            query = torch.tensor([[[head_id, rel_id, tail_id]]], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                try:
                    score = self.model(self.kg_data, query)
                    scores.append((tail_id, float(score.cpu().item())))
                except Exception as e:
                    # print(f"Scoring error: {e}")
                    scores.append((tail_id, -float('inf')))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        top_candidates = [self.id2entity[cid] for cid, _ in scores[:max_candidates]]
        return top_candidates
    
    def evaluate_minimal(self, test_path, test_name, max_queries=10):
        """Minimal evaluation to test if it works"""
        
        print(f"\n{'='*50}")
        print(f"Testing ULTRA (from scratch) on {test_name}")
        print(f"{'='*50}")
        
        test_kg = pd.read_csv(test_path, sep='\t')
        print(f"Test set: {len(test_kg)} triples")
        print(f"Testing on {max_queries} queries")
        
        # Sample queries
        test_sample = test_kg.sample(n=min(max_queries, len(test_kg)), random_state=42)
        
        mrr_scores = []
        successes = 0
        
        for idx, row in enumerate(test_sample.itertuples()):
            head, rel, tail = row.head, row.type, row.tail
            
            if head not in self.entity2id or tail not in self.entity2id:
                continue
            
            print(f"\nQuery {idx+1}: {head} {rel} ?")
            
            predictions = self.predict_simple(head, rel)
            
            if predictions:
                print(f"Top 3: {predictions[:3]}")
                
                if tail in predictions:
                    rank = predictions.index(tail) + 1
                    mrr = 1.0 / rank
                    mrr_scores.append(mrr)
                    successes += 1
                    print(f"✓ Found at rank {rank}")
                else:
                    mrr_scores.append(0.0)
                    print(f"✗ Not found")
            else:
                mrr_scores.append(0.0)
                print(f"✗ No predictions")
        
        if mrr_scores:
            print(f"\n{test_name} Results:")
            print(f"  MRR: {np.mean(mrr_scores):.3f}")
            print(f"  Success rate: {successes}/{len(mrr_scores)}")
        
        return np.mean(mrr_scores) if mrr_scores else 0.0

def main():
    print("🔥 Testing ULTRA from SCRATCH (no pre-trained weights)")
    print("="*50)
    
    # Initialize
    ultra = ULTRAFromScratch()
    
    # Test on small sample
    print("\nRunning minimal test to verify it works...")
    
    test_path = "data/nethack_kg_test_easy_proper.tsv"
    if os.path.exists(test_path):
        mrr = ultra.evaluate_minimal(test_path, "EASY Test", max_queries=5)
        
        print("\n" + "="*50)
        print("RESULTS:")
        print(f"  ULTRA from scratch: {mrr:.3f} MRR")
        print(f"  Heuristics baseline: 0.555 MRR")
        print("\nNote: Random initialization gives poor results.")
        print("Would need proper training to beat heuristics!")
    else:
        print("Test file not found!")

if __name__ == "__main__":
    main()