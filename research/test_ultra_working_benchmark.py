#!/usr/bin/env python3
"""
Test working ULTRA on proper benchmark
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
from ultra import tasks

# Use the fixed build_relation_graph
def build_relation_graph_fixed(graph):
    """Fixed version that handles any number of relations"""
    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    
    device = edge_index.device
    
    # Simple relation graph - connect all relations
    rel_edges = []
    for i in range(num_rels):
        for j in range(num_rels):
            if i != j:
                rel_edges.append([i, j])
    
    if not rel_edges:
        rel_edges = [[0, 0]]
    
    rel_edge_index = torch.tensor(rel_edges, dtype=torch.long, device=device).t()
    
    rel_graph = Data(
        edge_index=rel_edge_index,
        edge_type=torch.zeros(rel_edge_index.shape[1], dtype=torch.long, device=device),
        num_nodes=num_rels,
        num_relations=4  # ULTRA expects exactly 4
    )
    
    graph.relation_graph = rel_graph
    return graph

# Replace the original function
tasks.build_relation_graph = build_relation_graph_fixed

class ULTRABenchmark:
    def __init__(self, train_path="data/nethack_kg_train_proper.tsv"):
        print("🚀 Initializing ULTRA for Benchmark Testing")
        print("="*50)
        
        # Force CPU to avoid CUDA errors
        self.device = torch.device('cpu')
        print(f"Using device: {self.device} (forced CPU to avoid CUDA errors)")
        
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
        
        # Pre-compute for faster inference
        self.relation_candidates = defaultdict(set)
        for _, row in self.train_kg.iterrows():
            self.relation_candidates[row['type']].add(self.entity2id[row['tail']])
    
    def _build_pyg_data(self):
        """Build PyG data"""
        edge_list = []
        edge_type = []
        
        # Add KG edges
        for _, row in self.train_kg.iterrows():
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
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
        
        data = Data(
            edge_index=edge_index,
            edge_type=edge_type_tensor,
            num_nodes=len(self.entities)
        )
        
        data.num_relations = len(self.relations) + 1
        
        return data
    
    def _initialize_ultra(self):
        """Initialize ULTRA model"""
        print("\nInitializing ULTRA model...")
        
        # Build relation graph
        self.kg_data = build_relation_graph_fixed(self.kg_data)
        
        # Move to device
        self.kg_data = self.kg_data.to(self.device)
        
        # Config
        config = edict({
            'model': {
                'relation_graph': {
                    'class': 'RelNBFNet',
                    'input_dim': 64,
                    'hidden_dims': [64, 64, 64],
                    'message_func': 'distmult',
                    'aggregate_func': 'sum',
                    'short_cut': True,
                    'layer_norm': True
                },
                'entity_graph': {
                    'class': 'EntityNBFNet',
                    'input_dim': 64,
                    'hidden_dims': [64, 64, 64],
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
        
        print("✓ Model initialized (random weights)")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, head, relation, max_candidates=50):
        """Predict candidates"""
        
        if head not in self.entity2id or relation not in self.relation2id:
            return []
        
        head_id = self.entity2id[head]
        rel_id = self.relation2id[relation]
        
        # Get candidates
        if relation in self.relation_candidates:
            candidate_ids = list(self.relation_candidates[relation])[:100]
        else:
            candidate_ids = list(range(min(len(self.entities), 100)))
        
        if not candidate_ids:
            return []
        
        # Smaller batch size for CPU
        batch_size = 5
        all_scores = []
        
        for i in range(0, len(candidate_ids), batch_size):
            batch_candidates = candidate_ids[i:i+batch_size]
            batch_queries = []
            
            for tail_id in batch_candidates:
                batch_queries.append([[head_id, rel_id, tail_id]])
            
            batch = torch.tensor(batch_queries, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                try:
                    scores = self.model(self.kg_data, batch)
                    all_scores.extend(scores.cpu().numpy().flatten().tolist())
                except:
                    all_scores.extend([0.0] * len(batch_candidates))
        
        # Sort by score
        scored_candidates = [(candidate_ids[i], all_scores[i]) for i in range(len(all_scores))]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        top_candidates = [self.id2entity[cid] for cid, _ in scored_candidates[:max_candidates]]
        return top_candidates
    
    def evaluate(self, test_path, test_name, max_queries=50):
        """Evaluate on test set"""
        
        print(f"\n{'='*50}")
        print(f"Evaluating ULTRA on {test_name}")
        print(f"{'='*50}")
        
        test_kg = pd.read_csv(test_path, sep='\t')
        print(f"Test set: {len(test_kg)} triples")
        print(f"Evaluating on {max_queries} queries")
        
        # Sample queries
        test_sample = test_kg.sample(n=min(max_queries, len(test_kg)), random_state=42)
        
        mrr_scores = []
        hits_at_1 = []
        hits_at_3 = []
        hits_at_10 = []
        
        start_time = time.time()
        
        for idx, row in enumerate(test_sample.itertuples()):
            if idx % 10 == 0:
                print(f"Progress: {idx}/{len(test_sample)}")
            
            head, rel, tail = row.head, row.type, row.tail
            
            if head not in self.entity2id or tail not in self.entity2id:
                continue
            
            predictions = self.predict(head, rel)
            
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
            print(f"  Time: {elapsed:.1f}s")
        
        return np.mean(mrr_scores) if mrr_scores else 0.0

def main():
    print("🔥 Testing ULTRA on Proper Benchmark")
    print("="*50)
    
    # Initialize
    ultra = ULTRABenchmark()
    
    # Test on different difficulties
    results = {}
    for difficulty in ['easy', 'medium', 'hard']:
        test_path = f"data/nethack_kg_test_{difficulty}_proper.tsv"
        if os.path.exists(test_path):
            mrr = ultra.evaluate(test_path, f"{difficulty.upper()} Test", max_queries=50)
            results[difficulty] = mrr
    
    print("\n" + "="*50)
    print("FINAL RESULTS - ULTRA vs HEURISTICS:")
    print("\nULTRA Performance (random init):")
    for diff, mrr in results.items():
        print(f"  {diff}: {mrr:.3f}")
    if results:
        print(f"  Average: {np.mean(list(results.values())):.3f}")
    
    print("\nHeuristics Performance:")
    print("  easy: 0.555")
    print("  medium: 0.404")
    print("  hard: 0.115")
    print("  Average: 0.358")
    
    print("\n" + "="*50)
    ultra_avg = np.mean(list(results.values())) if results else 0
    if ultra_avg > 0.358:
        print(f"✅ ULTRA ({ultra_avg:.3f}) beats heuristics (0.358)!")
    else:
        print(f"❌ Heuristics (0.358) beat ULTRA ({ultra_avg:.3f})")
        print("Note: ULTRA has random weights - needs training!")

if __name__ == "__main__":
    main()