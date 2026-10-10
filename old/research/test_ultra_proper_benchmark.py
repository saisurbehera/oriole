#!/usr/bin/env python3
"""
Test ULTRA on the PROPER benchmark
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

class ULTRAProperBenchmark:
    def __init__(self, train_path="data/nethack_kg_train_proper.tsv"):
        print("🚀 Initializing ULTRA for proper benchmark")
        print("="*50)
        
        # Load training data
        print("Loading training data...")
        self.train_kg = pd.read_csv(train_path, sep='\t')
        print(f"Training set: {len(self.train_kg)} triples")
        
        # Build entity and relation mappings
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
        """Build PyG data from training set"""
        edge_list = []
        edge_type = []
        
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
        self.kg_data = build_relation_graph(self.kg_data)
        
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
    
    def predict_ultra(self, head, relation, max_candidates=20):
        """ULTRA prediction"""
        
        if head not in self.entity2id or relation not in self.relation2id:
            return []
        
        head_id = self.entity2id[head]
        rel_id = self.relation2id[relation]
        
        # Score all possible tail entities
        candidate_ids = list(range(len(self.entities)))
        
        # Batch predictions for efficiency
        batch_size = 100
        all_scores = []
        
        for i in range(0, len(candidate_ids), batch_size):
            batch_candidates = candidate_ids[i:i+batch_size]
            
            # Create batch
            batch_triples = []
            for tail_id in batch_candidates:
                batch_triples.append([head_id, rel_id, tail_id])
            
            batch = torch.tensor([batch_triples], dtype=torch.long)  # Shape: (1, batch_size, 3)
            
            with torch.no_grad():
                try:
                    scores = self.model(self.kg_data, batch)
                    all_scores.extend(scores.cpu().numpy().flatten().tolist())
                except:
                    # Fallback to individual scoring
                    for tail_id in batch_candidates:
                        try:
                            single_batch = torch.tensor([[[head_id, rel_id, tail_id]]], dtype=torch.long)
                            score = self.model(self.kg_data, single_batch)
                            all_scores.append(float(score.cpu()))
                        except:
                            all_scores.append(-float('inf'))
        
        # Sort by score
        scored_candidates = [(candidate_ids[i], all_scores[i]) for i in range(len(all_scores)) 
                           if not np.isnan(all_scores[i]) and not np.isinf(all_scores[i])]
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
        
        # Verify all entities are in vocabulary
        missing = 0
        for _, row in test_kg.iterrows():
            if row['head'] not in self.entity2id or row['tail'] not in self.entity2id:
                missing += 1
        
        if missing > 0:
            print(f"WARNING: {missing} triples have entities not in training vocabulary")
        
        # Sample for evaluation
        sample_size = min(100, len(test_kg))
        test_sample = test_kg.sample(n=sample_size, random_state=42)
        
        mrr_scores = []
        hits_at_1 = []
        hits_at_3 = []
        hits_at_10 = []
        
        for idx, row in test_sample.iterrows():
            if idx % 20 == 0:
                print(f"Progress: {idx}/{sample_size}")
            
            head, rel, tail = row['head'], row['type'], row['tail']
            
            # Skip if not in vocabulary
            if head not in self.entity2id or tail not in self.entity2id or rel not in self.relation2id:
                continue
            
            predictions = self.predict_ultra(head, rel)
            
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
            print(f"\nResults (on {len(mrr_scores)} queries):")
            print(f"  MRR: {np.mean(mrr_scores):.3f}")
            print(f"  Hits@1: {np.mean(hits_at_1):.3f}")
            print(f"  Hits@3: {np.mean(hits_at_3):.3f}")
            print(f"  Hits@10: {np.mean(hits_at_10):.3f}")
        
        return np.mean(mrr_scores) if mrr_scores else 0.0

def main():
    print("🔥 Testing ULTRA on PROPER Benchmark")
    print("="*50)
    
    # Initialize ULTRA
    ultra = ULTRAProperBenchmark()
    
    # Test on different difficulty levels
    results = {}
    for difficulty in ['easy', 'medium', 'hard']:
        test_path = f"data/nethack_kg_test_{difficulty}_proper.tsv"
        if os.path.exists(test_path):
            mrr = ultra.evaluate_on_test_set(test_path, f"{difficulty.upper()} Test Set")
            results[difficulty] = mrr
    
    print("\n" + "="*50)
    print("ULTRA vs HEURISTICS - PROPER BENCHMARK:")
    print("\nULTRA Performance:")
    for diff, mrr in results.items():
        print(f"  {diff}: {mrr:.3f}")
    print(f"  Average: {np.mean(list(results.values())):.3f}")
    
    print("\nHeuristics Performance (for comparison):")
    print("  easy: 0.555")
    print("  medium: 0.404")
    print("  hard: 0.115")
    print("  Average: 0.358")
    
    print("\n" + "="*50)
    ultra_avg = np.mean(list(results.values()))
    if ultra_avg > 0.358:
        print(f"✅ ULTRA WINS! {ultra_avg:.3f} vs 0.358 (heuristics)")
    else:
        print(f"❌ Heuristics better: 0.358 vs {ultra_avg:.3f} (ULTRA)")

if __name__ == "__main__":
    main()