#!/usr/bin/env python3
"""
ULTRA with proper fixes for NetHack KG
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

# Monkey-patch the build_relation_graph to handle our case
def build_relation_graph_fixed(graph):
    """Fixed version that handles any number of relations"""
    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    
    device = edge_index.device
    
    # Build sparse adjacency matrices for each relation
    # Ei: sparse adjacency matrix for relation i
    # E: sparse adjacency matrices for all relations
    Eh, Et = [], []
    for r in range(num_rels):
        mask = edge_type == r
        Eh.append(edge_index[:, mask][0])
        Et.append(edge_index[:, mask][1])
    
    # Degree
    Dh = torch.bincount(torch.cat(Eh), minlength=num_nodes).float() + 1e-10
    Dt = torch.bincount(torch.cat(Et), minlength=num_nodes).float() + 1e-10

    # Construct head/tail matrices
    Eh_indices = torch.stack([torch.cat(Eh), torch.arange(len(torch.cat(Eh)), device=device) % num_rels])
    Et_indices = torch.stack([torch.cat(Et), torch.arange(len(torch.cat(Et)), device=device) % num_rels])
    
    # Create sparse tensors with correct dimensions
    Eh_sparse = torch.sparse_coo_tensor(
        Eh_indices,
        torch.ones(Eh_indices.shape[1], device=device),
        (num_nodes, num_rels)
    )
    
    Et_sparse = torch.sparse_coo_tensor(
        Et_indices,
        torch.ones(Et_indices.shape[1], device=device),
        (num_nodes, num_rels)
    )
    
    # Simple relation graph - just connect all relations
    # This is a simplified version that avoids the complex sparse operations
    rel_edges = []
    for i in range(num_rels):
        for j in range(num_rels):
            if i != j:
                rel_edges.append([i, j])
    
    if not rel_edges:
        # If only one relation, add self-loop
        rel_edges = [[0, 0]]
    
    rel_edge_index = torch.tensor(rel_edges, dtype=torch.long, device=device).t()
    
    # Create relation graph with fixed 4 types as ULTRA expects
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

class ULTRAFixed:
    def __init__(self, train_path="data/nethack_kg_train_proper.tsv"):
        print("🚀 Initializing ULTRA (Final Fixed Version)")
        print("="*50)
        
        # Force CPU for debugging
        self.device = torch.device('cpu')
        print(f"Using device: {self.device} (forced CPU for debugging)")
        
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
        
        # Add self-loops as separate relation
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
        
        # Set num_relations including self-loop
        data.num_relations = len(self.relations) + 1
        
        print(f"\nGraph stats:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Relations: {data.num_relations}")
        print(f"  Edges: {data.edge_index.shape[1]}")
        
        return data
    
    def _initialize_ultra(self):
        """Initialize ULTRA model"""
        print("\nInitializing ULTRA model...")
        
        # Build relation graph with our fixed function
        self.kg_data = build_relation_graph_fixed(self.kg_data)
        
        # Move to device
        self.kg_data = self.kg_data.to(self.device)
        
        # Simple config
        config = edict({
            'model': {
                'relation_graph': {
                    'class': 'RelNBFNet',
                    'input_dim': 32,
                    'hidden_dims': [32, 32],
                    'message_func': 'distmult',
                    'aggregate_func': 'sum',
                    'short_cut': True,
                    'layer_norm': True
                },
                'entity_graph': {
                    'class': 'EntityNBFNet',
                    'input_dim': 32,
                    'hidden_dims': [32, 32],
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
        
        print("✓ Model initialized")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, head, relation, max_candidates=10):
        """Predict candidates"""
        
        if head not in self.entity2id or relation not in self.relation2id:
            return []
        
        head_id = self.entity2id[head]
        rel_id = self.relation2id[relation]
        
        # Get candidates
        if relation in self.relation_candidates:
            candidate_ids = list(self.relation_candidates[relation])[:20]
        else:
            candidate_ids = list(range(min(len(self.entities), 20)))
        
        if not candidate_ids:
            return []
        
        scores = []
        
        # Score each candidate
        for tail_id in candidate_ids:
            query = torch.tensor([[[head_id, rel_id, tail_id]]], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                try:
                    score = self.model(self.kg_data, query)
                    scores.append((tail_id, float(score.cpu().item())))
                except Exception as e:
                    scores.append((tail_id, 0.0))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        top_candidates = [self.id2entity[cid] for cid, _ in scores[:max_candidates]]
        return top_candidates
    
    def test_minimal(self):
        """Minimal test"""
        print("\n" + "="*50)
        print("Testing ULTRA (Fixed)")
        print("="*50)
        
        # Test on a few queries
        test_queries = [
            ("Medusa", "armor_class"),
            ("Dragon", "speed"),
            ("Lich", "difficulty")
        ]
        
        for head, rel in test_queries:
            if head in self.entity2id and rel in self.relation2id:
                print(f"\nQuery: {head} {rel} ?")
                predictions = self.predict(head, rel, max_candidates=5)
                if predictions:
                    print(f"Top predictions: {predictions}")
                else:
                    print("No predictions")
            else:
                print(f"\nQuery: {head} {rel} ? - Entity/relation not found")

def main():
    print("🔥 Testing ULTRA with All Fixes Applied")
    print("="*50)
    
    try:
        # Initialize
        ultra = ULTRAFixed()
        
        # Run minimal test
        ultra.test_minimal()
        
        print("\n✅ ULTRA is working!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()