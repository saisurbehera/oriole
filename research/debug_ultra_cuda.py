#!/usr/bin/env python3
"""
Debug and fix the CUDA index error in ULTRA
"""

import os
import sys
import pandas as pd
import torch
from torch_geometric.data import Data
from easydict import EasyDict as edict

# Add ULTRA path
sys.path.append('ultra')
from ultra.models import Ultra
from ultra.tasks import build_relation_graph

def debug_ultra():
    print("🔍 Debugging ULTRA CUDA Error")
    print("="*50)
    
    # Load minimal data
    train_kg = pd.read_csv("data/nethack_kg_train_proper.tsv", sep='\t')
    print(f"Training triples: {len(train_kg)}")
    
    # Get unique relations
    relations = sorted(train_kg['type'].unique())
    print(f"\nUnique relations in our KG: {len(relations)}")
    print(f"First few relations: {relations[:5]}")
    
    # Build minimal test
    entities = ['Medusa', 'Dragon', 'Lich', 'Castle', 'Dungeon']
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    
    print(f"\nTest entities: {len(entities)}")
    print(f"Test relations: {len(relations)}")
    
    # Create minimal graph
    edge_list = [[0, 1], [1, 2], [2, 3], [3, 4]]  # Simple chain
    edge_type = [0, 1, 2, 3]  # Different relation types
    
    # Add self-loops
    self_loop_rel_id = len(relations)
    for i in range(len(entities)):
        edge_list.append([i, i])
        edge_type.append(self_loop_rel_id)
    
    print(f"\nSelf-loop relation ID: {self_loop_rel_id}")
    print(f"Max relation ID in edges: {max(edge_type)}")
    
    # Build PyG data
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
    
    data = Data(
        edge_index=edge_index,
        edge_type=edge_type_tensor,
        num_nodes=len(entities)
    )
    
    # IMPORTANT: Set num_relations correctly
    data.num_relations = len(relations) + 1  # +1 for self-loop
    
    print(f"\nGraph stats:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Edge types: {data.edge_type.unique().tolist()}")
    print(f"  Num relations set to: {data.num_relations}")
    
    # Build relation graph
    print("\nBuilding relation graph...")
    try:
        kg_data = build_relation_graph(data)
        print("✓ Relation graph built successfully!")
        
        print(f"\nRelation graph stats:")
        print(f"  Nodes: {kg_data.relation_graph.num_nodes}")
        print(f"  Expected: {data.num_relations}")
        
        if hasattr(kg_data.relation_graph, 'edge_index'):
            print(f"  Edges: {kg_data.relation_graph.edge_index.shape}")
    except Exception as e:
        print(f"❌ Error building relation graph: {e}")
        return
    
    # Initialize ULTRA
    print("\nInitializing ULTRA model...")
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
    
    model = Ultra(
        rel_model_cfg=config.model.relation_graph,
        entity_model_cfg=config.model.entity_graph
    )
    
    # Don't load checkpoint for debugging
    print("✓ Model initialized (no checkpoint)")
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    kg_data = kg_data.to(device)
    model.eval()
    
    # Test prediction
    print(f"\nTesting prediction on {device}...")
    
    # Create test query: (Medusa, armor_class, ?)
    test_batch = torch.tensor([[[0, 0, 1]]], dtype=torch.long).to(device)  # Shape: (1, 1, 3)
    
    try:
        with torch.no_grad():
            score = model(kg_data, test_batch)
            print(f"✓ Prediction successful! Score shape: {score.shape}")
            print(f"  Score: {score.item():.3f}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        
        # Debug the error
        print("\nDebugging info:")
        print(f"  kg_data.num_relations: {kg_data.num_relations}")
        print(f"  relation_graph.num_nodes: {kg_data.relation_graph.num_nodes}")
        
        # Check if they match
        if kg_data.num_relations != kg_data.relation_graph.num_nodes:
            print(f"  ⚠️  Mismatch! num_relations != relation_graph.num_nodes")
            print(f"  This is likely the cause of the index error")

if __name__ == "__main__":
    debug_ultra()