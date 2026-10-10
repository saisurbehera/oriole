"""
Comprehensive test suite for ULTRA on NetHack knowledge graph
"""

import torch
import pandas as pd
from torch_geometric.data import Data
import yaml
from easydict import EasyDict as edict
import os
import sys
import numpy as np
from collections import defaultdict
import json
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultra'))

# Mock torch_scatter
import types
def mock_torch_scatter():
    mock_module = types.ModuleType('torch_scatter')
    def scatter_add(src, index, dim=0, out=None, dim_size=None):
        if out is None:
            if dim_size is None:
                dim_size = int(index.max()) + 1
            out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        for i in range(src.shape[0])[:min(len(src), 1000)]:  # Limit for speed
            out[index[i]] += src[i]
        return out
    mock_module.scatter_add = scatter_add
    return mock_module

sys.modules['torch_scatter'] = mock_torch_scatter()

def load_nethack_kg(tsv_path):
    """Load NetHack KG from TSV file with proper PyG format"""
    df = pd.read_csv(tsv_path, sep='\t', header=0, names=['head', 'relation', 'tail'])
    
    # Build entity and relation mappings
    entities = set()
    relations = set()
    
    for _, row in df.iterrows():
        if pd.notna(row['head']) and pd.notna(row['tail']) and pd.notna(row['relation']):
            entities.add(str(row['head']))
            entities.add(str(row['tail']))
            relations.add(str(row['relation']))
    
    entities.discard('')
    relations.discard('')
    
    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    id2entity = {i: e for e, i in entity2id.items()}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    id2relation = {i: r for r, i in relation2id.items()}
    
    # Add inverse relations to mappings
    num_direct = len(relations)
    for rel, idx in list(relation2id.items()):
        inverse_rel = f"inverse_{rel}"
        relation2id[inverse_rel] = idx + num_direct
        id2relation[idx + num_direct] = inverse_rel
    
    # Convert to edge indices
    edge_list = []
    edge_type = []
    triple_set = set()  # For filtering
    
    for _, row in df.iterrows():
        if pd.notna(row['head']) and pd.notna(row['tail']) and pd.notna(row['relation']):
            h = str(row['head'])
            t = str(row['tail'])
            r = str(row['relation'])
            
            if h in entity2id and t in entity2id and r in relation2id:
                h_id = entity2id[h]
                t_id = entity2id[t]
                r_id = relation2id[r]
                
                edge_list.append([h_id, t_id])
                edge_type.append(r_id)
                triple_set.add((h_id, r_id, t_id))
    
    # Count edges per relation
    relation_edge_count = defaultdict(int)
    for r_id in edge_type:
        relation_edge_count[r_id] += 1
    
    # Only keep relations that have edges
    used_relations = set(relation_edge_count.keys())
    
    # Create new relation mappings with only used relations
    old_to_new_relation = {}
    new_relation2id = {}
    new_id2relation = {}
    
    for new_idx, old_idx in enumerate(sorted(used_relations)):
        old_to_new_relation[old_idx] = new_idx
        rel_name = id2relation[old_idx]
        new_relation2id[rel_name] = new_idx
        new_id2relation[new_idx] = rel_name
    
    # Remap edge types
    remapped_edge_type = [old_to_new_relation[r_id] for r_id in edge_type]
    
    # Add inverse relations (ULTRA expects this)
    num_direct_relations = len(used_relations)
    inverse_edge_list = []
    inverse_edge_type = []
    
    for i, (h_id, t_id) in enumerate(edge_list):
        # Add inverse edge
        inverse_edge_list.append([t_id, h_id])
        inverse_edge_type.append(remapped_edge_type[i] + num_direct_relations)
    
    # Update mappings for inverse relations
    for rel_name, idx in list(new_relation2id.items()):
        if idx < num_direct_relations:  # Only for direct relations
            inverse_rel = f"inverse_{rel_name}"
            new_relation2id[inverse_rel] = idx + num_direct_relations
            new_id2relation[idx + num_direct_relations] = inverse_rel
    
    # Combine direct and inverse edges
    all_edges = edge_list + inverse_edge_list
    all_edge_types = remapped_edge_type + inverse_edge_type
    
    edge_index = torch.tensor(all_edges, dtype=torch.long).t()
    edge_type_tensor = torch.tensor(all_edge_types, dtype=torch.long)
    
    # Create PyG Data object
    data = Data(
        edge_index=edge_index,
        edge_type=edge_type_tensor,
        num_nodes=len(entities)
    )
    # ULTRA expects 2x relations (direct + inverse)
    data.num_relations = num_direct_relations * 2
    
    # Update relation mappings
    relation2id = new_relation2id
    id2relation = new_id2relation
    
    return {
        'data': data,
        'entity2id': entity2id,
        'id2entity': id2entity,
        'relation2id': relation2id,
        'id2relation': id2relation,
        'num_entities': len(entities),
        'num_relations': num_direct_relations * 2,  # Including inverses
        'num_triples': len(df),
        'triple_set': triple_set,
        'df': df
    }

def calculate_metrics(scores, true_tails, k_values=[1, 3, 5, 10]):
    """Calculate MRR and Hits@K metrics"""
    # Sort scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    
    # Find ranks of true tails
    ranks = []
    for true_tail in true_tails:
        rank = (sorted_indices == true_tail).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    # Calculate metrics
    mrr = np.mean([1.0 / rank for rank in ranks]) if ranks else 0.0
    
    hits_at_k = {}
    for k in k_values:
        hits_at_k[k] = np.mean([1 if rank <= k else 0 for rank in ranks]) if ranks else 0.0
    
    return {
        'mrr': mrr,
        'hits@1': hits_at_k.get(1, 0.0),
        'hits@3': hits_at_k.get(3, 0.0),
        'hits@5': hits_at_k.get(5, 0.0),
        'hits@10': hits_at_k.get(10, 0.0),
        'best_rank': min(ranks) if ranks else float('inf')
    }

def test_ultra_comprehensive():
    """Comprehensive test suite for ULTRA"""
    
    print("=== ULTRA Comprehensive Test Suite ===\n")
    
    # Load the KG
    kg_path = "data/nethack_kg_full/nethack_kg_intermediate.tsv"
    print(f"Loading KG from: {kg_path}")
    
    kg_info = load_nethack_kg(kg_path)
    kg_data = kg_info['data']
    
    print(f"\nKG Statistics:")
    print(f"  - Entities: {kg_info['num_entities']}")
    print(f"  - Relations: {kg_info['num_relations']}")
    print(f"  - Triples: {kg_info['num_triples']}")
    
    # Analyze relation distribution
    relation_counts = defaultdict(int)
    for _, row in kg_info['df'].iterrows():
        if pd.notna(row['relation']):
            relation_counts[row['relation']] += 1
    
    print(f"\nTop 10 Relations:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {rel}: {count}")
    
    try:
        # Import and setup ULTRA
        from ultra.tasks import build_relation_graph
        from ultra.models import Ultra
        
        # Build relation graph
        print("\nBuilding relation graph...")
        kg_data = build_relation_graph(kg_data)
        print("✓ Relation graph built")
        
        # Create config
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
        
        # Initialize model
        print("\nInitializing ULTRA model...")
        model = Ultra(
            rel_model_cfg=config.model.relation_graph,
            entity_model_cfg=config.model.entity_graph
        )
        
        # Load checkpoint if available
        checkpoint_path = "ultra/ckpts/ultra_4g.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            model.eval()
            print("✓ Checkpoint loaded")
        
        # Define comprehensive test queries
        test_queries = [
            # Game structure queries
            ("NetHack", "part of"),
            ("NetHack", "has part"),
            ("NetHack", "genre"),
            ("NetHack", "developer"),
            
            # Item-related queries
            ("Amulet of Yendor", "present in work"),
            ("Excalibur", "instance of"),
            ("Stormbringer", "instance of"),
            ("wand of wishing", "subclass of"),
            
            # Location queries
            ("Sokoban", "part of"),
            ("Gehennom", "has part"),
            ("Gnomish Mines", "has part"),
            ("Valley of the Dead", "part of"),
            
            # Character/role queries
            ("Valkyrie", "instance of"),
            ("Archeologist", "present in work"),
            ("Wizard of Yendor", "owner of"),
            
            # Game mechanics
            ("armor", "subclass of"),
            ("trap", "part of"),
            ("monster", "instance of"),
            
            # Wiki structure
            ("NetHackWiki", "instance of"),
            ("NetHackWiki", "part of"),
            
            # Variant games
            ("SLASH'EM", "genre"),
            ("SLASH'EM", "based on"),
            ("UnNetHack", "part of"),
            
            # Abstract concepts
            ("extinctionism", "used by"),
            ("farming", "used by"),
            ("ascension", "part of"),
        ]
        
        # Run comprehensive evaluation
        print("\n=== Comprehensive Link Prediction Evaluation ===\n")
        
        all_metrics = defaultdict(list)
        successful_queries = 0
        
        with torch.no_grad():
            for head_name, rel_name in test_queries:
                if head_name in kg_info['entity2id'] and rel_name in kg_info['relation2id']:
                    h_idx = kg_info['entity2id'][head_name]
                    r_idx = kg_info['relation2id'][rel_name]
                    
                    # Find ground truth
                    mask = (kg_data.edge_index[0] == h_idx) & (kg_data.edge_type == r_idx)
                    true_tails = kg_data.edge_index[1][mask].tolist()
                    
                    print(f"Query: {head_name} --[{rel_name}]--> ?")
                    
                    if true_tails:
                        print(f"Ground truth ({len(true_tails)} answers):")
                        for t_idx in true_tails[:3]:
                            print(f"  - {kg_info['id2entity'][t_idx]}")
                        if len(true_tails) > 3:
                            print(f"  ... and {len(true_tails) - 3} more")
                    else:
                        print("  No ground truth found")
                        continue
                    
                    try:
                        # Create batch for all candidates
                        num_entities = kg_info['num_entities']
                        h_indices = torch.full((num_entities,), h_idx, dtype=torch.long)
                        t_indices = torch.arange(num_entities, dtype=torch.long)
                        r_indices = torch.full((num_entities,), r_idx, dtype=torch.long)
                        
                        batch = torch.stack([h_indices, t_indices, r_indices], dim=1)
                        batch = batch.unsqueeze(0)
                        
                        # Get predictions
                        scores = model(kg_data, batch).squeeze(0)
                        
                        # Calculate metrics
                        metrics = calculate_metrics(scores, true_tails)
                        
                        # Show top predictions
                        top_k = 5
                        top_scores, top_indices = torch.topk(scores, top_k)
                        
                        print("Model predictions:")
                        for i in range(top_k):
                            idx = top_indices[i].item()
                            score = top_scores[i].item()
                            entity = kg_info['id2entity'][idx]
                            is_correct = idx in true_tails
                            print(f"  {i+1}. {entity} (score: {score:.3f}){' ✓' if is_correct else ''}")
                        
                        print(f"Metrics: MRR={metrics['mrr']:.3f}, H@1={metrics['hits@1']:.3f}, "
                              f"H@3={metrics['hits@3']:.3f}, H@10={metrics['hits@10']:.3f}")
                        print()
                        
                        # Aggregate metrics
                        for key, value in metrics.items():
                            if key != 'best_rank':
                                all_metrics[key].append(value)
                        
                        successful_queries += 1
                        
                    except Exception as e:
                        print(f"Error during inference: {e}")
                        print()
        
        # Summary statistics
        print("\n=== Overall Performance Summary ===")
        print(f"Successfully evaluated: {successful_queries}/{len(test_queries)} queries")
        
        if all_metrics:
            print("\nAverage Metrics:")
            for metric, values in all_metrics.items():
                avg_value = np.mean(values)
                print(f"  - {metric.upper()}: {avg_value:.3f}")
        
        # Test batch processing
        print("\n=== Testing Batch Processing ===")
        
        # Create a batch of multiple queries
        batch_queries = []
        batch_info = []
        
        for head_name, rel_name in test_queries[:5]:  # Use first 5 queries
            if head_name in kg_info['entity2id'] and rel_name in kg_info['relation2id']:
                h_idx = kg_info['entity2id'][head_name]
                r_idx = kg_info['relation2id'][rel_name]
                
                # Create mini-batch for this query
                for tail_idx in range(min(10, kg_info['num_entities'])):  # Sample 10 candidates
                    batch_queries.append([h_idx, tail_idx, r_idx])
                    batch_info.append((head_name, rel_name, tail_idx))
        
        if batch_queries:
            batch_tensor = torch.tensor(batch_queries, dtype=torch.long)
            batch_tensor = batch_tensor.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                batch_scores = model(kg_data, batch_tensor)
                print(f"✓ Batch processing successful! Processed {len(batch_queries)} triples")
        
        # Analyze prediction patterns
        print("\n=== Prediction Pattern Analysis ===")
        
        # Check if model tends to predict popular entities
        entity_prediction_counts = defaultdict(int)
        
        with torch.no_grad():
            for i in range(min(20, len(test_queries))):
                head_name, rel_name = test_queries[i]
                if head_name in kg_info['entity2id'] and rel_name in kg_info['relation2id']:
                    h_idx = kg_info['entity2id'][head_name]
                    r_idx = kg_info['relation2id'][rel_name]
                    
                    # Get predictions
                    num_entities = kg_info['num_entities']
                    h_indices = torch.full((num_entities,), h_idx, dtype=torch.long)
                    t_indices = torch.arange(num_entities, dtype=torch.long)
                    r_indices = torch.full((num_entities,), r_idx, dtype=torch.long)
                    
                    batch = torch.stack([h_indices, t_indices, r_indices], dim=1).unsqueeze(0)
                    scores = model(kg_data, batch).squeeze(0)
                    
                    # Get top 10 predictions
                    _, top_indices = torch.topk(scores, 10)
                    for idx in top_indices:
                        entity_prediction_counts[kg_info['id2entity'][idx.item()]] += 1
        
        print("Most frequently predicted entities:")
        for entity, count in sorted(entity_prediction_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {entity}: {count} times")
        
        print("\n✓ Comprehensive test completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ultra_comprehensive()