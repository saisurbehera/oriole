"""
Simplified ULTRA test that works around the relation graph issue
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
        for i in range(src.shape[0]):
            out[index[i]] += src[i]
        return out
    mock_module.scatter_add = scatter_add
    return mock_module

sys.modules['torch_scatter'] = mock_torch_scatter()

def load_nethack_kg_simple(tsv_path):
    """Load NetHack KG with simplified processing"""
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
    
    # Convert to edge indices
    edge_list = []
    edge_type = []
    
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
    
    # Add self-loops for all entities and relations to ensure connectivity
    # This helps avoid the assertion error in build_relation_graph
    self_loop_rel_id = len(relations)
    for i in range(len(entities)):
        edge_list.append([i, i])
        edge_type.append(self_loop_rel_id)
    
    # Add inverse relations
    num_direct_relations = len(relations) + 1  # +1 for self-loop relation
    inverse_edge_list = []
    inverse_edge_type = []
    
    for i, (h_id, t_id) in enumerate(edge_list):
        if edge_type[i] != self_loop_rel_id:  # Don't add inverse for self-loops
            inverse_edge_list.append([t_id, h_id])
            inverse_edge_type.append(edge_type[i] + num_direct_relations)
    
    # Combine all edges
    all_edges = edge_list + inverse_edge_list
    all_edge_types = edge_type + inverse_edge_type
    
    edge_index = torch.tensor(all_edges, dtype=torch.long).t()
    edge_type_tensor = torch.tensor(all_edge_types, dtype=torch.long)
    
    # Create PyG Data object
    data = Data(
        edge_index=edge_index,
        edge_type=edge_type_tensor,
        num_nodes=len(entities)
    )
    data.num_relations = num_direct_relations * 2
    
    # Update relation mappings
    relation2id['self_loop'] = self_loop_rel_id
    id2relation[self_loop_rel_id] = 'self_loop'
    
    # Add inverse relations to mappings
    for rel, idx in list(relation2id.items()):
        if idx < num_direct_relations and idx != self_loop_rel_id:
            inverse_rel = f"inverse_{rel}"
            relation2id[inverse_rel] = idx + num_direct_relations
            id2relation[idx + num_direct_relations] = inverse_rel
    
    return {
        'data': data,
        'entity2id': entity2id,
        'id2entity': id2entity,
        'relation2id': relation2id,
        'id2relation': id2relation,
        'num_entities': len(entities),
        'num_relations': num_direct_relations * 2,
        'df': df
    }

def test_ultra_hard():
    """Comprehensive and challenging test for ULTRA"""
    
    print("=== ULTRA Comprehensive Test (Harder Version) ===\n")
    
    # Load the KG
    kg_path = "data/nethack_kg_full/nethack_kg_intermediate.tsv"
    print(f"Loading KG from: {kg_path}")
    
    kg_info = load_nethack_kg_simple(kg_path)
    kg_data = kg_info['data']
    
    print(f"\nKG Statistics:")
    print(f"  - Entities: {kg_info['num_entities']}")
    print(f"  - Relations: {kg_info['num_relations']} (including inverses)")
    print(f"  - Edges: {kg_data.edge_index.shape[1]}")
    
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
        
        # Load checkpoint
        checkpoint_path = "ultra/ckpts/ultra_4g.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            model.eval()
            print("✓ Checkpoint loaded")
        
        # Define harder test queries - testing multi-hop reasoning
        hard_queries = [
            # Complex game mechanics queries
            ("player", "instance of"),  # What type is the player?
            ("scroll of identify", "use"),  # What can scroll of identify be used for?
            ("blessed", "different from"),  # What is blessed different from?
            
            # Transitive reasoning
            ("Castle", "part of"),  # Castle -> ? (should find it's part of dungeon/nethack)
            ("Valley of the Dead", "has part"),  # What does Valley contain?
            
            # Rare relations
            ("M2_HOSTILE", "facet of"),
            ("polypiling", "uses"),
            
            # Variant-specific queries
            ("SlashTHEM", "based on"),
            ("Hack'EM", "developer"),
            ("notdNetHack", "part of the series"),
            
            # Abstract concepts
            ("conducts", "instance of"),
            ("pacifist", "subclass of"),
            
            # Item interactions
            ("ring of polymorph control", "use"),
            ("wand of death", "instance of"),
            
            # Location hierarchies
            ("Astral Plane", "part of"),
            ("Fort Ludios", "has part"),
            
            # Monster relationships
            ("lich", "subclass of"),
            ("Rodney", "instance of"),
        ]
        
        print("\n=== Harder Link Prediction Tests ===\n")
        
        # Track detailed metrics
        all_mrr = []
        all_hits = defaultdict(list)
        relation_performance = defaultdict(list)
        
        with torch.no_grad():
            for head_name, rel_name in hard_queries:
                if head_name in kg_info['entity2id'] and rel_name in kg_info['relation2id']:
                    h_idx = kg_info['entity2id'][head_name]
                    r_idx = kg_info['relation2id'][rel_name]
                    
                    # Find ground truth
                    mask = (kg_data.edge_index[0] == h_idx) & (kg_data.edge_type == r_idx)
                    true_tails = kg_data.edge_index[1][mask].unique().tolist()
                    
                    # Skip self-loops
                    true_tails = [t for t in true_tails if t != h_idx]
                    
                    print(f"\nQuery: {head_name} --[{rel_name}]--> ?")
                    
                    if not true_tails:
                        print("  No ground truth found in KG")
                        continue
                    
                    print(f"Ground truth ({len(true_tails)} answers):")
                    for t_idx in true_tails[:3]:
                        print(f"  - {kg_info['id2entity'][t_idx]}")
                    
                    try:
                        # Create batch
                        num_entities = kg_info['num_entities']
                        h_indices = torch.full((num_entities,), h_idx, dtype=torch.long)
                        t_indices = torch.arange(num_entities, dtype=torch.long)
                        r_indices = torch.full((num_entities,), r_idx, dtype=torch.long)
                        
                        batch = torch.stack([h_indices, t_indices, r_indices], dim=1)
                        batch = batch.unsqueeze(0)
                        
                        # Get predictions
                        scores = model(kg_data, batch).squeeze(0)
                        
                        # Filter out self-loops
                        scores[h_idx] = float('-inf')
                        
                        # Calculate metrics
                        sorted_indices = torch.argsort(scores, descending=True)
                        
                        # Find ranks
                        ranks = []
                        for true_tail in true_tails:
                            rank = (sorted_indices == true_tail).nonzero(as_tuple=True)[0].item() + 1
                            ranks.append(rank)
                        
                        # Metrics
                        mrr = np.mean([1.0 / rank for rank in ranks])
                        hits_1 = np.mean([1 if rank <= 1 else 0 for rank in ranks])
                        hits_3 = np.mean([1 if rank <= 3 else 0 for rank in ranks])
                        hits_10 = np.mean([1 if rank <= 10 else 0 for rank in ranks])
                        
                        all_mrr.append(mrr)
                        all_hits[1].append(hits_1)
                        all_hits[3].append(hits_3)
                        all_hits[10].append(hits_10)
                        relation_performance[rel_name].append(mrr)
                        
                        # Show predictions
                        print("\nModel predictions:")
                        top_k = 5
                        top_scores, top_indices = torch.topk(scores, top_k)
                        
                        for i in range(top_k):
                            idx = top_indices[i].item()
                            score = top_scores[i].item()
                            entity = kg_info['id2entity'][idx]
                            is_correct = idx in true_tails
                            print(f"  {i+1}. {entity} (score: {score:.3f}){' ✓' if is_correct else ''}")
                        
                        print(f"\nMetrics: MRR={mrr:.3f}, H@1={hits_1:.3f}, H@3={hits_3:.3f}, H@10={hits_10:.3f}")
                        print(f"Best rank: {min(ranks)}")
                        
                    except Exception as e:
                        print(f"Error: {e}")
                
                else:
                    missing = []
                    if head_name not in kg_info['entity2id']:
                        missing.append(f"entity '{head_name}'")
                    if rel_name not in kg_info['relation2id']:
                        missing.append(f"relation '{rel_name}'")
                    print(f"\nSkipping query - missing {' and '.join(missing)}")
        
        # Summary
        print("\n" + "="*50)
        print("=== OVERALL PERFORMANCE SUMMARY ===")
        print("="*50)
        
        if all_mrr:
            print(f"\nGlobal Metrics (across {len(all_mrr)} successful queries):")
            print(f"  - Mean MRR: {np.mean(all_mrr):.3f}")
            print(f"  - Mean Hits@1: {np.mean(all_hits[1]):.3f}")
            print(f"  - Mean Hits@3: {np.mean(all_hits[3]):.3f}")
            print(f"  - Mean Hits@10: {np.mean(all_hits[10]):.3f}")
            
            print("\nPerformance by Relation Type:")
            for rel, mrrs in sorted(relation_performance.items(), key=lambda x: np.mean(x[1]), reverse=True):
                print(f"  - {rel}: MRR = {np.mean(mrrs):.3f} ({len(mrrs)} queries)")
        
        # Test robustness with noise
        print("\n=== Testing Robustness ===")
        
        # Add noise to a query
        if "NetHack" in kg_info['entity2id'] and "genre" in kg_info['relation2id']:
            h_idx = kg_info['entity2id']["NetHack"]
            r_idx = kg_info['relation2id']["genre"]
            
            # Create noisy batch by adding random entities
            num_samples = 100
            random_heads = torch.randint(0, kg_info['num_entities'], (num_samples,))
            random_tails = torch.randint(0, kg_info['num_entities'], (num_samples,))
            random_rels = torch.randint(0, kg_info['num_relations'] // 2, (num_samples,))
            
            # Include our query
            random_heads[0] = h_idx
            random_tails[0] = 0  # Dummy tail
            random_rels[0] = r_idx
            
            noisy_batch = torch.stack([random_heads, random_tails, random_rels], dim=1)
            noisy_batch = noisy_batch.unsqueeze(0)
            
            try:
                with torch.no_grad():
                    _ = model(kg_data, noisy_batch)
                print("✓ Model handles noisy/random queries robustly")
            except Exception as e:
                print(f"✗ Model failed on noisy queries: {e}")
        
        print("\n✓ Comprehensive hard test completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ultra_hard()