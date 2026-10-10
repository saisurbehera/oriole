"""
Test ULTRA with properly configured PyG Data object
"""

import torch
import pandas as pd
from torch_geometric.data import Data, HeteroData
import yaml
from easydict import EasyDict as edict
import os
import sys
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

def load_nethack_kg(tsv_path):
    """Load NetHack KG from TSV file with proper PyG format"""
    # Read TSV
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
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    
    # Create PyG Data object with all required attributes
    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=len(entities)
    )
    
    # Add num_relations as an attribute
    data.num_relations = len(relations)
    
    return {
        'data': data,
        'entity2id': entity2id,
        'id2entity': id2entity,
        'relation2id': relation2id,
        'id2relation': id2relation,
        'num_entities': len(entities),
        'num_relations': len(relations),
        'num_triples': len(df)
    }

def test_ultra_with_fixed_data():
    """Test ULTRA with properly formatted data"""
    
    print("=== Testing ULTRA with Fixed Data Format ===\n")
    
    # Load the KG
    kg_path = "data/nethack_kg_full/nethack_kg_intermediate.tsv"
    print(f"Loading KG from: {kg_path}")
    
    kg_info = load_nethack_kg(kg_path)
    kg_data = kg_info['data']
    
    print(f"Loaded NetHack KG:")
    print(f"  - Entities: {kg_info['num_entities']}")
    print(f"  - Relations: {kg_info['num_relations']}")
    print(f"  - Triples: {kg_info['num_triples']}")
    print(f"  - Data attributes: {list(kg_data.keys())}")
    
    try:
        # Import ULTRA components
        from ultra.tasks import build_relation_graph
        
        # Build relation graph
        print("\nBuilding relation graph...")
        print(f"Data has num_relations: {kg_data.num_relations}")
        
        # Try to build relation graph
        kg_data = build_relation_graph(kg_data)
        print("✓ Relation graph built successfully!")
        
        # Check what attributes were added
        print(f"\nData attributes after build_relation_graph: {list(kg_data.keys())}")
        
        # Now try to load the model
        from ultra.models import Ultra
        
        # Create minimal config without loading YAML
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
        print("✓ Model initialized")
        
        # Load checkpoint if available
        checkpoint_path = "ultra/ckpts/ultra_4g.pth"
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            model.eval()
            print("✓ Checkpoint loaded")
        
        # Test basic queries
        print("\n=== Testing Basic Queries ===")
        
        test_queries = [
            ("NetHack", "part of"),
            ("Amulet of Yendor", "present in work"),
            ("Sokoban", "part of"),
        ]
        
        with torch.no_grad():
            for head_name, rel_name in test_queries:
                if head_name in kg_info['entity2id'] and rel_name in kg_info['relation2id']:
                    h_idx = kg_info['entity2id'][head_name]
                    r_idx = kg_info['relation2id'][rel_name]
                    
                    print(f"\nQuery: {head_name} --[{rel_name}]--> ?")
                    
                    # Find ground truth
                    mask = (kg_data.edge_index[0] == h_idx) & (kg_data.edge_type == r_idx)
                    true_tails = kg_data.edge_index[1][mask].tolist()
                    
                    if true_tails:
                        print(f"Ground truth ({len(true_tails)} answers):")
                        for t_idx in true_tails[:3]:
                            print(f"  - {kg_info['id2entity'][t_idx]}")
                    
                    # Try model inference
                    try:
                        # For inference, create a batch with all possible tails
                        num_entities = kg_info['num_entities']
                        
                        # Create expanded batch for all candidate tails
                        # Shape: (num_entities, 3)
                        h_indices = torch.full((num_entities,), h_idx, dtype=torch.long)
                        t_indices = torch.arange(num_entities, dtype=torch.long)
                        r_indices = torch.full((num_entities,), r_idx, dtype=torch.long)
                        
                        # Stack into batch: (num_entities, 3)
                        batch = torch.stack([h_indices, t_indices, r_indices], dim=1)
                        
                        # Add batch dimension for ULTRA: (1, num_entities, 3)
                        batch = batch.unsqueeze(0)
                        
                        # Get predictions
                        with torch.no_grad():
                            scores = model(kg_data, batch)
                            
                        # scores should be shape [1, num_entities]
                        scores = scores.squeeze(0)
                        
                        # Get top predictions
                        top_k = min(5, len(scores))
                        top_scores, top_indices = torch.topk(scores, top_k)
                        
                        print("Model predictions:")
                        for i in range(top_k):
                            idx = top_indices[i].item()
                            score = top_scores[i].item()
                            entity = kg_info['id2entity'][idx]
                            is_correct = idx in true_tails
                            print(f"  {i+1}. {entity} (score: {score:.3f}){' ✓' if is_correct else ''}")
                            
                    except Exception as e:
                        print(f"Model inference failed: {e}")
                        import traceback
                        traceback.print_exc()
        
        print("\n✓ ULTRA integration test completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to understand the error
        if "num_relations" in str(e):
            print("\nThe error suggests the Data object needs additional attributes.")
            print("Let's check what build_relation_graph expects...")

if __name__ == "__main__":
    test_ultra_with_fixed_data()