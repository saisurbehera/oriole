"""
Test ULTRA model with NetHack knowledge graph
"""

import torch
import pandas as pd
from torch_geometric.data import Data
import yaml
from easydict import EasyDict as edict
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultra'))

def load_nethack_kg(tsv_path):
    """Load NetHack KG from TSV file"""
    # Read TSV
    df = pd.read_csv(tsv_path, sep='\t', header=0, names=['head', 'relation', 'tail'])
    
    # Build entity and relation mappings
    entities = set()
    relations = set()
    
    for _, row in df.iterrows():
        # Convert to string and skip NaN values
        if pd.notna(row['head']) and pd.notna(row['tail']) and pd.notna(row['relation']):
            entities.add(str(row['head']))
            entities.add(str(row['tail']))
            relations.add(str(row['relation']))
    
    # Remove any empty strings
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
    
    return {
        'data': Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(entities)
        ),
        'entity2id': entity2id,
        'id2entity': id2entity,
        'relation2id': relation2id,
        'id2relation': id2relation,
        'num_entities': len(entities),
        'num_relations': len(relations),
        'num_triples': len(df)
    }

def test_ultra_inference():
    """Test ULTRA on NetHack KG"""
    
    print("=== Testing ULTRA with NetHack KG ===\n")
    
    # Load the KG
    kg_path = "data/nethack_kg_full/nethack_kg_intermediate.tsv"
    print(f"Loading KG from: {kg_path}")
    
    kg_info = load_nethack_kg(kg_path)
    kg_data = kg_info['data']
    
    print(f"Loaded NetHack KG:")
    print(f"  - Entities: {kg_info['num_entities']}")
    print(f"  - Relations: {kg_info['num_relations']}")
    print(f"  - Triples: {kg_info['num_triples']}")
    
    try:
        # Import ULTRA components
        from ultra.tasks import build_relation_graph
        from ultra.models import Ultra
        
        # Build relation graph
        print("\nBuilding relation graph...")
        kg_data = build_relation_graph(kg_data)
        print("✓ Relation graph built")
        
        # Load config
        config_path = "ultra/config/transductive/inference.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Handle the problematic key
            if 'model' in config and 'relation_graph' in config['model']:
                if 'layer' in config['model']['relation_graph']:
                    del config['model']['relation_graph']['layer']
            config = edict(config)
        
        # Initialize model
        print("\nInitializing ULTRA model...")
        model = Ultra(
            rel_model_cfg=config.model.relation_graph,
            entity_model_cfg=config.model.entity_graph
        )
        
        # Load checkpoint
        checkpoint_path = "ultra/ckpts/ultra_4g.pth"
        print(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()
        
        print("✓ Model loaded successfully!")
        
        # Test some queries
        print("\n=== Testing Link Prediction ===\n")
        
        test_queries = [
            ("NetHack", "genre"),
            ("Amulet of Yendor", "present in work"),
            ("Wizard of Yendor", "owner of"),
            ("Excalibur", "instance of"),
            ("Valkyrie", "instance of"),
            ("wand of wishing", "subclass of"),
            ("Sokoban", "part of"),
            ("Gehennom", "has part")
        ]
        
        with torch.no_grad():
            for head_name, rel_name in test_queries:
                if head_name in kg_info['entity2id'] and rel_name in kg_info['relation2id']:
                    h_idx = kg_info['entity2id'][head_name]
                    r_idx = kg_info['relation2id'][rel_name]
                    
                    # Create batch
                    batch = torch.tensor([[h_idx, 0, r_idx]], dtype=torch.long)
                    
                    # Get predictions
                    try:
                        pred = model(kg_data, batch)
                        scores = pred.squeeze()
                        
                        # Get top predictions
                        top_k = min(5, len(scores))
                        top_scores, top_indices = torch.topk(scores, top_k)
                        
                        print(f"Query: {head_name} --[{rel_name}]--> ?")
                        for i in range(top_k):
                            idx = top_indices[i].item()
                            score = top_scores[i].item()
                            if idx < kg_info['num_entities']:
                                entity = kg_info['id2entity'][idx]
                                print(f"  {i+1}. {entity} (score: {score:.3f})")
                        print()
                        
                    except Exception as e:
                        print(f"Error during inference: {e}")
                        print("This might be due to architecture mismatch\n")
                
                else:
                    missing = []
                    if head_name not in kg_info['entity2id']:
                        missing.append(f"entity '{head_name}'")
                    if rel_name not in kg_info['relation2id']:
                        missing.append(f"relation '{rel_name}'")
                    print(f"Skipping query - missing {' and '.join(missing)}\n")
        
        # Show some statistics about the KG
        print("\n=== KG Statistics ===")
        
        # Most connected entities
        entity_degrees = torch.zeros(kg_info['num_entities'])
        for i in range(kg_data.edge_index.shape[1]):
            entity_degrees[kg_data.edge_index[0, i]] += 1
            entity_degrees[kg_data.edge_index[1, i]] += 1
        
        top_entities_idx = torch.topk(entity_degrees, 10).indices
        
        print("\nMost connected entities:")
        for idx in top_entities_idx:
            entity = kg_info['id2entity'][idx.item()]
            degree = entity_degrees[idx].item()
            print(f"  - {entity}: {int(degree)} connections")
        
    except ImportError as e:
        print(f"Could not import ULTRA components: {e}")
        print("\nTo use ULTRA, ensure PyTorch Geometric is installed:")
        print("pip install torch-geometric")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ultra_inference()