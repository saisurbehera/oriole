"""
Test KG wrapper with ULTRA model (if available)
"""

import sys
import os
import torch

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultra'))

def test_with_ultra():
    """Test with actual ULTRA model if dependencies are available"""
    
    try:
        # Try importing ULTRA components
        print("Attempting to import ULTRA components...")
        from ultra.datasets import build_relation_graph
        from ultra.models import Ultra
        from torch_geometric.data import Data
        import yaml
        from easydict import EasyDict as edict
        
        print("✓ ULTRA imports successful!")
        
        # Load NetHack KG data
        print("\nLoading NetHack A+ KG...")
        import pandas as pd
        
        # Load the A+ KG
        kg_df = pd.read_csv('data/nethack_kg_final_a_plus.tsv', sep='\t')
        print(f"Loaded {len(kg_df)} triples from A+ KG")
        
        # Take a subset for testing (ULTRA can be memory intensive)
        kg_subset = kg_df.head(1000)
        
        # Create entity and relation mappings
        all_entities = list(set(kg_subset['head'].tolist() + kg_subset['tail'].tolist()))
        all_relations = list(kg_subset['type'].unique())
        
        entity_to_id = {entity: i for i, entity in enumerate(all_entities)}
        relation_to_id = {relation: i for i, relation in enumerate(all_relations)}
        
        # Create triples with IDs
        triples = []
        for _, row in kg_subset.iterrows():
            h_id = entity_to_id[row['head']]
            r_id = relation_to_id[row['type']]
            t_id = entity_to_id[row['tail']]
            triples.append((h_id, r_id, t_id))
        
        entity_names = all_entities
        relation_names = all_relations
        
        print(f"NetHack KG subset: {len(entity_names)} entities, {len(relation_names)} relations")
        
        # Create edge tensors
        edge_index = torch.tensor([[h, t] for h, r, t in triples], dtype=torch.long).t()
        edge_type = torch.tensor([r for h, r, t in triples], dtype=torch.long)
        
        # Create PyG Data object
        kg_data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(entity_names)
        )
        
        print(f"Created KG with {kg_data.num_nodes} entities and {len(relation_names)} relations")
        
        # Build relation graph
        print("\nBuilding relation graph...")
        kg_data = build_relation_graph(kg_data)
        print("✓ Relation graph built successfully!")
        
        # Check if checkpoint exists
        checkpoint_paths = [
            "ultra/ckpts/ultra_4g.pth",
            "ultra/ckpts/ultra_3g.pth",
            "/home/sai/Desktop/oriole/ultra/ckpts/ultra_4g.pth",
            "/home/sai/Desktop/oriole/ultra/ckpts/ultra_3g.pth"
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path:
            print(f"\nFound checkpoint: {checkpoint_path}")
            
            # Load config
            config_paths = [
                "ultra/config/transductive/inference.yaml",
                "/home/sai/Desktop/oriole/ultra/config/transductive/inference.yaml"
            ]
            
            config_path = None
            for path in config_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path:
                print(f"Loading config from: {config_path}")
                with open(config_path, 'r') as f:
                    config = edict(yaml.safe_load(f))
                
                # Initialize model
                print("\nInitializing ULTRA model...")
                model = Ultra(
                    rel_model_cfg=config.model.relation_graph,
                    entity_model_cfg=config.model.entity_graph
                )
                
                # Load checkpoint
                print("Loading model weights...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'], strict=False)
                model.eval()
                
                print("✓ Model loaded successfully!")
                
                # Test inference
                print("\n=== Testing Link Prediction ===")
                
                # Create a test query: Who does Alice know? (0, 0, ?)
                with torch.no_grad():
                    batch = torch.tensor([[0, -1, 0]], dtype=torch.long)  # head=Alice, rel=knows, tail=?
                    
                    print(f"\nQuery: Who does {entity_names[0]} know?")
                    
                    try:
                        pred = model(kg_data, batch)
                        scores = pred.squeeze()
                        
                        # Get predictions
                        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                        
                        print("Predictions:")
                        for i in range(min(3, len(sorted_indices))):
                            idx = sorted_indices[i]
                            score = sorted_scores[i]
                            if idx < len(entity_names):
                                print(f"  {entity_names[idx]}: {score:.3f}")
                    
                    except Exception as e:
                        print(f"Inference error: {e}")
                        print("This might be due to model architecture mismatch or missing features")
                
            else:
                print("Config file not found")
        else:
            print("\nNo ULTRA checkpoint found. Would need to download from:")
            print("- https://github.com/DeepGraphLearning/ULTRA")
            print("- Or from HuggingFace: https://huggingface.co/collections/mgalkin/ultra-65699bb28369400a5827669d")
        
    except ImportError as e:
        print(f"Could not import ULTRA components: {e}")
        print("\nTo use ULTRA, you need to:")
        print("1. Install PyTorch and PyTorch Geometric")
        print("2. Install ULTRA dependencies: pip install easydict pyyaml")
        print("3. Ensure the ULTRA submodule is properly initialized")
        
    except Exception as e:
        print(f"Error during ULTRA test: {e}")
        import traceback
        traceback.print_exc()


def test_wrapper_without_model():
    """Test the wrapper functionality without requiring the full model"""
    print("\n\n=== Testing Wrapper Logic (No Model Required) ===")
    
    try:
        from kg_qa_wrapper import KnowledgeGraphQA
        
        print("Testing KnowledgeGraphQA wrapper...")
        qa = KnowledgeGraphQA(device="cpu")
        
        # Test with simple triples
        triples = [
            ("Alice", "knows", "Bob"),
            ("Bob", "knows", "Charlie"),
            ("Alice", "works_at", "TechCorp"),
            ("Bob", "works_at", "TechCorp"),
        ]
        
        qa.parse_kg_from_triples(triples)
        print(f"✓ Loaded {len(qa.entity2id)} entities and {len(qa.relation2id)} relations")
        
        # Test entity lookup
        print("\nEntity relations:")
        for entity in ["Alice", "Bob"]:
            rels = qa.get_entity_relations(entity)
            print(f"  {entity}: {rels}")
            
    except ImportError:
        print("Could not import kg_qa_wrapper - dependencies missing")


if __name__ == "__main__":
    print("ULTRA Model Test")
    print("=" * 50)
    
    # Test with ULTRA if available
    test_with_ultra()
    
    # Test wrapper logic
    test_wrapper_without_model()
    
    print("\n" + "=" * 50)
    print("Test completed!")