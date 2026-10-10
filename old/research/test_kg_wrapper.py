"""
Test script for KG QA wrapper with sample data files
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultra'))

# For testing without full model, let's create a simplified version
import torch
import pandas as pd
from torch_geometric.data import Data

def test_movies_kg():
    """Test with movies knowledge graph"""
    print("=== Testing Movies KG ===\n")
    
    # Initialize wrapper
    qa = KnowledgeGraphQA(
        checkpoint_path="ultra/ckpts/ultra_4g.pth",
        device="cpu"  # Use CPU for testing
    )
    
    # Load KG from file
    print("Loading movies KG from file...")
    qa.parse_kg_from_file("sample_data/movies_kg.tsv", delimiter="\t")
    
    print(f"Loaded KG with {len(qa.entity2id)} entities and {len(qa.relation2id)} relations")
    print(f"Entities: {list(qa.entity2id.keys())[:10]}...")
    print(f"Relations: {list(qa.relation2id.keys())}\n")
    
    # Load model
    print("Loading ULTRA model...")
    try:
        qa.load_model()
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Note: Model loading skipped for demo - {e}\n")
        print("Continuing with other tests...\n")
    
    # Test entity relations lookup
    print("Testing entity relations lookup:")
    entity = "Christopher_Nolan"
    relations = qa.get_entity_relations(entity)
    print(f"Relations for {entity}: {relations}\n")
    
    # If model was loaded, test predictions
    if qa.model is not None:
        print("Testing link prediction:")
        results = qa.link_prediction("Christopher_Nolan", "directed", top_k=5)
        print(f"What movies did Christopher Nolan direct?")
        for movie, score in results:
            print(f"  - {movie}: {score:.3f}")
        print()
        
        results = qa.link_prediction("Inception", "genre", top_k=3)
        print(f"What genres is Inception?")
        for genre, score in results:
            print(f"  - {genre}: {score:.3f}")
        print()

def test_tech_kg():
    """Test with tech companies knowledge graph"""
    print("\n=== Testing Tech KG ===\n")
    
    # Initialize wrapper
    qa = KnowledgeGraphQA(device="cpu")
    
    # Load KG from file
    print("Loading tech KG from file...")
    qa.parse_kg_from_file("sample_data/tech_kg.tsv", delimiter="\t")
    
    print(f"Loaded KG with {len(qa.entity2id)} entities and {len(qa.relation2id)} relations")
    print(f"Sample entities: {list(qa.entity2id.keys())[:10]}...")
    print(f"Sample relations: {list(qa.relation2id.keys())[:5]}...\n")
    
    # Test without model (just data loading and parsing)
    print("Testing entity lookup:")
    for entity in ["Google", "Apple", "Microsoft"]:
        relations = qa.get_entity_relations(entity)
        print(f"{entity} has relations: {relations}")

def test_custom_triples():
    """Test with custom triples"""
    print("\n=== Testing Custom Triples ===\n")
    
    qa = KnowledgeGraphQA(device="cpu")
    
    # Create custom triples
    triples = [
        ("Alice", "friendOf", "Bob"),
        ("Bob", "friendOf", "Charlie"),
        ("Alice", "worksAt", "TechCorp"),
        ("Bob", "worksAt", "TechCorp"),
        ("Charlie", "worksAt", "StartupXYZ"),
        ("TechCorp", "locatedIn", "NYC"),
        ("StartupXYZ", "locatedIn", "SF"),
        ("Alice", "livesIn", "NYC"),
        ("Bob", "livesIn", "Brooklyn"),
        ("Charlie", "livesIn", "SF"),
    ]
    
    print("Loading custom KG...")
    qa.parse_kg_from_triples(triples)
    
    print(f"Created KG with {len(qa.entity2id)} entities and {len(qa.relation2id)} relations")
    print(f"Entities: {list(qa.entity2id.keys())}")
    print(f"Relations: {list(qa.relation2id.keys())}\n")
    
    # Test graph structure
    print("Graph structure:")
    for entity in ["Alice", "Bob", "Charlie"]:
        relations = qa.get_entity_relations(entity)
        print(f"  {entity}: {relations}")

def visualize_kg_stats(qa):
    """Print basic KG statistics"""
    if qa.kg_data is None:
        print("No KG data loaded")
        return
        
    print("\n=== KG Statistics ===")
    print(f"Number of entities: {len(qa.entity2id)}")
    print(f"Number of relations: {len(qa.relation2id)}")
    print(f"Number of triples: {qa.kg_data.edge_index.shape[1]}")
    
    # Relation distribution
    print("\nRelation distribution:")
    relation_counts = {}
    for r_idx in qa.kg_data.edge_type.tolist():
        r_name = qa.id2relation[r_idx]
        relation_counts[r_name] = relation_counts.get(r_name, 0) + 1
    
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {rel}: {count} triples")

if __name__ == "__main__":
    # Create sample data directory if it doesn't exist
    os.makedirs("sample_data", exist_ok=True)
    
    print("KG QA Wrapper Test Suite")
    print("=" * 50)
    
    # Run tests
    test_movies_kg()
    test_tech_kg()
    test_custom_triples()
    
    print("\n" + "=" * 50)
    print("Tests completed!")
    print("\nNote: Full model inference requires downloading ULTRA checkpoints.")
    print("The tests demonstrate KG loading, parsing, and basic operations.")