#!/usr/bin/env python3
"""
NetHack dataset for ULTRA fine-tuning
"""

import sys
import os

# Add ULTRA to path
sys.path.append('ultra')

from ultra.datasets import TransductiveDataset

class NetHackDataset(TransductiveDataset):
    """NetHack KG dataset for ULTRA"""
    
    # Use file:// URLs for local files
    urls = [
        f"file://{os.path.abspath('data/nethack_ultra/train.txt')}",
        f"file://{os.path.abspath('data/nethack_ultra/valid.txt')}",
        f"file://{os.path.abspath('data/nethack_ultra/test.txt')}",
    ]
    
    name = "NetHack"
    delimiter = " "  # Space-separated format

def test_dataset():
    """Test that the dataset loads correctly"""
    print("🧪 Testing NetHack dataset for ULTRA")
    print("="*50)
    
    try:
        dataset = NetHackDataset(root="data/")
        
        print(f"✓ Dataset loaded successfully")
        print(f"✓ Name: {dataset.name}")
        print(f"✓ Train graph: {dataset.train_graph}")
        print(f"✓ Valid graph: {dataset.valid_graph}")
        print(f"✓ Test graph: {dataset.test_graph}")
        
        # Print some stats
        train_data = dataset.train_graph
        print(f"\nDataset statistics:")
        print(f"  Nodes: {train_data.num_nodes}")
        print(f"  Edges: {train_data.num_edges}")
        print(f"  Relations: {train_data.num_relations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dataset()