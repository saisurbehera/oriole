#!/usr/bin/env python3
"""
Prepare NetHack KG data in ULTRA format for fine-tuning
"""

import pandas as pd
import os

def convert_to_ultra_format():
    """Convert NetHack KG to ULTRA expected format"""
    print("📊 Converting NetHack KG to ULTRA format")
    print("="*50)
    
    # Read NetHack data
    train_df = pd.read_csv("data/nethack_kg_train_proper.tsv", sep='\t')
    
    # Create output directory
    os.makedirs("data/nethack_ultra", exist_ok=True)
    
    # Convert to space-separated format (head relation tail)
    train_ultra = train_df[['head', 'type', 'tail']]
    
    # Save in ULTRA expected format (tab-separated to handle spaces in names)
    train_ultra.to_csv("data/nethack_ultra/train.txt", 
                      sep='\t', header=False, index=False)
    
    # For validation and test, we'll use the medium test set
    # (ULTRA expects all 3 files even for fine-tuning)
    test_df = pd.read_csv("data/nethack_kg_test_medium_proper.tsv", sep='\t')
    test_ultra = test_df[['head', 'type', 'tail']]
    
    test_ultra.to_csv("data/nethack_ultra/valid.txt", 
                     sep='\t', header=False, index=False)
    test_ultra.to_csv("data/nethack_ultra/test.txt", 
                     sep='\t', header=False, index=False)
    
    print(f"✓ Training set: {len(train_ultra)} triples → data/nethack_ultra/train.txt")
    print(f"✓ Valid/Test set: {len(test_ultra)} triples → data/nethack_ultra/valid.txt")
    print(f"✓ Files ready for ULTRA!")
    
    return len(train_ultra), len(test_ultra)

if __name__ == "__main__":
    convert_to_ultra_format()