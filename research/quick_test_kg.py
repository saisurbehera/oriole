#!/usr/bin/env python3
"""
Quick test to verify KG processing works
"""

import os
import pandas as pd

print("Quick KG Processing Test")
print("="*50)

# Check for required files
files_to_check = [
    ('XML file', 'data/nethackwiki_current.xml'),
    ('Process script', 'process_full_wiki_to_a_plus.py'),
    ('Extraction script', 'optimized_extraction.py'),
    ('Enhancement script', 'kg_enhancement_pipeline.py'),
    ('Advanced enhancer', 'advanced_kg_enhancer.py')
]

all_good = True
for name, path in files_to_check:
    if os.path.exists(path):
        print(f"✓ {name}: {path}")
    else:
        print(f"✗ {name}: {path} NOT FOUND")
        all_good = False

if not all_good:
    print("\nSome files are missing. Cannot proceed.")
    exit(1)

# Check if we have any existing KG
existing_kgs = [
    'data/nethack_kg_optimized_100.tsv',
    'data/nethack_kg_partial.tsv',
    'data/nethack_kg_enhanced.tsv',
    'data/nethack_kg_final_a_plus.tsv'
]

print("\nExisting KGs:")
for kg_path in existing_kgs:
    if os.path.exists(kg_path):
        df = pd.read_csv(kg_path, sep='\t')
        print(f"  {kg_path}: {len(df)} triples")

print("\nTo process the full wiki to A+ quality, run:")
print("  python3 research/process_full_wiki_to_a_plus.py")
print("\nOr use the convenience script:")
print("  ./create_nethack_kg.sh")