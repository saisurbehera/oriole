#!/usr/bin/env python3
"""
Debug script for FAISS-based NetHack QA system
Run this to test the entity matching issues
"""

import sys
import os

# Check if FAISS is available
try:
    import faiss
    print("✓ FAISS is installed")
except ImportError:
    print("✗ FAISS not found. Install with: pip install faiss-cpu")
    sys.exit(1)

# Check other dependencies
try:
    from sentence_transformers import SentenceTransformer
    print("✓ Sentence Transformers is installed")
except ImportError:
    print("✗ Sentence Transformers not found. Install with: pip install sentence-transformers")
    sys.exit(1)

try:
    import pandas as pd
    print("✓ Pandas is installed")
except ImportError:
    print("✗ Pandas not found. Install with: pip install pandas")
    sys.exit(1)

print("\nAll dependencies found. Testing FAISS QA system...\n")

# Import the FAISS QA system
try:
    from nethack_kgqa_fast import FastNetHackKGQA
except ImportError as e:
    print(f"Error importing FastNetHackKGQA: {e}")
    sys.exit(1)

# Initialize the system
print("Loading NetHack KG...")
kg_path = 'data/nethack_kg_full/nethack_kg_intermediate.tsv'

if not os.path.exists(kg_path):
    print(f"Error: KG file not found at {kg_path}")
    print("Please make sure the knowledge graph has been generated.")
    sys.exit(1)

qa = FastNetHackKGQA(kg_path)
print(f"Loaded {len(qa.entities)} entities, {len(qa.relations)} relations")

# Debug entity data
print("\n" + "="*60)
print("ENTITY DATA ANALYSIS")
print("="*60)

# Check for problematic single-character entities
single_char_entities = [e for e in qa.entities if len(e) == 1]
print(f"\nSingle character entities: {len(single_char_entities)}")
if single_char_entities:
    print("Examples:", single_char_entities[:10])

# Look for key entities
print("\nKey entities check:")
key_entities = ['Amulet of Yendor', 'NetHack', 'Sokoban', 'Gehennom', 'Excalibur']
for entity in key_entities:
    if entity in qa.entities:
        print(f"  ✓ Found: {entity}")
    else:
        # Check case variations
        found = False
        for e in qa.entities:
            if entity.lower() == e.lower():
                print(f"  ~ Found (different case): {e}")
                found = True
                break
        if not found:
            print(f"  ✗ Missing: {entity}")

# Sample entities
print("\nSample entities (length > 3):")
meaningful_entities = [e for e in qa.entities if len(e) > 3]
for entity in sorted(meaningful_entities)[:20]:
    print(f"  - {entity}")

# Test entity matching
print("\n" + "="*60)
print("ENTITY MATCHING TEST")
print("="*60)

test_queries = [
    "What is the Amulet of Yendor?",
    "Tell me about amulet of yendor",
    "amulet yendor"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    matches = qa._match_entities_fast(query.lower())
    print(f"Matched {len(matches)} entities:")
    for match in matches[:5]:
        print(f"  - {match.name} (score: {match.final_score:.3f}, exact: {match.exact_match})")

# Test the full QA
print("\n" + "="*60)
print("FULL QA TEST")
print("="*60)

questions = [
    "What is the Amulet of Yendor?",
    "What is NetHack?",
    "What is Sokoban?"
]

for q in questions:
    print(f"\nQ: {q}")
    answer = qa.answer(q)
    print(f"A: {answer.text}")
    print(f"Confidence: {answer.confidence:.1%}")
    if answer.evidence_triples:
        print("Evidence:")
        for h, r, t in answer.evidence_triples[:2]:
            print(f"  - ({h}) --[{r}]--> ({t})")

# Check entity importance scores
print("\n" + "="*60)
print("ENTITY IMPORTANCE ANALYSIS")
print("="*60)

# Top entities by importance
sorted_entities = sorted(qa.entity_importance.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 entities by importance:")
for entity, score in sorted_entities[:10]:
    if len(entity) > 1:  # Skip single chars
        print(f"  - {entity}: {score:.3f}")

# Check specific entity triples
print("\n" + "="*60)
print("ENTITY TRIPLE ANALYSIS")
print("="*60)

# Look for Amulet of Yendor
amulet_entities = [e for e in qa.entities if 'amulet' in e.lower() and 'yendor' in e.lower()]
if amulet_entities:
    for entity in amulet_entities[:3]:
        print(f"\nTriples for '{entity}':")
        triples = qa.entity_triples.get(entity, [])
        for idx, role, rel, other in triples[:5]:
            if role == 'head':
                print(f"  - {entity} --[{rel}]--> {other}")
            else:
                print(f"  - {other} --[{rel}]--> {entity}")
else:
    print("\nNo 'Amulet of Yendor' entities found!")
    
    # Try to find partial matches
    print("\nEntities with 'amulet':")
    for e in qa.entities:
        if 'amulet' in e.lower():
            print(f"  - {e}")
    
    print("\nEntities with 'yendor':")
    for e in qa.entities:
        if 'yendor' in e.lower():
            print(f"  - {e}")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)