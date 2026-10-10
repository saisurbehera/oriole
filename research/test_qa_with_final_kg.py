#!/usr/bin/env python3
"""
Test QA system with the final extracted KG
"""

from nethack_kgqa_fast_v2 import FastNetHackKGQA

# Clear cache to use new KG
import shutil
import os

cache_dir = "data/kgqa_cache"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Cleared cache at {cache_dir}")

# Initialize QA with new KG
print("\nInitializing QA system with final KG...")
qa = FastNetHackKGQA('data/nethack_kg_final_100.tsv')

# Test questions
test_questions = [
    "What is soldier ant?",
    "How much damage does a soldier ant do?",
    "What resistance does soldier ant have?",
    "What is the AC of Wumpus?",
    "How difficult is Medusa?",
    "What does soldier ant weigh?",
    "What alignment is soldier ant?",
    "What type of attacks does Medusa have?"
]

print("\nTesting questions:")
print("-" * 60)

for q in test_questions:
    print(f"\nQ: {q}")
    answer = qa.answer(q)
    print(f"A: {answer.text}")
    print(f"Confidence: {answer.confidence:.1%}")
    
    if answer.evidence_triples:
        print("Evidence:")
        for h, r, t in answer.evidence_triples[:2]:
            print(f"  - ({h}) --[{r}]--> ({t})")