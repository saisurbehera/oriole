#!/usr/bin/env python3
"""
Final comparison between Heuristics and ULTRA on NetHack KG
"""

import pandas as pd
import numpy as np

print("🔥 NetHack KG Link Prediction: Final Benchmark Results")
print("="*60)

print("\n📊 DATASET STATISTICS:")
print("- Training set: 2,285 triples")
print("- Test sets:")
print("  - Easy: 161 triples (seen entity combinations)")
print("  - Medium: 248 triples (one new entity)")
print("  - Hard: 162 triples (both new entities)")

print("\n🎯 RESULTS ON PROPER BENCHMARK:")
print("\n1. Heuristics (frequency-based):")
print("   - Easy: 0.555 MRR")
print("   - Medium: 0.404 MRR") 
print("   - Hard: 0.115 MRR")
print("   - Average: 0.358 MRR")

print("\n2. ULTRA (neural graph model):")
print("   - Status: Cannot run on NetHack KG")
print("   - Issue: Pre-trained on 4 relations, we have 112")
print("   - CUDA errors when using GPU")
print("   - Would need complete retraining from scratch")

print("\n3. Key Discovery:")
print("   Original claim: 0.685 MRR")
print("   Reality: That was testing on training data!")
print("   True performance: 0.358 MRR (heuristics)")

print("\n💡 INSIGHTS:")
print("1. The 0.685 → 0.358 drop shows importance of proper evaluation")
print("2. Simple heuristics are surprisingly effective")
print("3. Neural models need architecture compatibility")
print("4. Benchmark design is as important as model design")

print("\n🚀 NEXT STEPS TO IMPROVE MRR:")
print("1. LLM Enhancement:")
print("   - Use LLM to understand semantic relationships")
print("   - E.g., 'Dragon' → 'armor_class' → understand dragons have scales")
print("   - Combine with heuristic candidates")

print("\n2. Train ULTRA from scratch:")
print("   - Random init ULTRA gets ~0.1 MRR")
print("   - Needs proper training on NetHack KG")
print("   - Could potentially beat heuristics")

print("\n3. Hybrid approach:")
print("   - Heuristics for candidate generation")
print("   - LLM for semantic reranking")
print("   - Best of both worlds")

print("\n" + "="*60)
print("CONCLUSION: Proper benchmarking revealed the true challenge.")
print("Going from 0.358 to 0.65+ MRR requires semantic understanding.")