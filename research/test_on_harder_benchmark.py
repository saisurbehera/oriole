#!/usr/bin/env python3
"""
Test our heuristics on the harder benchmark
This will show the real performance when not cheating
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import requests

class HeuristicModel:
    """Our heuristic model that was getting 0.685 MRR"""
    
    def __init__(self, train_path="data/nethack_kg_train.tsv"):
        print("Loading training data...")
        self.train_kg = pd.read_csv(train_path, sep='\t')
        print(f"Training on {len(self.train_kg)} triples")
        
        # Build structures from TRAINING data only
        self.entity_facts = defaultdict(list)
        self.relation_facts = defaultdict(list)
        
        for _, row in self.train_kg.iterrows():
            head, rel, tail = row['head'], row['type'], row['tail']
            self.entity_facts[head].append((rel, tail))
            self.relation_facts[rel].append((head, tail))
    
    def predict(self, head, relation, max_candidates=10):
        """Heuristic scoring (same as before)"""
        
        # Get all possible tails for this relation FROM TRAINING DATA
        all_candidates = list(set([t for h, t in self.relation_facts.get(relation, [])]))
        
        if not all_candidates:
            return []
        
        scores = {}
        for candidate in all_candidates:
            score = 0.0
            
            # Direct frequency
            relation_tails = [t for h, t in self.relation_facts.get(relation, [])]
            freq = relation_tails.count(candidate)
            score += freq * 2.0
            
            # Co-occurrence
            head_facts = self.entity_facts.get(head, [])
            for rel, tail in head_facts:
                if tail == candidate:
                    score += 1.0
            
            scores[candidate] = score
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cand for cand, score in ranked[:max_candidates]]

class LLMEnhancedModel:
    """Heuristic + LLM model"""
    
    def __init__(self, train_path="data/nethack_kg_train.tsv", vllm_url="http://localhost:8080"):
        self.base_model = HeuristicModel(train_path)
        self.vllm_url = vllm_url
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    def predict(self, head, relation, max_candidates=10):
        # Get base predictions
        base_predictions = self.base_model.predict(head, relation, max_candidates)
        
        if len(base_predictions) <= 1:
            return base_predictions
        
        # LLM reranking
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a NetHack expert. Pick the most likely answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {head} {relation} ?

Options: {', '.join(base_predictions[:5])}

Most likely answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        try:
            response = requests.post(
                f"{self.vllm_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": 50,
                    "temperature": 0.1,
                    "stop": ["<|eot_id|>", "\n"]
                },
                timeout=5
            )
            
            if response.status_code == 200:
                llm_answer = response.json()['choices'][0]['text'].strip()
                
                # Reorder with LLM preference first
                reordered = []
                for cand in base_predictions:
                    if cand.lower() in llm_answer.lower() or llm_answer.lower() in cand.lower():
                        reordered.insert(0, cand)
                    else:
                        reordered.append(cand)
                
                return reordered
        except:
            pass
        
        return base_predictions

def evaluate_model(model, test_path, test_name):
    """Evaluate a model on test set"""
    
    print(f"\n{'='*50}")
    print(f"Evaluating on {test_name}")
    print(f"{'='*50}")
    
    test_kg = pd.read_csv(test_path, sep='\t')
    print(f"Test set size: {len(test_kg)} triples")
    
    # Sample for faster testing
    sample_size = min(100, len(test_kg))
    test_sample = test_kg.sample(n=sample_size, random_state=42)
    
    mrr_scores = []
    hits_at_1 = []
    hits_at_3 = []
    hits_at_10 = []
    no_predictions = 0
    
    for idx, row in test_sample.iterrows():
        head, rel, tail = row['head'], row['type'], row['tail']
        
        predictions = model.predict(head, rel)
        
        if not predictions:
            no_predictions += 1
            mrr_scores.append(0.0)
            hits_at_1.append(0)
            hits_at_3.append(0)
            hits_at_10.append(0)
            continue
        
        if tail in predictions:
            rank = predictions.index(tail) + 1
            mrr = 1.0 / rank
            
            mrr_scores.append(mrr)
            hits_at_1.append(1 if rank == 1 else 0)
            hits_at_3.append(1 if rank <= 3 else 0)
            hits_at_10.append(1 if rank <= 10 else 0)
        else:
            mrr_scores.append(0.0)
            hits_at_1.append(0)
            hits_at_3.append(0)
            hits_at_10.append(0)
    
    print(f"\nResults:")
    print(f"  MRR: {np.mean(mrr_scores):.3f}")
    print(f"  Hits@1: {np.mean(hits_at_1):.3f}")
    print(f"  Hits@3: {np.mean(hits_at_3):.3f}")
    print(f"  Hits@10: {np.mean(hits_at_10):.3f}")
    print(f"  No predictions: {no_predictions}/{sample_size}")
    
    return np.mean(mrr_scores)

def main():
    print("🔥 Testing on Harder Benchmark")
    print("="*50)
    
    # Test heuristic model
    print("\n1. Testing Pure Heuristics:")
    heuristic_model = HeuristicModel()
    
    easy_mrr = evaluate_model(heuristic_model, "data/nethack_kg_test_easy.tsv", "EASY Test Set")
    medium_mrr = evaluate_model(heuristic_model, "data/nethack_kg_test_medium.tsv", "MEDIUM Test Set")
    hard_mrr = evaluate_model(heuristic_model, "data/nethack_kg_test_hard.tsv", "HARD Test Set")
    zero_mrr = evaluate_model(heuristic_model, "data/nethack_kg_test_zero_shot.tsv", "ZERO-SHOT Test Set")
    
    print("\n" + "="*50)
    print("HEURISTIC MODEL SUMMARY:")
    print(f"  Easy: {easy_mrr:.3f}")
    print(f"  Medium: {medium_mrr:.3f}")
    print(f"  Hard: {hard_mrr:.3f}")
    print(f"  Zero-shot: {zero_mrr:.3f}")
    print(f"  Average: {np.mean([easy_mrr, medium_mrr, hard_mrr]):.3f}")
    
    # Test LLM-enhanced model
    print("\n\n2. Testing Heuristics + LLM:")
    llm_model = LLMEnhancedModel()
    
    llm_hard_mrr = evaluate_model(llm_model, "data/nethack_kg_test_hard.tsv", "HARD Test Set (with LLM)")
    
    print("\n" + "="*50)
    print("REALITY CHECK:")
    print(f"  'Easy' benchmark (cheating): 0.685 MRR")
    print(f"  Real hard benchmark: {hard_mrr:.3f} MRR")
    print(f"  With LLM enhancement: {llm_hard_mrr:.3f} MRR")
    print("\nThe heuristics only work well when they can memorize!")

if __name__ == "__main__":
    main()