#!/usr/bin/env python3
"""
Enhanced ULTRA with LLM integration for better MRR
Combines knowledge graph reasoning with language model predictions
"""

import pandas as pd
import torch
import numpy as np
import requests
from collections import defaultdict
import json
import time

class EnhancedULTRAWithLLM:
    def __init__(self, kg_path="data/nethack_kg_final_a_plus.tsv", vllm_url="http://localhost:8080"):
        self.vllm_url = vllm_url
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        
        # Load KG
        print("Loading knowledge graph...")
        self.kg = pd.read_csv(kg_path, sep='\t')
        print(f"Loaded {len(self.kg)} triples")
        
        # Build entity and relation mappings
        self.entities = sorted(list(set(self.kg['head'].tolist() + self.kg['tail'].tolist())))
        self.relations = sorted(self.kg['type'].unique().tolist())
        
        self.entity2id = {e: i for i, e in enumerate(self.entities)}
        self.relation2id = {r: i for i, r in enumerate(self.relations)}
        
        # Build adjacency structures
        self.entity_relations = defaultdict(list)
        self.relation_facts = defaultdict(list)
        
        for _, row in self.kg.iterrows():
            head, rel, tail = row['head'], row['type'], row['tail']
            self.entity_relations[head].append((rel, tail))
            self.relation_facts[rel].append((head, tail))
        
        print(f"Entities: {len(self.entities)}, Relations: {len(self.relations)}")
    
    def get_llm_prediction(self, head_entity, relation, candidates, max_candidates=10):
        """Use LLM to rank candidates for (head, relation, ?) queries"""
        
        # Get context about the head entity
        head_facts = self.entity_relations.get(head_entity, [])[:8]
        context_lines = [f"- {head_entity} {rel} {tail}" for rel, tail in head_facts]
        context = "\n".join(context_lines) if context_lines else f"Entity: {head_entity}"
        
        # Get examples of this relation
        relation_examples = self.relation_facts.get(relation, [])[:5]
        examples = [f"- {h} {relation} {t}" for h, t in relation_examples]
        examples_text = "\n".join(examples) if examples else f"Relation: {relation}"
        
        # Limit candidates for efficiency
        candidate_list = candidates[:max_candidates]
        candidates_text = ", ".join(candidate_list)
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a NetHack expert. Given entity facts and relation examples, predict which candidate is most likely to complete the triple.<|eot_id|><|start_header_id|>user<|end_header_id|>

Entity facts:
{context}

Relation examples:
{examples_text}

Question: What is most likely for "{head_entity} {relation} ?"

Candidates: {candidates_text}

Answer with just the most likely candidate:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        try:
            response = requests.post(
                f"{self.vllm_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": 50,
                    "temperature": 0.1,
                    "stop": ["<|eot_id|>", "\n", ","]
                },
                timeout=8
            )
            
            if response.status_code == 200:
                prediction = response.json()['choices'][0]['text'].strip()
                # Clean the prediction
                prediction = prediction.split('\n')[0].strip()
                return prediction
            
        except Exception as e:
            print(f"LLM prediction error: {e}")
        
        return None
    
    def get_ultra_scores(self, head, relation, candidates):
        """Get ULTRA-style scores (simplified structural scoring)"""
        scores = {}
        head_rels = set(rel for rel, _ in self.entity_relations.get(head, []))
        
        for candidate in candidates:
            score = 0.0
            candidate_rels = set(rel for rel, _ in self.entity_relations.get(candidate, []))
            
            # Relation overlap bonus
            overlap = len(head_rels & candidate_rels)
            score += overlap * 0.3
            
            # Frequency bonus for this relation
            relation_tails = [t for h, t in self.relation_facts.get(relation, []) if h != head]
            if candidate in relation_tails:
                score += 2.0
            
            # Co-occurrence bonus
            for rel, tail in self.entity_relations.get(head, []):
                if tail == candidate:
                    score += 1.0
            
            scores[candidate] = score
        
        return scores
    
    def hybrid_predict(self, head, relation, candidates, llm_weight=0.4, ultra_weight=0.6):
        """Combine LLM and ULTRA predictions"""
        
        # Get ULTRA scores
        ultra_scores = self.get_ultra_scores(head, relation, candidates)
        max_ultra = max(ultra_scores.values()) if ultra_scores.values() else 1.0
        if max_ultra > 0:
            ultra_scores = {k: v/max_ultra for k, v in ultra_scores.items()}
        
        # Get LLM prediction
        llm_prediction = self.get_llm_prediction(head, relation, candidates)
        
        # Combine scores
        final_scores = {}
        for candidate in candidates:
            ultra_score = ultra_scores.get(candidate, 0.0)
            llm_score = 1.0 if candidate == llm_prediction else 0.0
            
            final_scores[candidate] = ultra_weight * ultra_score + llm_weight * llm_score
        
        # Sort by score
        ranked_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [cand for cand, score in ranked_candidates]
    
    def evaluate_mrr(self, test_queries=None, max_queries=15):
        """Evaluate MRR on test queries"""
        
        if test_queries is None:
            # Create test queries from existing triples
            test_queries = []
            entities_with_facts = [e for e in self.entities if len(self.entity_relations[e]) >= 3]
            
            for entity in entities_with_facts[:max_queries]:
                facts = self.entity_relations[entity]
                if len(facts) >= 2:
                    # Pick a random fact to predict
                    import random
                    relation, target = random.choice(facts)
                    test_queries.append((entity, relation, target))
        
        print(f"\nEvaluating Enhanced ULTRA+LLM on {len(test_queries)} queries...")
        
        mrr_scores = []
        hits_at_1 = []
        hits_at_3 = []
        hits_at_10 = []
        
        for i, (head, relation, target) in enumerate(test_queries):
            print(f"\nQuery {i+1}: {head} {relation} ?")
            print(f"Target: {target}")
            
            # Get all possible candidates for this relation
            all_tails = [t for h, t in self.relation_facts.get(relation, [])]
            if not all_tails:
                print("No candidates found")
                continue
            
            # Remove duplicates and the head entity itself
            candidates = list(set(all_tails))
            if head in candidates:
                candidates.remove(head)
            
            if not candidates:
                print("No valid candidates")
                continue
            
            # Get hybrid predictions
            ranked_candidates = self.hybrid_predict(head, relation, candidates)
            
            print(f"Top predictions: {ranked_candidates[:5]}")
            
            # Calculate metrics
            try:
                rank = ranked_candidates.index(target) + 1
                mrr = 1.0 / rank
                hit_1 = 1 if rank == 1 else 0
                hit_3 = 1 if rank <= 3 else 0
                hit_10 = 1 if rank <= 10 else 0
                
                print(f"Target rank: {rank}, MRR: {mrr:.3f}")
                
                mrr_scores.append(mrr)
                hits_at_1.append(hit_1)
                hits_at_3.append(hit_3)
                hits_at_10.append(hit_10)
                
            except ValueError:
                print(f"Target '{target}' not found in candidates")
                mrr_scores.append(0.0)
                hits_at_1.append(0)
                hits_at_3.append(0)
                hits_at_10.append(0)
        
        if mrr_scores:
            print(f"\n🎯 Enhanced ULTRA+LLM Results:")
            print(f"Average MRR: {np.mean(mrr_scores):.3f}")
            print(f"Hits@1: {np.mean(hits_at_1):.3f}")
            print(f"Hits@3: {np.mean(hits_at_3):.3f}")
            print(f"Hits@10: {np.mean(hits_at_10):.3f}")
            
            return {
                'mrr': np.mean(mrr_scores),
                'hits_at_1': np.mean(hits_at_1),
                'hits_at_3': np.mean(hits_at_3),
                'hits_at_10': np.mean(hits_at_10),
                'num_queries': len(mrr_scores)
            }
        
        return None

def main():
    print("🚀 Enhanced ULTRA with LLM Integration")
    print("="*50)
    
    # Initialize enhanced system
    enhancer = EnhancedULTRAWithLLM()
    
    # Test specific challenging queries
    test_queries = [
        ("Medusa", "location", "Dungeon level 20"),
        ("Dragon", "difficulty", "15"),
        ("Soldier ant", "speed", "18"),
        ("Floating eye", "corpse_effect", "Telepathy"),
        ("Minotaur", "location", "Maze"),
        ("Lich", "magic_resistance", "High"),
        ("Troll", "regeneration", "Yes"),
        ("Unicorn", "horn_material", "Unicorn horn"),
        ("Giant spider", "web_ability", "Creates webs"),
        ("Vampire", "drain_ability", "Drains levels"),
    ]
    
    # Run evaluation
    results = enhancer.evaluate_mrr(test_queries, max_queries=10)
    
    if results:
        print(f"\n📊 Final Enhanced Results:")
        print(f"   MRR: {results['mrr']:.3f}")
        print(f"   Improvement target: >0.65 (vs 0.603 baseline)")
        
        if results['mrr'] > 0.65:
            print("✅ Target achieved!")
        else:
            print("🎯 Still improving...")

if __name__ == "__main__":
    main()