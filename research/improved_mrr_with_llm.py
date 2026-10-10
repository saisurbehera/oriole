#!/usr/bin/env python3
"""
Improved MRR with LLM - using actual KG facts for evaluation
"""

import pandas as pd
import numpy as np
import requests
from collections import defaultdict
import random

class ImprovedMRRWithLLM:
    def __init__(self, kg_path="data/nethack_kg_final_a_plus.tsv", vllm_url="http://localhost:8080"):
        self.vllm_url = vllm_url
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        
        # Load KG
        print("Loading knowledge graph...")
        self.kg = pd.read_csv(kg_path, sep='\t')
        print(f"Loaded {len(self.kg)} triples")
        
        # Build structures
        self.entity_facts = defaultdict(list)
        self.relation_facts = defaultdict(list)
        
        for _, row in self.kg.iterrows():
            head, rel, tail = row['head'], row['type'], row['tail']
            self.entity_facts[head].append((rel, tail))
            self.relation_facts[rel].append((head, tail))
        
        print(f"Entities: {len(self.entity_facts)}, Relations: {len(self.relation_facts)}")
    
    def get_llm_ranking(self, head, relation, candidates):
        """Get LLM to rank candidates"""
        
        # Get context about head entity  
        head_facts = self.entity_facts.get(head, [])[:6]
        context = "\n".join([f"- {head} {rel} {tail}" for rel, tail in head_facts])
        
        # Get relation examples
        examples = self.relation_facts.get(relation, [])[:4]
        example_text = "\n".join([f"- {h} {relation} {t}" for h, t in examples if h != head])
        
        # Format candidates
        candidate_list = candidates[:8]  # Limit for better performance
        candidates_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidate_list)])
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a NetHack expert. Rank candidates by likelihood for completing the triple.<|eot_id|><|start_header_id|>user<|end_header_id|>

Entity facts:
{context}

Relation examples:
{example_text}

Complete: "{head} {relation} ?"

Candidates:
{candidates_text}

Rank them 1-{len(candidate_list)} (most likely first):<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        try:
            response = requests.post(
                f"{self.vllm_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.2,
                    "stop": ["<|eot_id|>"]
                },
                timeout=10
            )
            
            if response.status_code == 200:
                ranking_text = response.json()['choices'][0]['text'].strip()
                
                # Parse ranking - look for numbers
                ranking = []
                lines = ranking_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and line[0].isdigit():
                        try:
                            num = int(line[0]) - 1  # Convert to 0-based index
                            if 0 <= num < len(candidate_list):
                                candidate = candidate_list[num]
                                if candidate not in ranking:
                                    ranking.append(candidate)
                        except:
                            continue
                
                # Add any missing candidates
                for candidate in candidate_list:
                    if candidate not in ranking:
                        ranking.append(candidate)
                
                return ranking
            
        except Exception as e:
            print(f"LLM error: {e}")
        
        return candidates  # Fallback to original order
    
    def get_structural_scores(self, head, relation, candidates):
        """Get structural/frequency scores"""
        scores = {}
        
        # Get head entity's relations
        head_relations = set(rel for rel, _ in self.entity_facts.get(head, []))
        
        for candidate in candidates:
            score = 0.0
            
            # Frequency in this relation
            relation_tails = [t for h, t in self.relation_facts.get(relation, [])]
            freq = relation_tails.count(candidate)
            score += freq * 2.0
            
            # Structural similarity (shared relations)
            if candidate in self.entity_facts:
                cand_relations = set(rel for rel, _ in self.entity_facts[candidate])
                overlap = len(head_relations & cand_relations)
                score += overlap * 0.5
            
            # Co-occurrence bonus
            for rel, tail in self.entity_facts.get(head, []):
                if tail == candidate:
                    score += 1.0
            
            scores[candidate] = score
        
        return scores
    
    def evaluate_improved_mrr(self, num_queries=15):
        """Evaluate with improved hybrid approach"""
        
        # Select diverse test cases from actual KG
        test_cases = []
        
        # Get entities with multiple facts
        entities_with_facts = [(e, facts) for e, facts in self.entity_facts.items() 
                             if len(facts) >= 3]
        
        # Sort by number of facts (more facts = better test)
        entities_with_facts.sort(key=lambda x: len(x[1]), reverse=True)
        
        for entity, facts in entities_with_facts[:num_queries]:
            # Pick a fact to predict
            rel, tail = random.choice(facts)
            
            # Make sure there are other entities with this relation
            other_tails = [t for h, t in self.relation_facts.get(rel, []) if h != entity]
            if len(other_tails) >= 2:  # Need some candidates
                test_cases.append((entity, rel, tail))
        
        print(f"\nEvaluating Improved LLM+Structural on {len(test_cases)} queries...")
        
        mrr_scores = []
        hits_at_1 = []
        hits_at_3 = []
        hits_at_10 = []
        
        for i, (head, relation, target) in enumerate(test_cases):
            print(f"\n🔍 Query {i+1}: {head} {relation} ?")
            print(f"Target: {target}")
            
            # Get all possible tails for this relation
            all_candidates = list(set([t for h, t in self.relation_facts.get(relation, [])]))
            
            # Remove the head entity if it appears as a candidate
            if head in all_candidates:
                all_candidates.remove(head)
            
            if len(all_candidates) < 2:
                continue
            
            # Limit candidates for efficiency
            candidates = all_candidates[:15]
            if target not in candidates:
                candidates = candidates[:-1] + [target]  # Ensure target is included
            
            print(f"Candidates ({len(candidates)}): {candidates[:5]}...")
            
            # Get LLM ranking
            llm_ranking = self.get_llm_ranking(head, relation, candidates)
            
            # Get structural scores
            struct_scores = self.get_structural_scores(head, relation, candidates)
            
            # Combine rankings (LLM weight 0.6, structural 0.4)
            final_scores = {}
            for j, candidate in enumerate(candidates):
                # LLM score (inverse rank)
                llm_rank = llm_ranking.index(candidate) if candidate in llm_ranking else len(llm_ranking)
                llm_score = (len(candidates) - llm_rank) / len(candidates)
                
                # Structural score (normalized)
                struct_score = struct_scores.get(candidate, 0.0)
                max_struct = max(struct_scores.values()) if struct_scores.values() else 1.0
                struct_score_norm = struct_score / max_struct if max_struct > 0 else 0.0
                
                # Combined score
                final_scores[candidate] = 0.6 * llm_score + 0.4 * struct_score_norm
            
            # Final ranking
            final_ranking = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            final_candidates = [cand for cand, score in final_ranking]
            
            print(f"Top 3 predictions: {final_candidates[:3]}")
            
            # Calculate metrics
            if target in final_candidates:
                rank = final_candidates.index(target) + 1
                mrr = 1.0 / rank
                
                mrr_scores.append(mrr)
                hits_at_1.append(1 if rank == 1 else 0)
                hits_at_3.append(1 if rank <= 3 else 0)
                hits_at_10.append(1 if rank <= 10 else 0)
                
                print(f"✅ Target rank: {rank}, MRR: {mrr:.3f}")
            else:
                print(f"❌ Target not found in ranking")
                mrr_scores.append(0.0)
                hits_at_1.append(0)
                hits_at_3.append(0)
                hits_at_10.append(0)
        
        if mrr_scores:
            print(f"\n🎯 Improved LLM+Structural Results:")
            print(f"   Average MRR: {np.mean(mrr_scores):.3f}")
            print(f"   Hits@1: {np.mean(hits_at_1):.3f}")
            print(f"   Hits@3: {np.mean(hits_at_3):.3f}")
            print(f"   Hits@10: {np.mean(hits_at_10):.3f}")
            print(f"   Successful queries: {len(mrr_scores)}")
            
            return np.mean(mrr_scores)
        
        return 0.0

def main():
    print("🚀 Improved MRR with LLM Integration")
    print("="*50)
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Initialize system
    system = ImprovedMRRWithLLM()
    
    # Run evaluation
    final_mrr = system.evaluate_improved_mrr(num_queries=12)
    
    print(f"\n📊 Final Results:")
    print(f"   MRR: {final_mrr:.3f}")
    print(f"   Baseline: 0.603")
    print(f"   Target: >0.65")
    
    if final_mrr > 0.65:
        print("🎉 Target achieved! MRR improved with LLM integration!")
    elif final_mrr > 0.603:
        print("📈 Improvement over baseline!")
    else:
        print("🔧 Need further optimization...")

if __name__ == "__main__":
    main()