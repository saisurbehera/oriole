#!/usr/bin/env python3
"""
Real ULTRA + LLM Integration
Combines actual ULTRA neural model with LLM enhancement
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import requests
from collections import defaultdict
from easydict import EasyDict as edict
from torch_geometric.data import Data

# Add ULTRA path
sys.path.append('ultra')
from ultra import Ultra
from ultra.tasks import build_relation_graph

class RealULTRAWithLLM:
    def __init__(self, kg_path="data/nethack_kg_final_a_plus.tsv", vllm_url="http://localhost:8080"):
        self.vllm_url = vllm_url
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        
        print("🚀 Loading Real ULTRA + LLM System")
        print("="*50)
        
        # Load KG
        print("Loading knowledge graph...")
        self.kg = pd.read_csv(kg_path, sep='\t')
        print(f"Loaded {len(self.kg)} triples")
        
        # Build entity and relation mappings
        entities = set()
        relations = set()
        
        for _, row in self.kg.iterrows():
            entities.add(row['head'])
            entities.add(row['tail'])
            relations.add(row['type'])
        
        self.entities = sorted(list(entities))
        self.relations = sorted(list(relations))
        
        self.entity2id = {e: i for i, e in enumerate(self.entities)}
        self.relation2id = {r: i for i, r in enumerate(self.relations)}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.id2relation = {i: r for r, i in self.relation2id.items()}
        
        print(f"Entities: {len(self.entities)}, Relations: {len(self.relations)}")
        
        # Build PyG data structure
        self.kg_data = self._build_pyg_data()
        
        # Initialize ULTRA model
        self.ultra_model = self._initialize_ultra()
        
        # Build helper structures
        self.entity_facts = defaultdict(list)
        for _, row in self.kg.iterrows():
            self.entity_facts[row['head']].append((row['type'], row['tail']))
        
        print("✅ Real ULTRA + LLM system ready!")
    
    def _build_pyg_data(self):
        """Build PyTorch Geometric data structure"""
        print("Building PyG data structure...")
        
        edge_list = []
        edge_type = []
        
        for _, row in self.kg.iterrows():
            h_id = self.entity2id[row['head']]
            r_id = self.relation2id[row['type']]
            t_id = self.entity2id[row['tail']]
            
            edge_list.append([h_id, t_id])
            edge_type.append(r_id)
        
        # Add self-loops
        self_loop_rel_id = len(self.relations)
        for i in range(len(self.entities)):
            edge_list.append([i, i])
            edge_type.append(self_loop_rel_id)
        
        # Add inverse relations
        num_direct_relations = len(self.relations) + 1
        inverse_edge_list = []
        inverse_edge_type = []
        
        for i, (h_id, t_id) in enumerate(edge_list):
            if edge_type[i] != self_loop_rel_id:
                inverse_edge_list.append([t_id, h_id])
                inverse_edge_type.append(edge_type[i] + num_direct_relations)
        
        # Combine all edges
        all_edges = edge_list + inverse_edge_list
        all_edge_types = edge_type + inverse_edge_type
        
        edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        edge_type_tensor = torch.tensor(all_edge_types, dtype=torch.long)
        
        data = Data(
            edge_index=edge_index,
            edge_type=edge_type_tensor,
            num_nodes=len(self.entities)
        )
        data.num_relations = num_direct_relations * 2
        
        return data
    
    def _initialize_ultra(self):
        """Initialize ULTRA model"""
        print("Initializing ULTRA model...")
        
        # Build relation graph
        kg_data = build_relation_graph(self.kg_data)
        
        # Create config
        config = edict({
            'model': {
                'relation_graph': {
                    'class': 'RelNBFNet',
                    'input_dim': 64,
                    'hidden_dims': [64, 64, 64, 64, 64, 64],
                    'message_func': 'distmult',
                    'aggregate_func': 'sum',
                    'short_cut': True,
                    'layer_norm': True
                },
                'entity_graph': {
                    'class': 'EntityNBFNet',
                    'input_dim': 64,
                    'hidden_dims': [64, 64, 64, 64, 64, 64],
                    'message_func': 'distmult',
                    'aggregate_func': 'sum',
                    'short_cut': True,
                    'layer_norm': True
                }
            }
        })
        
        # Initialize model
        model = Ultra(
            rel_model_cfg=config.model.relation_graph,
            entity_model_cfg=config.model.entity_graph
        )
        
        # Load checkpoint
        checkpoint_path = "ultra/ckpts/ultra_4g.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            print("✓ ULTRA checkpoint loaded")
        else:
            print("⚠️ No checkpoint found, using random weights")
        
        model.eval()
        return model
    
    def get_llm_reranking(self, head, relation, candidates, ultra_scores):
        """Get LLM-based reranking of ULTRA predictions"""
        
        # Get entity context
        head_facts = self.entity_facts.get(head, [])[:5]
        context_lines = []
        for rel, tail in head_facts:
            if rel != relation:
                context_lines.append(f"- {head} {rel} {tail}")
        
        context = "\n".join(context_lines) if context_lines else f"Entity: {head}"
        
        # Format top ULTRA predictions
        top_candidates = candidates[:8]  # Limit for LLM focus
        candidates_text = "\n".join([f"{i+1}. {cand}" for i, cand in enumerate(top_candidates)])
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a NetHack expert. Rerank these candidates for the query based on NetHack knowledge.<|eot_id|><|start_header_id|>user<|end_header_id|>

Entity facts:
{context}

Query: {head} {relation} ?

Candidates (ranked by AI model):
{candidates_text}

Rerank 1-{len(top_candidates)} based on NetHack accuracy (most likely first):<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        try:
            response = requests.post(
                f"{self.vllm_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.1,
                    "stop": ["<|eot_id|>"]
                },
                timeout=8
            )
            
            if response.status_code == 200:
                reranking_text = response.json()['choices'][0]['text'].strip()
                
                # Parse reranking
                llm_ranking = []
                lines = reranking_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and line[0].isdigit():
                        try:
                            num = int(line[0]) - 1
                            if 0 <= num < len(top_candidates):
                                candidate = top_candidates[num]
                                if candidate not in llm_ranking:
                                    llm_ranking.append(candidate)
                        except:
                            continue
                
                # Add any missing candidates
                for candidate in top_candidates:
                    if candidate not in llm_ranking:
                        llm_ranking.append(candidate)
                
                return llm_ranking
            
        except Exception as e:
            print(f"LLM reranking error: {e}")
        
        return candidates  # Fallback to ULTRA ranking
    
    def predict_with_ultra_llm(self, head, relation, max_candidates=20):
        """Predict using ULTRA + LLM enhancement"""
        
        # Get head and relation IDs
        if head not in self.entity2id or relation not in self.relation2id:
            return []
        
        head_id = self.entity2id[head]
        rel_id = self.relation2id[relation]
        
        # Get all possible tail candidates
        all_tail_ids = list(range(len(self.entities)))
        
        # Create query batch for ULTRA
        batch_size = min(len(all_tail_ids), 1000)  # Limit for memory
        tail_batch = all_tail_ids[:batch_size]
        
        queries = torch.tensor([[head_id, rel_id, t_id] for t_id in tail_batch], dtype=torch.long)
        
        # Get ULTRA predictions
        with torch.no_grad():
            try:
                scores = self.ultra_model(self.kg_data, queries)
                scores = scores.cpu().numpy()
            except Exception as e:
                print(f"ULTRA prediction error: {e}")
                return []
        
        # Sort by ULTRA scores
        scored_tails = [(tail_batch[i], scores[i]) for i in range(len(tail_batch))]
        scored_tails.sort(key=lambda x: x[1], reverse=True)
        
        # Get top candidates
        top_tail_ids = [t_id for t_id, score in scored_tails[:max_candidates]]
        top_candidates = [self.id2entity[t_id] for t_id in top_tail_ids]
        ultra_scores = [score for t_id, score in scored_tails[:max_candidates]]
        
        # LLM reranking
        if len(top_candidates) > 1:
            reranked_candidates = self.get_llm_reranking(head, relation, top_candidates, ultra_scores)
            return reranked_candidates
        
        return top_candidates
    
    def evaluate_ultra_llm(self, test_queries=None, num_queries=12):
        """Evaluate ULTRA + LLM system"""
        
        if test_queries is None:
            # Create test queries from existing facts
            test_queries = []
            entity_counts = defaultdict(int)
            
            for entity in self.entities:
                facts = self.entity_facts.get(entity, [])
                entity_counts[entity] = len(facts)
            
            # Sort by number of facts
            sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Create diverse test cases
            used_relations = set()
            for entity, count in sorted_entities:
                if len(test_queries) >= num_queries:
                    break
                
                facts = self.entity_facts.get(entity, [])
                for relation, tail in facts:
                    if relation not in used_relations and len(test_queries) < num_queries:
                        # Check if relation has multiple values
                        relation_facts = [t for h, r_facts in self.entity_facts.items() 
                                        for r, t in r_facts if r == relation]
                        if len(set(relation_facts)) >= 3:
                            test_queries.append((entity, relation, tail))
                            used_relations.add(relation)
                            break
        
        print(f"\n🎯 Evaluating ULTRA+LLM on {len(test_queries)} queries...")
        
        mrr_scores = []
        hits_at_1 = []
        hits_at_3 = []
        hits_at_10 = []
        
        for i, (head, relation, target) in enumerate(test_queries):
            print(f"\nQuery {i+1}: {head} {relation} ?")
            print(f"Target: {target}")
            
            # Get predictions
            predictions = self.predict_with_ultra_llm(head, relation, max_candidates=15)
            
            if not predictions:
                print("No predictions generated")
                continue
            
            print(f"Top 3: {predictions[:3]}")
            
            # Calculate metrics
            if target in predictions:
                rank = predictions.index(target) + 1
                mrr = 1.0 / rank
                
                mrr_scores.append(mrr)
                hits_at_1.append(1 if rank == 1 else 0)
                hits_at_3.append(1 if rank <= 3 else 0)
                hits_at_10.append(1 if rank <= 10 else 0)
                
                print(f"✅ Rank: {rank}, MRR: {mrr:.3f}")
            else:
                print(f"❌ Target not in predictions")
                mrr_scores.append(0.0)
                hits_at_1.append(0)
                hits_at_3.append(0)
                hits_at_10.append(0)
        
        if mrr_scores:
            avg_mrr = np.mean(mrr_scores)
            print(f"\n🎯 ULTRA+LLM Results:")
            print(f"   MRR: {avg_mrr:.3f}")
            print(f"   Hits@1: {np.mean(hits_at_1):.3f}")
            print(f"   Hits@3: {np.mean(hits_at_3):.3f}")
            print(f"   Hits@10: {np.mean(hits_at_10):.3f}")
            
            return avg_mrr
        
        return 0.0

def main():
    print("🚀 Real ULTRA + LLM Integration")
    print("="*50)
    
    try:
        system = RealULTRAWithLLM()
        final_mrr = system.evaluate_ultra_llm(num_queries=10)
        
        print(f"\n📊 Final Results:")
        print(f"   ULTRA+LLM MRR: {final_mrr:.3f}")
        print(f"   Pure ULTRA: 0.603")
        print(f"   Target: >0.65")
        
        if final_mrr > 0.65:
            print("🎉 SUCCESS! Real ULTRA+LLM beats target!")
        elif final_mrr > 0.603:
            print("📈 Improvement over pure ULTRA!")
        else:
            print("⚖️ Comparable performance")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()