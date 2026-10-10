#!/usr/bin/env python3
"""
NetHack QA System combining ULTRA link prediction with VLLM generation
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetHackUltraVLLM:
    def __init__(
        self, 
        kg_path: str = "data/nethack_kg_final_a_plus.tsv",
        vllm_url: str = "http://localhost:8080",
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    ):
        """Initialize NetHack QA system"""
        self.vllm_url = vllm_url
        self.model_name = model_name
        
        # Load KG
        logger.info(f"Loading KG from {kg_path}")
        self.kg = pd.read_csv(kg_path, sep='\t')
        logger.info(f"Loaded {len(self.kg)} triples")
        
        # Build indices
        self._build_indices()
        
        # Test VLLM
        self._test_vllm()
        
    def _build_indices(self):
        """Build lookup indices"""
        # Entity facts
        self.entity_facts = {}
        for _, row in self.kg.iterrows():
            head = str(row['head'])
            if head not in self.entity_facts:
                self.entity_facts[head] = []
            self.entity_facts[head].append({
                'relation': row['type'],
                'tail': row['tail'],
                'priority': row.get('priority', 5)
            })
        
        # Relation types
        self.relation_types = self.kg['type'].unique()
        
        # High priority facts
        self.high_priority = self.kg[self.kg['priority'] >= 8]
        
        logger.info(f"Indexed {len(self.entity_facts)} entities")
        
    def _test_vllm(self):
        """Test VLLM connection"""
        try:
            resp = requests.get(f"{self.vllm_url}/v1/models")
            if resp.status_code == 200:
                logger.info("✓ VLLM connected")
            else:
                logger.warning(f"VLLM status: {resp.status_code}")
        except Exception as e:
            logger.error(f"VLLM error: {e}")
    
    def predict_links(self, entity: str, relation: str = None, top_k: int = 5) -> List[Dict]:
        """
        Predict missing links for entity (simulating ULTRA functionality)
        """
        predictions = []
        
        if relation:
            # Find entities with similar relations
            similar = self.kg[self.kg['type'] == relation]
            
            # Get common tail values for this relation
            tail_counts = similar['tail'].value_counts()
            
            for tail, count in tail_counts.head(top_k).items():
                # Check if this link already exists
                exists = ((self.kg['head'] == entity) & 
                         (self.kg['type'] == relation) & 
                         (self.kg['tail'] == tail)).any()
                
                if not exists:
                    predictions.append({
                        'head': entity,
                        'relation': relation,
                        'tail': tail,
                        'score': count / len(similar),  # Simple frequency-based score
                        'confidence': 'predicted'
                    })
        else:
            # Predict most likely relations for this entity
            if entity in self.entity_facts:
                existing_rels = [f['relation'] for f in self.entity_facts[entity]]
                
                # Find similar entities
                similar_entities = []
                for other_entity, facts in self.entity_facts.items():
                    if other_entity != entity:
                        other_rels = [f['relation'] for f in facts]
                        overlap = len(set(existing_rels) & set(other_rels))
                        if overlap > 0:
                            similar_entities.append((other_entity, overlap))
                
                # Sort by similarity
                similar_entities.sort(key=lambda x: x[1], reverse=True)
                
                # Get relations from similar entities
                rel_scores = {}
                for similar_entity, _ in similar_entities[:10]:
                    for fact in self.entity_facts[similar_entity]:
                        if fact['relation'] not in existing_rels:
                            if fact['relation'] not in rel_scores:
                                rel_scores[fact['relation']] = 0
                            rel_scores[fact['relation']] += 1
                
                # Top predicted relations
                for rel, score in sorted(rel_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                    predictions.append({
                        'head': entity,
                        'relation': rel,
                        'tail': '?',
                        'score': score / len(similar_entities[:10]),
                        'confidence': 'predicted'
                    })
        
        return predictions
    
    def get_facts(self, entity: str, include_predictions: bool = True) -> List[Dict]:
        """Get facts about entity including predictions"""
        facts = []
        
        # Get existing facts
        if entity in self.entity_facts:
            for fact in self.entity_facts[entity]:
                facts.append({
                    'head': entity,
                    'relation': fact['relation'],
                    'tail': fact['tail'],
                    'priority': fact['priority'],
                    'confidence': 'known'
                })
        
        # Add predictions
        if include_predictions:
            predictions = self.predict_links(entity, top_k=3)
            facts.extend(predictions)
        
        # Sort by priority/score
        facts.sort(key=lambda x: x.get('priority', x.get('score', 0)), reverse=True)
        
        return facts
    
    def generate_answer(self, question: str, context_facts: List[Dict]) -> str:
        """Generate natural language answer using VLLM"""
        
        # Format facts
        fact_lines = []
        known_facts = []
        predicted_facts = []
        
        for fact in context_facts[:7]:  # Use top 7 facts
            fact_str = f"{fact['head']} {fact['relation']} {fact['tail']}"
            if fact.get('confidence') == 'known':
                known_facts.append(fact_str)
            else:
                predicted_facts.append(fact_str + " (predicted)")
        
        # Build context
        context_parts = []
        if known_facts:
            context_parts.append("Known facts:\n" + "\n".join(f"- {f}" for f in known_facts))
        if predicted_facts:
            context_parts.append("Predicted facts:\n" + "\n".join(f"- {f}" for f in predicted_facts))
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a NetHack expert assistant. Use the provided facts to answer questions accurately. When using predicted facts, mention they are predictions.<|eot_id|><|start_header_id|>user<|end_header_id|>

{context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Call VLLM
        try:
            response = requests.post(
                f"{self.vllm_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": 150,
                    "temperature": 0.7,
                    "stop": ["<|eot_id|>", "\n\n", "Question:"]
                }
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['text'].strip()
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {e}"
    
    def answer_question(self, question: str) -> Dict:
        """Complete Q&A pipeline"""
        # Extract entity from question (simple approach)
        entities = []
        for entity in self.entity_facts.keys():
            if entity.lower() in question.lower():
                entities.append(entity)
        
        if not entities:
            # Try to find entities by checking tails too
            for _, row in self.kg.iterrows():
                if str(row['tail']).lower() in question.lower():
                    entities.append(str(row['head']))
        
        # Use first matched entity or a default
        entity = entities[0] if entities else "NetHack"
        
        # Get facts and predictions
        facts = self.get_facts(entity, include_predictions=True)
        
        # Generate answer
        answer = self.generate_answer(question, facts)
        
        return {
            'question': question,
            'entity': entity,
            'facts_used': facts[:5],
            'answer': answer
        }
    
    def interactive(self):
        """Interactive Q&A session"""
        print("\n🎮 NetHack ULTRA+VLLM Q&A System")
        print("="*60)
        print("Ask questions about NetHack! I'll use known facts and predictions.")
        print("Type 'quit' to exit.\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Farewell, adventurer!")
                break
            
            if not question:
                continue
            
            # Get answer
            result = self.answer_question(question)
            
            # Display
            print(f"\n📍 Entity: {result['entity']}")
            
            if result['facts_used']:
                print("\n📚 Facts used:")
                for fact in result['facts_used'][:3]:
                    conf = "✓" if fact.get('confidence') == 'known' else "?"
                    print(f"   {conf} {fact['head']} → {fact['tail']}")
            
            print(f"\n🤖 Answer: {result['answer']}")
            print("-"*60)


def main():
    # Initialize system
    qa = NetHackUltraVLLM()
    
    # Test questions
    print("\n🧪 Testing NetHack ULTRA+VLLM System")
    print("="*60)
    
    test_questions = [
        "What armor class does Medusa have?",
        "How can I defeat a soldier ant?",
        "What resistances might a dragon have?",
        "What items should I use against a lich?",
        "How dangerous is the Wizard of Yendor?"
    ]
    
    for q in test_questions:
        print(f"\n❓ {q}")
        result = qa.answer_question(q)
        print(f"📍 Entity: {result['entity']}")
        print(f"💬 {result['answer']}")
    
    # Interactive mode
    print("\n" + "="*60)
    qa.interactive()


if __name__ == "__main__":
    main()