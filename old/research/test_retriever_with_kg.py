#!/usr/bin/env python3
"""
Test retriever-based QA with the game-specific KG
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRetrieverQA:
    def __init__(self, kg_path: str):
        print("Initializing retriever-based QA...")
        
        # Load KG
        self.df = pd.read_csv(kg_path, sep='\t')
        print(f"Loaded {len(self.df)} triples")
        
        # Initialize encoder
        print("Loading sentence encoder...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build entity and relation lists
        self.entities = list(set(self.df['head'].tolist() + self.df['tail'].tolist()))
        self.relations = list(self.df['type'].unique())
        
        print(f"Found {len(self.entities)} unique entities, {len(self.relations)} relation types")
        
        # Pre-compute embeddings
        self._compute_embeddings()
        
    def _compute_embeddings(self):
        """Pre-compute embeddings for entities and triples"""
        print("Computing entity embeddings...")
        
        # Entity embeddings
        entity_texts = [str(e) for e in self.entities]
        self.entity_embeddings = self.encoder.encode(entity_texts, show_progress_bar=True)
        
        # Triple embeddings (for fact retrieval)
        print("Computing triple embeddings...")
        triple_texts = []
        for _, row in self.df.iterrows():
            # Create natural language representation of triple
            triple_text = f"{row['head']} {row['type'].replace('_', ' ')} {row['tail']}"
            triple_texts.append(triple_text)
        
        self.triple_embeddings = self.encoder.encode(triple_texts, show_progress_bar=True)
        self.triple_texts = triple_texts
    
    def find_entities(self, query: str, top_k: int = 5):
        """Find entities mentioned in query using semantic similarity"""
        query_embedding = self.encoder.encode([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.entity_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Threshold
                results.append((self.entities[idx], similarities[idx]))
        
        return results
    
    def retrieve_facts(self, query: str, top_k: int = 5):
        """Retrieve relevant facts using semantic similarity"""
        query_embedding = self.encoder.encode([query])
        
        # Compute similarities with triples
        similarities = cosine_similarity(query_embedding, self.triple_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.4:  # Threshold
                results.append((self.triple_texts[idx], similarities[idx], idx))
        
        return results
    
    def answer_question(self, question: str):
        """Answer question using retrieval"""
        print(f"\nQuestion: {question}")
        
        # Find relevant entities
        entities = self.find_entities(question, top_k=3)
        print(f"\nRelevant entities found:")
        for entity, score in entities:
            print(f"  - {entity} (score: {score:.3f})")
        
        # Retrieve relevant facts
        facts = self.retrieve_facts(question, top_k=5)
        print(f"\nRelevant facts retrieved:")
        for fact, score, idx in facts:
            print(f"  - {fact} (score: {score:.3f})")
        
        # Generate answer based on retrieved facts
        if not facts:
            return "I couldn't find relevant information to answer this question."
        
        # Analyze question type
        question_lower = question.lower()
        
        if 'what is' in question_lower or 'tell me about' in question_lower:
            # Definition question - gather all facts about entity
            if entities:
                target_entity = entities[0][0]
                entity_facts = []
                
                for fact, score, idx in facts:
                    if str(target_entity) in fact:
                        entity_facts.append(fact)
                
                if entity_facts:
                    return f"{target_entity}: " + ". ".join(entity_facts[:3]) + "."
        
        elif 'how much damage' in question_lower or 'damage' in question_lower:
            # Damage question
            for fact, score, idx in facts:
                if 'attack_damage' in fact or 'damage' in fact:
                    return fact.replace('attack_damage', 'does').replace('_', ' ') + " damage."
        
        elif 'resistance' in question_lower:
            # Resistance question
            resistance_facts = []
            for fact, score, idx in facts:
                if 'resistance' in fact:
                    resistance_facts.append(fact.replace('_', ' '))
            
            if resistance_facts:
                return ". ".join(resistance_facts) + "."
        
        elif any(word in question_lower for word in ['armor class', 'ac', 'armor']):
            # AC question
            for fact, score, idx in facts:
                if 'armor_class' in fact:
                    return fact.replace('armor_class', 'has armor class').replace('_', ' ') + "."
        
        elif 'weight' in question_lower or 'weigh' in question_lower:
            # Weight question
            for fact, score, idx in facts:
                if 'weight' in fact:
                    return fact.replace('weight', 'weighs').replace('_', ' ') + " units."
        
        # Default: return top facts
        answer_facts = [fact.replace('_', ' ') for fact, _, _ in facts[:2]]
        return ". ".join(answer_facts) + "."

# Test the system
def main():
    qa = SimpleRetrieverQA('data/nethack_kg_final_100.tsv')
    
    print("\n" + "="*60)
    print("TESTING RETRIEVER-BASED QA")
    print("="*60)
    
    test_questions = [
        "What is soldier ant?",
        "How much damage does soldier ant do?",
        "What resistance does soldier ant have?",
        "Tell me about Medusa",
        "What is the armor class of Wumpus?",
        "How much does a Wumpus weigh?",
        "What attacks does Medusa have?",
        "What level is soldier ant?",
        "Is soldier ant poisonous?"
    ]
    
    for question in test_questions:
        answer = qa.answer_question(question)
        print(f"\nAnswer: {answer}")
        print("-" * 40)
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("="*60)
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit']:
            break
        
        if question:
            answer = qa.answer_question(question)
            print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()