#!/usr/bin/env python3
"""
Analyze retriever-based QA performance with game KG
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RetrieverQAAnalyzer:
    def __init__(self, kg_path: str):
        print("Initializing QA analyzer...")
        
        # Load KG
        self.df = pd.read_csv(kg_path, sep='\t')
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build entity and relation lists
        self.entities = list(set(self.df['head'].tolist() + self.df['tail'].tolist()))
        self.relations = list(self.df['type'].unique())
        
        # Pre-compute embeddings
        print("Computing embeddings...")
        self.entity_embeddings = self.encoder.encode([str(e) for e in self.entities])
        
        # Triple embeddings
        self.triple_texts = []
        for _, row in self.df.iterrows():
            triple_text = f"{row['head']} {row['type'].replace('_', ' ')} {row['tail']}"
            self.triple_texts.append(triple_text)
        self.triple_embeddings = self.encoder.encode(self.triple_texts)
        
    def analyze_coverage(self):
        """Analyze what the KG covers"""
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH COVERAGE ANALYSIS")
        print("="*60)
        
        # Entity types
        print("\nEntity Analysis:")
        entity_types = {
            'monsters': [],
            'items': [],
            'properties': [],
            'numbers': []
        }
        
        for entity in self.entities:
            entity_str = str(entity).lower()
            if any(monster in entity_str for monster in ['ant', 'medusa', 'wumpus']):
                entity_types['monsters'].append(entity)
            elif 'resistance' in entity_str or 'poison' in entity_str:
                entity_types['properties'].append(entity)
            elif entity_str.isdigit() or 'd' in entity_str:
                entity_types['numbers'].append(entity)
            else:
                entity_types['items'].append(entity)
        
        for etype, entities in entity_types.items():
            print(f"  {etype}: {len(entities)} entities")
            if entities:
                print(f"    Examples: {', '.join(str(e) for e in entities[:5])}")
        
        # Relation coverage
        print("\nRelation Coverage:")
        relation_counts = self.df['type'].value_counts()
        for rel, count in relation_counts.items():
            print(f"  {rel}: {count} facts")
        
        # Monster coverage
        print("\nMonster Information:")
        monsters = ['soldier ant', 'Medusa', 'Wumpus']
        for monster in monsters:
            facts = self.df[self.df['head'] == monster]
            if not facts.empty:
                print(f"\n  {monster}:")
                for _, row in facts.iterrows():
                    print(f"    - {row['type']}: {row['tail']}")
    
    def test_question_types(self):
        """Test different question types"""
        print("\n" + "="*60)
        print("QUESTION TYPE ANALYSIS")
        print("="*60)
        
        question_types = {
            'Property Questions': [
                ("What is the AC of soldier ant?", "armor_class"),
                ("How difficult is Medusa?", "difficulty"),
                ("What level is Wumpus?", "base_level"),
                ("How fast is soldier ant?", "speed")
            ],
            'Combat Questions': [
                ("What damage does soldier ant do?", "attack_damage"),
                ("What attacks does Medusa have?", "attack_type"),
                ("What type of damage does Wumpus do?", "damage_type")
            ],
            'Resistance Questions': [
                ("What resistance does soldier ant have?", "has_resistance"),
                ("What resistance does soldier ant grant?", "grants_resistance")
            ]
        }
        
        for qtype, questions in question_types.items():
            print(f"\n{qtype}:")
            
            for question, expected_relation in questions:
                # Get question embedding
                q_emb = self.encoder.encode([question])
                
                # Find most similar triples
                similarities = cosine_similarity(q_emb, self.triple_embeddings)[0]
                top_idx = np.argmax(similarities)
                
                best_triple = self.triple_texts[top_idx]
                best_score = similarities[top_idx]
                
                # Check if correct relation was found
                found_correct = expected_relation in best_triple
                
                print(f"\n  Q: {question}")
                print(f"  Best match: {best_triple} (score: {best_score:.3f})")
                print(f"  Found correct relation: {'✓' if found_correct else '✗'}")
    
    def analyze_similarity_patterns(self):
        """Analyze similarity patterns in embeddings"""
        print("\n" + "="*60)
        print("SIMILARITY PATTERN ANALYSIS")
        print("="*60)
        
        # Analyze which entities are most similar
        print("\nMost similar entity pairs:")
        entity_names = [str(e) for e in self.entities]
        
        # Compute all pairwise similarities
        similarities = cosine_similarity(self.entity_embeddings)
        
        # Find top similar pairs (excluding self)
        pairs = []
        for i in range(len(self.entities)):
            for j in range(i+1, len(self.entities)):
                pairs.append((i, j, similarities[i][j]))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, sim in pairs[:10]:
            print(f"  {entity_names[i]} <-> {entity_names[j]}: {sim:.3f}")
        
        # Analyze relation embeddings
        print("\nRelation embedding analysis:")
        relation_texts = [r.replace('_', ' ') for r in self.relations]
        relation_embeddings = self.encoder.encode(relation_texts)
        
        # Find similar relations
        rel_similarities = cosine_similarity(relation_embeddings)
        
        print("\nMost similar relation pairs:")
        rel_pairs = []
        for i in range(len(self.relations)):
            for j in range(i+1, len(self.relations)):
                rel_pairs.append((i, j, rel_similarities[i][j]))
        
        rel_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, sim in rel_pairs[:5]:
            print(f"  {self.relations[i]} <-> {self.relations[j]}: {sim:.3f}")
    
    def test_retrieval_accuracy(self):
        """Test retrieval accuracy for known facts"""
        print("\n" + "="*60)
        print("RETRIEVAL ACCURACY TEST")
        print("="*60)
        
        # Test if we can retrieve exact facts
        test_facts = [
            ("soldier ant has AC 3", "soldier ant armor_class 3"),
            ("Medusa difficulty 25", "Medusa difficulty 25"),
            ("Wumpus weighs 2500", "Wumpus weight 2500"),
            ("soldier ant stings for 3d4", "soldier ant attack_damage 3d4")
        ]
        
        correct = 0
        for query, expected in test_facts:
            q_emb = self.encoder.encode([query])
            similarities = cosine_similarity(q_emb, self.triple_embeddings)[0]
            top_idx = np.argmax(similarities)
            retrieved = self.triple_texts[top_idx]
            
            is_correct = retrieved == expected
            if is_correct:
                correct += 1
            
            print(f"\nQuery: {query}")
            print(f"Expected: {expected}")
            print(f"Retrieved: {retrieved} (score: {similarities[top_idx]:.3f})")
            print(f"Correct: {'✓' if is_correct else '✗'}")
        
        accuracy = correct / len(test_facts)
        print(f"\nOverall accuracy: {accuracy:.1%}")

def main():
    analyzer = RetrieverQAAnalyzer('data/nethack_kg_final_100.tsv')
    
    # Run all analyses
    analyzer.analyze_coverage()
    analyzer.test_question_types()
    analyzer.analyze_similarity_patterns()
    analyzer.test_retrieval_accuracy()

if __name__ == "__main__":
    main()