#!/usr/bin/env python3
"""
Improved retriever-based QA with better query understanding
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class ImprovedRetrieverQA:
    def __init__(self, kg_path: str):
        print("Initializing improved retriever QA...")
        
        # Load KG
        self.df = pd.read_csv(kg_path, sep='\t')
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Query expansion mappings
        self.query_expansions = {
            'ac': ['armor class', 'AC', 'armor_class'],
            'damage': ['attack_damage', 'damage', 'does damage'],
            'weight': ['weighs', 'weight'],
            'fast': ['speed', 'fast', 'movement'],
            'level': ['base_level', 'level', 'dungeon level'],
            'difficult': ['difficulty', 'hard', 'challenging'],
            'resist': ['resistance', 'resist', 'immune'],
            'attack': ['attack_type', 'attacks with', 'uses']
        }
        
        # Relation aliases for better matching
        self.relation_aliases = {
            'armor_class': 'has armor class',
            'attack_damage': 'does damage',
            'weight': 'weighs',
            'speed': 'has speed',
            'base_level': 'appears on level',
            'difficulty': 'has difficulty',
            'attack_type': 'attacks with',
            'damage_type': 'deals damage type',
            'has_resistance': 'resists',
            'grants_resistance': 'grants resistance to'
        }
        
        # Build indexes
        self._build_indexes()
    
    def _build_indexes(self):
        """Build various indexes for efficient retrieval"""
        print("Building indexes...")
        
        # Entity index
        self.entities = list(set(self.df['head'].tolist() + self.df['tail'].tolist()))
        self.entity_to_facts = {}
        
        for entity in self.entities:
            facts = []
            # As head
            for _, row in self.df[self.df['head'] == entity].iterrows():
                facts.append({
                    'triple': (row['head'], row['type'], row['tail']),
                    'text': f"{row['head']} {self.relation_aliases.get(row['type'], row['type'])} {row['tail']}",
                    'role': 'head'
                })
            # As tail
            for _, row in self.df[self.df['tail'] == entity].iterrows():
                facts.append({
                    'triple': (row['head'], row['type'], row['tail']),
                    'text': f"{row['head']} {self.relation_aliases.get(row['type'], row['type'])} {row['tail']}",
                    'role': 'tail'
                })
            self.entity_to_facts[entity] = facts
        
        # Create embeddings for all fact texts
        all_fact_texts = []
        self.fact_index = []
        
        for entity, facts in self.entity_to_facts.items():
            for fact in facts:
                all_fact_texts.append(fact['text'])
                self.fact_index.append(fact)
        
        if all_fact_texts:
            self.fact_embeddings = self.encoder.encode(all_fact_texts)
        else:
            self.fact_embeddings = np.array([])
        
        # Entity embeddings
        self.entity_embeddings = self.encoder.encode([str(e) for e in self.entities])
    
    def expand_query(self, query: str):
        """Expand query with synonyms and variations"""
        expanded = [query]
        query_lower = query.lower()
        
        # Add expansions
        for key, expansions in self.query_expansions.items():
            if key in query_lower:
                for expansion in expansions:
                    expanded_query = query_lower.replace(key, expansion)
                    expanded.append(expanded_query)
        
        return expanded
    
    def find_entities_in_query(self, query: str):
        """Find entities mentioned in query"""
        query_lower = query.lower()
        found_entities = []
        
        # Exact match first
        for entity in self.entities:
            entity_str = str(entity).lower()
            if entity_str in query_lower:
                found_entities.append((entity, 1.0))
        
        # If no exact match, use embeddings
        if not found_entities:
            query_emb = self.encoder.encode([query])
            similarities = cosine_similarity(query_emb, self.entity_embeddings)[0]
            
            top_indices = np.argsort(similarities)[::-1][:3]
            for idx in top_indices:
                if similarities[idx] > 0.5:
                    found_entities.append((self.entities[idx], similarities[idx]))
        
        return found_entities
    
    def answer_question(self, question: str, verbose: bool = True):
        """Answer question with improved retrieval"""
        if verbose:
            print(f"\nQuestion: {question}")
        
        # Find entities
        entities = self.find_entities_in_query(question)
        if verbose and entities:
            print(f"Entities found: {[(e, f'{s:.2f}') for e, s in entities]}")
        
        # Expand query
        expanded_queries = self.expand_query(question)
        
        # Retrieve facts for all expanded queries
        all_retrieved = []
        
        for expanded_q in expanded_queries:
            query_emb = self.encoder.encode([expanded_q])
            
            if len(self.fact_embeddings) > 0:
                similarities = cosine_similarity(query_emb, self.fact_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:10]
                
                for idx in top_indices:
                    if similarities[idx] > 0.4:
                        fact = self.fact_index[idx]
                        all_retrieved.append((fact, similarities[idx], expanded_q))
        
        # Deduplicate and sort
        seen_facts = set()
        unique_facts = []
        for fact, score, source_query in sorted(all_retrieved, key=lambda x: x[1], reverse=True):
            fact_key = str(fact['triple'])
            if fact_key not in seen_facts:
                seen_facts.add(fact_key)
                unique_facts.append((fact, score, source_query))
        
        if verbose and unique_facts:
            print(f"\nTop retrieved facts:")
            for fact, score, source in unique_facts[:5]:
                print(f"  - {fact['text']} (score: {score:.3f})")
        
        # Generate answer based on question type
        answer = self._generate_answer(question, entities, unique_facts)
        
        return answer
    
    def _generate_answer(self, question: str, entities, facts):
        """Generate natural language answer"""
        if not facts:
            return "I don't have enough information to answer that question."
        
        question_lower = question.lower()
        
        # Pattern-based answer generation
        patterns = [
            (r'what (?:is|are) (?:the )?(.+)', self._answer_definition),
            (r'how much damage', self._answer_damage),
            (r'what.*(?:resistance|resist)', self._answer_resistance),
            (r'what.*(?:ac|armor class)', self._answer_ac),
            (r'how.*(?:difficult|hard)', self._answer_difficulty),
            (r'what.*(?:level)', self._answer_level),
            (r'how.*(?:fast|speed)', self._answer_speed),
            (r'how.*(?:heavy|weigh)', self._answer_weight),
            (r'what.*attack', self._answer_attacks),
        ]
        
        for pattern, handler in patterns:
            match = re.search(pattern, question_lower)
            if match:
                return handler(entities, facts)
        
        # Default: return top facts
        fact_texts = []
        for fact, score, _ in facts[:2]:
            text = fact['text']
            # Make more natural
            text = text.replace('_', ' ')
            fact_texts.append(text)
        
        return ". ".join(fact_texts) + "."
    
    def _answer_definition(self, entities, facts):
        """Answer 'what is X' questions"""
        if not entities:
            return "I'm not sure what you're asking about."
        
        target = entities[0][0]
        entity_facts = []
        
        for fact, score, _ in facts:
            if fact['triple'][0] == target or fact['triple'][2] == target:
                entity_facts.append(fact['text'])
        
        if entity_facts:
            return f"{target} is a creature that " + ", ".join(entity_facts[:3]).lower() + "."
        
        return f"I don't have specific information about {target}."
    
    def _answer_damage(self, entities, facts):
        """Answer damage questions"""
        for fact, score, _ in facts:
            if 'attack_damage' in str(fact['triple']):
                entity = fact['triple'][0]
                damage = fact['triple'][2]
                return f"{entity} does {damage} damage."
        
        return "I don't have damage information for that."
    
    def _answer_resistance(self, entities, facts):
        """Answer resistance questions"""
        resistances = []
        
        for fact, score, _ in facts:
            if 'resistance' in str(fact['triple']):
                resistances.append(fact['text'])
        
        if resistances:
            return ". ".join(resistances[:2]) + "."
        
        return "I don't have resistance information."
    
    def _answer_ac(self, entities, facts):
        """Answer AC questions"""
        for fact, score, _ in facts:
            if 'armor_class' in str(fact['triple']):
                entity = fact['triple'][0]
                ac = fact['triple'][2]
                return f"{entity} has armor class {ac}."
        
        return "I don't have armor class information."
    
    def _answer_difficulty(self, entities, facts):
        """Answer difficulty questions"""
        for fact, score, _ in facts:
            if 'difficulty' in str(fact['triple']):
                entity = fact['triple'][0]
                diff = fact['triple'][2]
                return f"{entity} has difficulty {diff}."
        
        return "I don't have difficulty information."
    
    def _answer_level(self, entities, facts):
        """Answer level questions"""
        for fact, score, _ in facts:
            if 'base_level' in str(fact['triple']):
                entity = fact['triple'][0]
                level = fact['triple'][2]
                return f"{entity} has base level {level}."
        
        return "I don't have level information."
    
    def _answer_speed(self, entities, facts):
        """Answer speed questions"""
        for fact, score, _ in facts:
            if 'speed' in str(fact['triple']):
                entity = fact['triple'][0]
                speed = fact['triple'][2]
                return f"{entity} has speed {speed}."
        
        return "I don't have speed information."
    
    def _answer_weight(self, entities, facts):
        """Answer weight questions"""
        for fact, score, _ in facts:
            if 'weight' in str(fact['triple']):
                entity = fact['triple'][0]
                weight = fact['triple'][2]
                return f"{entity} weighs {weight} units."
        
        return "I don't have weight information."
    
    def _answer_attacks(self, entities, facts):
        """Answer attack questions"""
        attacks = []
        
        for fact, score, _ in facts:
            if 'attack' in str(fact['triple']):
                attacks.append(fact['text'])
        
        if attacks:
            return "Attack information: " + ". ".join(attacks[:3]) + "."
        
        return "I don't have attack information."

def main():
    qa = ImprovedRetrieverQA('data/nethack_kg_final_a_plus.tsv')
    
    print("\n" + "="*60)
    print("TESTING IMPROVED RETRIEVER QA")
    print("="*60)
    
    test_questions = [
        # Property questions
        "What is the AC of soldier ant?",
        "How difficult is Medusa?",
        "What level is Wumpus?",
        "How fast is soldier ant?",
        "How much does Wumpus weigh?",
        
        # Combat questions
        "How much damage does soldier ant do?",
        "What attacks does Medusa have?",
        
        # Resistance questions
        "What resistance does soldier ant have?",
        "What resistance does soldier ant grant?",
        
        # General questions
        "What is soldier ant?",
        "Tell me about Medusa"
    ]
    
    for question in test_questions:
        answer = qa.answer_question(question, verbose=True)
        print(f"\nAnswer: {answer}")
        print("-" * 60)

if __name__ == "__main__":
    main()