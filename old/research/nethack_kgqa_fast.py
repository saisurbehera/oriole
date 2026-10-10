"""
Fast NetHack KGQA with FAISS indexing and caching
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import re
import json
import logging
from enum import Enum
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchedEntity:
    name: str
    semantic_score: float
    exact_match: bool
    context_score: float
    importance_score: float
    final_score: float
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


@dataclass
class Answer:
    text: str
    confidence: float
    evidence_triples: List[Tuple[str, str, str]]
    reasoning_path: List[str]
    answer_type: str


class FastNetHackKGQA:
    """Optimized KGQA with FAISS indexing"""
    
    def __init__(self, kg_path: str, cache_dir: str = "data/kgqa_cache"):
        self.kg_path = kg_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if cached indexes exist
        entity_index_path = os.path.join(cache_dir, "entity_faiss.index")
        relation_index_path = os.path.join(cache_dir, "relation_faiss.index")
        metadata_path = os.path.join(cache_dir, "metadata.pkl")
        
        if all(os.path.exists(p) for p in [entity_index_path, relation_index_path, metadata_path]):
            logger.info("Loading from cache...")
            self._load_from_cache()
        else:
            logger.info("Building indexes from scratch...")
            self._build_from_scratch()
            self._save_to_cache()
    
    def _load_from_cache(self):
        """Load pre-computed indexes and data"""
        cache_files = {
            'entity_index': os.path.join(self.cache_dir, "entity_faiss.index"),
            'relation_index': os.path.join(self.cache_dir, "relation_faiss.index"),
            'metadata': os.path.join(self.cache_dir, "metadata.pkl")
        }
        
        # Load FAISS indexes
        self.entity_index = faiss.read_index(cache_files['entity_index'])
        self.relation_index = faiss.read_index(cache_files['relation_index'])
        
        # Load metadata
        with open(cache_files['metadata'], 'rb') as f:
            metadata = pickle.load(f)
            
        self.entities = metadata['entities']
        self.relations = metadata['relations']
        self.entity_list = metadata['entity_list']
        self.relation_list = metadata['relation_list']
        self.entity_triples = metadata['entity_triples']
        self.relation_triples = metadata['relation_triples']
        self.entity_importance = metadata['entity_importance']
        self.df = metadata['df']
        
        # Initialize encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info(f"Loaded {len(self.entities)} entities, {len(self.relations)} relations from cache")
    
    def _build_from_scratch(self):
        """Build indexes from KG file"""
        # Initialize encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load KG
        logger.info(f"Loading KG from {self.kg_path}")
        self.df = pd.read_csv(self.kg_path, sep='\t', header=0)
        if len(self.df.columns) == 3:
            self.df.columns = ['head', 'relation', 'tail']
        else:
            self.df.columns = ['head', 'relation', 'tail', 'source']
        
        # Build indexes
        self.entities = set()
        self.relations = set()
        self.entity_triples = defaultdict(list)
        self.relation_triples = defaultdict(list)
        
        for idx, row in self.df.iterrows():
            h, r, t = str(row['head']), str(row['relation']), str(row['tail'])
            
            self.entities.add(h)
            self.entities.add(t)
            self.relations.add(r)
            
            self.entity_triples[h].append((idx, 'head', r, t))
            self.entity_triples[t].append((idx, 'tail', r, h))
            self.relation_triples[r].append((h, t))
        
        # Convert to lists for indexing
        self.entity_list = sorted(list(self.entities))
        self.relation_list = sorted(list(self.relations))
        
        # Compute entity importance (simple degree-based)
        self.entity_importance = {}
        for entity in self.entities:
            self.entity_importance[entity] = len(self.entity_triples[entity]) / 100.0
        
        # Build FAISS indexes
        self._build_faiss_indexes()
    
    def _build_faiss_indexes(self):
        """Build FAISS indexes for entities and relations"""
        logger.info("Building FAISS indexes...")
        
        # Entity embeddings
        logger.info(f"Encoding {len(self.entity_list)} entities...")
        entity_embeddings = self.encoder.encode(self.entity_list, 
                                              batch_size=256, 
                                              show_progress_bar=False,
                                              convert_to_numpy=True)
        
        # Build entity index
        dimension = entity_embeddings.shape[1]
        self.entity_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        faiss.normalize_L2(entity_embeddings)  # Normalize for cosine similarity
        self.entity_index.add(entity_embeddings)
        
        # Relation embeddings with variations
        logger.info(f"Encoding {len(self.relation_list)} relations...")
        relation_texts = []
        for rel in self.relation_list:
            relation_texts.append(rel)
            relation_texts.append(rel.replace('_', ' '))
        
        relation_embeddings = self.encoder.encode(relation_texts,
                                                batch_size=256,
                                                show_progress_bar=False,
                                                convert_to_numpy=True)
        
        # Build relation index
        self.relation_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(relation_embeddings)
        self.relation_index.add(relation_embeddings)
    
    def _save_to_cache(self):
        """Save indexes and metadata to cache"""
        logger.info("Saving to cache...")
        
        # Save FAISS indexes
        faiss.write_index(self.entity_index, 
                         os.path.join(self.cache_dir, "entity_faiss.index"))
        faiss.write_index(self.relation_index, 
                         os.path.join(self.cache_dir, "relation_faiss.index"))
        
        # Save metadata
        metadata = {
            'entities': self.entities,
            'relations': self.relations,
            'entity_list': self.entity_list,
            'relation_list': self.relation_list,
            'entity_triples': dict(self.entity_triples),
            'relation_triples': dict(self.relation_triples),
            'entity_importance': self.entity_importance,
            'df': self.df
        }
        
        with open(os.path.join(self.cache_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
    
    def _search_entities(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Fast entity search using FAISS"""
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.entity_index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and score > 0.5:  # Threshold
                results.append((self.entity_list[idx], float(score)))
        
        return results
    
    def _search_relations(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Fast relation search using FAISS"""
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.relation_index.search(query_embedding, k * 2)  # *2 for variations
        
        seen = set()
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and score > 0.4:
                rel_idx = idx // 2  # Map back to original relation
                rel_name = self.relation_list[rel_idx]
                if rel_name not in seen:
                    seen.add(rel_name)
                    results.append((rel_name, float(score)))
        
        return results[:k]
    
    def answer(self, question: str) -> Answer:
        """Fast question answering"""
        question_lower = question.lower().strip()
        
        # Fast entity matching
        entity_matches = self._match_entities_fast(question_lower)
        relation_matches = self._search_relations(question_lower)
        
        # Debug output
        if entity_matches:
            logger.info(f"Matched entities: {[(e.name, f'{e.final_score:.2f}') for e in entity_matches[:3]]}")
        
        # Route to appropriate handler
        if question_lower.startswith('what is') or question_lower.startswith('what are'):
            return self._answer_definition_fast(question_lower, entity_matches)
        elif 'how many' in question_lower or 'count' in question_lower:
            return self._answer_count_fast(question_lower, entity_matches)
        elif any(word in question_lower for word in ['difference', 'compare', 'vs']):
            return self._answer_comparison_fast(question_lower, entity_matches)
        elif question_lower.startswith(('is ', 'are ', 'does ', 'can ')):
            return self._answer_boolean_fast(question_lower, entity_matches, relation_matches)
        else:
            return self._answer_general_fast(question_lower, entity_matches, relation_matches)
    
    def _match_entities_fast(self, question: str) -> List[MatchedEntity]:
        """Fast entity matching with context"""
        matches = []
        candidates = set()
        
        # Check exact matches first (case insensitive, but only for entities > 2 chars)
        # First try exact case match for proper nouns
        for entity in ['Amulet of Yendor', 'NetHack', 'Sokoban', 'Gehennom', 'SLASH\'EM', 
                      'Wizard of Yendor', 'Excalibur', 'Stormbringer']:
            if entity.lower() in question.lower():
                matches.append(MatchedEntity(
                    name=entity,
                    semantic_score=1.0,
                    exact_match=True,
                    context_score=1.0,
                    importance_score=self.entity_importance.get(entity, 0.5),
                    final_score=1.0
                ))
                candidates.add(entity)
        
        # Then check other entities
        for entity in self.entities:
            if entity not in candidates and len(entity) > 2 and entity.lower() in question.lower():
                # Check word boundaries to avoid partial matches
                import re
                pattern = r'\b' + re.escape(entity.lower()) + r'\b'
                if re.search(pattern, question.lower()):
                    matches.append(MatchedEntity(
                        name=entity,
                        semantic_score=1.0,
                        exact_match=True,
                        context_score=1.0,
                        importance_score=self.entity_importance.get(entity, 0.5),
                        final_score=1.0
                    ))
                    candidates.add(entity)
        
        # Extract multi-word phrases for semantic search
        words = question.split()
        phrases_to_search = []
        
        # Add the full question
        phrases_to_search.append(question)
        
        # Add 2-3 word combinations
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words)+1)):
                phrase = ' '.join(words[i:j])
                if len(phrase) > 3:  # Skip very short phrases
                    phrases_to_search.append(phrase)
        
        # Semantic search on meaningful phrases
        for phrase in phrases_to_search[:5]:  # Limit searches
            semantic_matches = self._search_entities(phrase, k=3)
            for entity, score in semantic_matches:
                if entity not in candidates and score > 0.7 and len(entity) > 2:
                    matches.append(MatchedEntity(
                        name=entity,
                        semantic_score=score,
                        exact_match=False,
                        context_score=0.8,
                        importance_score=self.entity_importance.get(entity, 0.5),
                        final_score=score * 0.9
                    ))
                    candidates.add(entity)
        
        # Sort by final score and importance
        matches.sort(key=lambda x: (x.final_score, x.importance_score), reverse=True)
        return matches[:10]
    
    def _answer_definition_fast(self, question: str, entities: List[MatchedEntity]) -> Answer:
        """Fast definition answering"""
        if not entities:
            return Answer(
                text="I couldn't identify what you're asking about.",
                confidence=0.0,
                evidence_triples=[],
                reasoning_path=[],
                answer_type='definition'
            )
        
        target = entities[0]
        facts = []
        evidence = []
        
        # Get facts about entity
        for idx, role, relation, other in self.entity_triples.get(target.name, []):
            facts.append((relation, role, other))
            if role == 'head':
                evidence.append((target.name, relation, other))
            else:
                evidence.append((other, relation, target.name))
        
        if not facts:
            return Answer(
                text=f"I don't have information about {target.name}.",
                confidence=0.3,
                evidence_triples=[],
                reasoning_path=[],
                answer_type='definition'
            )
        
        # Build answer
        answer_parts = []
        
        # Type info
        for rel, role, other in facts:
            if rel in ['instance_of', 'subclass_of'] and role == 'head':
                answer_parts.append(f"{target.name} is a {other}")
                break
        
        # Location info
        for rel, role, other in facts:
            if rel == 'part_of' and role == 'head':
                answer_parts.append(f"It is part of {other}")
                break
        
        # Key properties
        for rel, role, other in facts[:5]:
            if rel not in ['instance_of', 'subclass_of', 'part_of'] and role == 'head':
                answer_parts.append(f"It {rel} {other}")
                if len(answer_parts) >= 3:
                    break
        
        answer_text = '. '.join(answer_parts[:3]) + '.'
        confidence = min(1.0, 0.5 + 0.1 * len(facts)) * target.final_score
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=evidence[:5],
            reasoning_path=[f"Found {len(facts)} facts about {target.name}"],
            answer_type='definition'
        )
    
    def _answer_boolean_fast(self, question: str, entities: List[MatchedEntity], 
                            relations: List[Tuple[str, float]]) -> Answer:
        """Fast boolean answering"""
        if len(entities) >= 2:
            e1, e2 = entities[0], entities[1]
            
            # Check connection
            for _, role, rel, other in self.entity_triples.get(e1.name, []):
                if other == e2.name:
                    answer_text = f"Yes, {e1.name} {rel} {e2.name}."
                    return Answer(
                        text=answer_text,
                        confidence=0.9 * min(e1.final_score, e2.final_score),
                        evidence_triples=[(e1.name, rel, e2.name)],
                        reasoning_path=["Found direct connection"],
                        answer_type='boolean'
                    )
            
            return Answer(
                text=f"No, I don't have evidence that {e1.name} is directly related to {e2.name}.",
                confidence=0.7 * min(e1.final_score, e2.final_score),
                evidence_triples=[],
                reasoning_path=["No direct connection found"],
                answer_type='boolean'
            )
        
        return Answer(
            text="I need more information to answer this yes/no question.",
            confidence=0.0,
            evidence_triples=[],
            reasoning_path=[],
            answer_type='boolean'
        )
    
    def _answer_count_fast(self, question: str, entities: List[MatchedEntity]) -> Answer:
        """Fast counting"""
        # Simple implementation - count variants
        if 'variant' in question and any(e.name == 'NetHack' for e in entities):
            count = sum(1 for e in self.entities if 'NetHack' in e and e != 'NetHack')
            answer_text = f"NetHack has approximately {count} variants in the knowledge graph."
            confidence = 0.8
        else:
            answer_text = "I'm not sure what to count. Please be more specific."
            confidence = 0.3
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=[],
            reasoning_path=["Counted entities"],
            answer_type='count'
        )
    
    def _answer_comparison_fast(self, question: str, entities: List[MatchedEntity]) -> Answer:
        """Fast comparison"""
        if len(entities) < 2:
            return Answer(
                text="I need two things to compare.",
                confidence=0.0,
                evidence_triples=[],
                reasoning_path=[],
                answer_type='comparison'
            )
        
        e1, e2 = entities[0], entities[1]
        
        # Get properties
        e1_props = set()
        e2_props = set()
        
        for _, role, rel, other in self.entity_triples.get(e1.name, []):
            if role == 'head':
                e1_props.add(f"{rel} {other}")
        
        for _, role, rel, other in self.entity_triples.get(e2.name, []):
            if role == 'head':
                e2_props.add(f"{rel} {other}")
        
        common = e1_props & e2_props
        only_e1 = e1_props - e2_props
        only_e2 = e2_props - e1_props
        
        answer_parts = []
        if common:
            answer_parts.append(f"Both: {'; '.join(list(common)[:2])}")
        if only_e1:
            answer_parts.append(f"{e1.name}: {'; '.join(list(only_e1)[:2])}")
        if only_e2:
            answer_parts.append(f"{e2.name}: {'; '.join(list(only_e2)[:2])}")
        
        answer_text = ' | '.join(answer_parts) if answer_parts else f"No clear differences found between {e1.name} and {e2.name}."
        confidence = 0.7 * min(e1.final_score, e2.final_score)
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=[],
            reasoning_path=["Compared properties"],
            answer_type='comparison'
        )
    
    def _answer_general_fast(self, question: str, entities: List[MatchedEntity],
                           relations: List[Tuple[str, float]]) -> Answer:
        """Fast general answering"""
        if not entities:
            return Answer(
                text="I couldn't understand what you're asking about.",
                confidence=0.0,
                evidence_triples=[],
                reasoning_path=[],
                answer_type='general'
            )
        
        target = entities[0]
        facts = []
        
        # Get relevant facts
        for idx, role, rel, other in self.entity_triples.get(target.name, [])[:10]:
            if relations and rel == relations[0][0]:
                facts.append((rel, other, 1.0))
            else:
                facts.append((rel, other, 0.5))
        
        if not facts:
            return Answer(
                text=f"I don't have information about {target.name}.",
                confidence=0.3,
                evidence_triples=[],
                reasoning_path=[],
                answer_type='general'
            )
        
        # Sort by relevance
        facts.sort(key=lambda x: x[2], reverse=True)
        
        # Format answer
        if len(facts) == 1:
            answer_text = f"{target.name} {facts[0][0]} {facts[0][1]}."
        else:
            answer_text = f"{target.name}: " + "; ".join([f"{r} {o}" for r, o, _ in facts[:3]])
        
        evidence = [(target.name, r, o) for r, o, _ in facts[:3]]
        confidence = 0.7 * target.final_score
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=evidence,
            reasoning_path=[f"Found {len(facts)} relevant facts"],
            answer_type='general'
        )


def test_fast_qa():
    """Test the fast QA system"""
    print("Initializing Fast NetHack KGQA...")
    qa = FastNetHackKGQA('data/nethack_kg_full/nethack_kg_intermediate.tsv')
    
    test_questions = [
        "What is the Amulet of Yendor?",
        "Is Sokoban part of NetHack?",
        "How many variants does NetHack have?",
        "What's the difference between blessed and cursed?",
        "What items are in Gehennom?"
    ]
    
    print("\nTesting questions:")
    for q in test_questions:
        print(f"\nQ: {q}")
        answer = qa.answer(q)
        print(f"A: {answer.text}")
        print(f"Confidence: {answer.confidence:.1%}")
    
    print("\n\nInteractive mode (type 'quit' to exit):")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit']:
            break
        
        if question:
            answer = qa.answer(question)
            print(f"Answer: {answer.text}")
            print(f"Confidence: {answer.confidence:.1%}")


if __name__ == "__main__":
    test_fast_qa()