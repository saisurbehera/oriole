"""
Fast NetHack KGQA with FAISS indexing and improved answer generation
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


# Priority relations for better answers
PRIORITY_RELATIONS = {
    'definition': ['instance_of', 'subclass_of', 'said to be the same as'],
    'description': ['facet_of', 'described by', 'characteristics', 'notable type'],
    'usage': ['uses', 'used by', 'used for', 'useful for'],
    'location': ['part_of', 'location', 'found in', 'appears in'],
    'properties': ['has_property', 'has_effect', 'grants', 'provides'],
    'game_mechanics': ['damage', 'armor_class', 'weight', 'cost', 'material'],
}

# Relations to deprioritize
BORING_RELATIONS = ['different from', 'follows', 'followed by', 'point in time']


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
    """Optimized KGQA with FAISS indexing and better answers"""
    
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
        
        # Precompute relation priorities
        self._init_relation_priorities()
    
    def _init_relation_priorities(self):
        """Initialize relation priority scoring"""
        self.relation_priority = {}
        
        # Assign priority scores
        for category, rels in PRIORITY_RELATIONS.items():
            for rel in rels:
                if rel in self.relations:
                    self.relation_priority[rel] = 10 - list(PRIORITY_RELATIONS.keys()).index(category)
        
        # Low priority for boring relations
        for rel in BORING_RELATIONS:
            if rel in self.relations:
                self.relation_priority[rel] = 0.1
        
        # Default priority
        for rel in self.relations:
            if rel not in self.relation_priority:
                self.relation_priority[rel] = 1.0
    
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
        
        # Compute entity importance (degree + special entities)
        self.entity_importance = {}
        for entity in self.entities:
            base_score = len(self.entity_triples[entity]) / 100.0
            
            # Boost important entities
            if any(key in entity.lower() for key in ['amulet', 'yendor', 'nethack', 'artifact']):
                base_score *= 2
            
            self.entity_importance[entity] = base_score
        
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
        """Fast question answering with better responses"""
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
        
        # Priority entities for exact matching
        priority_entities = [
            'Amulet of Yendor', 'NetHack', 'Sokoban', 'Gehennom', 'SLASH\'EM',
            'Wizard of Yendor', 'Excalibur', 'Stormbringer', 'Mjollnir',
            'Castle', 'Astral Plane', 'Dungeons of Doom', 'Gnomish Mines'
        ]
        
        # Check exact matches first
        for entity in priority_entities:
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
        
        # Then check other entities with word boundaries
        for entity in self.entities:
            if entity not in candidates and len(entity) > 2:
                # Check for word boundary match
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
        
        # Semantic search for phrases
        words = question.split()
        phrases_to_search = []
        
        # Add meaningful phrases
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words)+1)):
                phrase = ' '.join(words[i:j])
                if len(phrase) > 3:
                    phrases_to_search.append(phrase)
        
        # Limit semantic searches
        for phrase in phrases_to_search[:3]:
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
        """Fast definition answering with better descriptions"""
        if not entities:
            return Answer(
                text="I couldn't identify what you're asking about.",
                confidence=0.0,
                evidence_triples=[],
                reasoning_path=[],
                answer_type='definition'
            )
        
        target = entities[0]
        
        # Collect facts organized by relation type
        facts_by_type = defaultdict(list)
        all_evidence = []
        
        for idx, role, relation, other in self.entity_triples.get(target.name, []):
            priority = self.relation_priority.get(relation, 1.0)
            facts_by_type[relation].append((other, role, priority))
            
            if role == 'head':
                all_evidence.append((target.name, relation, other))
            else:
                all_evidence.append((other, relation, target.name))
        
        if not facts_by_type:
            return Answer(
                text=f"I don't have information about {target.name}.",
                confidence=0.3,
                evidence_triples=[],
                reasoning_path=[],
                answer_type='definition'
            )
        
        # Build comprehensive answer
        answer_parts = []
        used_relations = set()
        
        # 1. What it is (instance/subclass)
        for rel in ['instance_of', 'subclass_of']:
            if rel in facts_by_type and rel not in used_relations:
                items = [other for other, role, _ in facts_by_type[rel] if role == 'head']
                if items:
                    answer_parts.append(f"{target.name} is a {items[0]}")
                    used_relations.add(rel)
                    break
        
        # 2. Description/characteristics
        for rel in ['facet_of', 'described_by', 'characteristics']:
            if rel in facts_by_type and rel not in used_relations:
                items = [other for other, role, _ in facts_by_type[rel] if role == 'head']
                if items:
                    answer_parts.append(f"It is {rel.replace('_', ' ')} {items[0]}")
                    used_relations.add(rel)
                    break
        
        # 3. Where it's found
        if 'part_of' in facts_by_type and 'part_of' not in used_relations:
            locations = [other for other, role, _ in facts_by_type['part_of'] if role == 'head']
            if locations:
                answer_parts.append(f"It is found in {locations[0]}")
                used_relations.add('part_of')
        
        # 4. What it does/used for
        for rel in ['uses', 'used_for', 'grants', 'provides']:
            if rel in facts_by_type and rel not in used_relations:
                items = [other for other, role, _ in facts_by_type[rel] if role == 'head']
                if items:
                    answer_parts.append(f"It {rel.replace('_', ' ')} {items[0]}")
                    used_relations.add(rel)
                    break
        
        # 5. Additional important facts
        sorted_facts = []
        for rel, facts in facts_by_type.items():
            if rel not in used_relations and rel not in BORING_RELATIONS:
                for other, role, priority in facts:
                    if role == 'head':
                        sorted_facts.append((rel, other, priority))
        
        sorted_facts.sort(key=lambda x: x[2], reverse=True)
        for rel, other, _ in sorted_facts[:2]:
            answer_parts.append(f"It {rel.replace('_', ' ')} {other}")
            if len(answer_parts) >= 5:
                break
        
        # Combine answer
        if answer_parts:
            answer_text = '. '.join(answer_parts) + '.'
        else:
            answer_text = f"{target.name} is mentioned in the knowledge base but lacks detailed information."
        
        confidence = min(1.0, 0.5 + 0.1 * len(answer_parts)) * target.final_score
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=all_evidence[:5],
            reasoning_path=[f"Found {len(facts_by_type)} different types of information about {target.name}"],
            answer_type='definition'
        )
    
    def _answer_boolean_fast(self, question: str, entities: List[MatchedEntity], 
                            relations: List[Tuple[str, float]]) -> Answer:
        """Fast boolean answering"""
        if len(entities) >= 2:
            e1, e2 = entities[0], entities[1]
            
            # Check direct connection
            for _, role, rel, other in self.entity_triples.get(e1.name, []):
                if other == e2.name:
                    if role == 'head':
                        answer_text = f"Yes, {e1.name} {rel.replace('_', ' ')} {e2.name}."
                    else:
                        answer_text = f"Yes, {e2.name} {rel.replace('_', ' ')} {e1.name}."
                    
                    return Answer(
                        text=answer_text,
                        confidence=0.9 * min(e1.final_score, e2.final_score),
                        evidence_triples=[(e1.name, rel, e2.name) if role == 'head' else (e2.name, rel, e1.name)],
                        reasoning_path=["Found direct connection"],
                        answer_type='boolean'
                    )
            
            # Check reverse connection
            for _, role, rel, other in self.entity_triples.get(e2.name, []):
                if other == e1.name:
                    if role == 'head':
                        answer_text = f"Yes, {e2.name} {rel.replace('_', ' ')} {e1.name}."
                    else:
                        answer_text = f"Yes, {e1.name} {rel.replace('_', ' ')} {e2.name}."
                    
                    return Answer(
                        text=answer_text,
                        confidence=0.9 * min(e1.final_score, e2.final_score),
                        evidence_triples=[(e2.name, rel, e1.name) if role == 'head' else (e1.name, rel, e2.name)],
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
        """Fast counting with more intelligence"""
        # NetHack variants
        if 'variant' in question and any(e.name == 'NetHack' for e in entities):
            variants = [e for e in self.entities if 'NetHack' in e or e.endswith('Hack') or e.endswith('hack')]
            variants = [v for v in variants if v != 'NetHack' and 'talk' not in v.lower()]
            
            answer_text = f"NetHack has approximately {len(variants)} variants in the knowledge graph, including SLASH'EM, UnNetHack, EvilHack, and others."
            confidence = 0.8
        
        # Artifacts
        elif 'artifact' in question:
            artifacts = []
            for entity in self.entities:
                for _, role, rel, other in self.entity_triples.get(entity, []):
                    if rel == 'instance_of' and other == 'artifact' and role == 'head':
                        artifacts.append(entity)
                        break
            
            answer_text = f"There are {len(artifacts)} artifacts in the knowledge graph."
            confidence = 0.7
        
        # General counting
        elif entities:
            target = entities[0]
            count = len(self.entity_triples.get(target.name, []))
            answer_text = f"{target.name} has {count} relationships in the knowledge graph."
            confidence = 0.6
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
        e1_facts = defaultdict(set)
        e2_facts = defaultdict(set)
        
        for _, role, rel, other in self.entity_triples.get(e1.name, []):
            if role == 'head' and rel not in BORING_RELATIONS:
                e1_facts[rel].add(other)
        
        for _, role, rel, other in self.entity_triples.get(e2.name, []):
            if role == 'head' and rel not in BORING_RELATIONS:
                e2_facts[rel].add(other)
        
        # Find similarities and differences
        common_relations = set(e1_facts.keys()) & set(e2_facts.keys())
        only_e1 = set(e1_facts.keys()) - set(e2_facts.keys())
        only_e2 = set(e2_facts.keys()) - set(e1_facts.keys())
        
        answer_parts = []
        
        # Similarities
        if common_relations:
            similarities = []
            for rel in list(common_relations)[:3]:
                if e1_facts[rel] == e2_facts[rel]:
                    similarities.append(f"both {rel.replace('_', ' ')} {list(e1_facts[rel])[0]}")
            if similarities:
                answer_parts.append(f"Similarities: {', '.join(similarities)}")
        
        # Differences
        if only_e1:
            diff1 = []
            for rel in list(only_e1)[:2]:
                diff1.append(f"{rel.replace('_', ' ')} {list(e1_facts[rel])[0]}")
            if diff1:
                answer_parts.append(f"{e1.name}: {', '.join(diff1)}")
        
        if only_e2:
            diff2 = []
            for rel in list(only_e2)[:2]:
                diff2.append(f"{rel.replace('_', ' ')} {list(e2_facts[rel])[0]}")
            if diff2:
                answer_parts.append(f"{e2.name}: {', '.join(diff2)}")
        
        if answer_parts:
            answer_text = '. '.join(answer_parts) + '.'
        else:
            answer_text = f"Both {e1.name} and {e2.name} are in the knowledge base but have few comparable properties."
        
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
        """Fast general answering with context awareness"""
        if not entities:
            return Answer(
                text="I couldn't understand what you're asking about.",
                confidence=0.0,
                evidence_triples=[],
                reasoning_path=[],
                answer_type='general'
            )
        
        target = entities[0]
        
        # Look for context clues in the question
        context_clues = {
            'where': ['part_of', 'location', 'found_in'],
            'who': ['owned_by', 'created_by', 'wielded_by'],
            'when': ['publication_date', 'release_date', 'point_in_time'],
            'how': ['uses', 'method', 'process'],
            'why': ['purpose', 'reason', 'because'],
        }
        
        # Determine relevant relations based on question
        relevant_relations = []
        for clue, rels in context_clues.items():
            if clue in question.lower():
                relevant_relations.extend(rels)
        
        # Also use matched relations
        if relations:
            relevant_relations.extend([r for r, _ in relations[:2]])
        
        # Get facts prioritizing relevant relations
        facts = []
        for idx, role, rel, other in self.entity_triples.get(target.name, [])[:20]:
            if rel in relevant_relations:
                priority = 2.0
            else:
                priority = self.relation_priority.get(rel, 1.0)
            
            if rel not in BORING_RELATIONS:
                facts.append((rel, other, role, priority))
        
        if not facts:
            return Answer(
                text=f"I don't have relevant information about {target.name} for your question.",
                confidence=0.3,
                evidence_triples=[],
                reasoning_path=[],
                answer_type='general'
            )
        
        # Sort by relevance
        facts.sort(key=lambda x: x[3], reverse=True)
        
        # Build answer
        answer_parts = []
        evidence = []
        
        for rel, other, role, _ in facts[:3]:
            if role == 'head':
                answer_parts.append(f"{target.name} {rel.replace('_', ' ')} {other}")
                evidence.append((target.name, rel, other))
            else:
                answer_parts.append(f"{other} {rel.replace('_', ' ')} {target.name}")
                evidence.append((other, rel, target.name))
        
        answer_text = '. '.join(answer_parts) + '.'
        confidence = min(0.8, 0.5 + 0.1 * len(answer_parts)) * target.final_score
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=evidence,
            reasoning_path=[f"Found {len(facts)} relevant facts about {target.name}"],
            answer_type='general'
        )


def test_improved_qa():
    """Test the improved QA system"""
    print("Initializing Improved Fast NetHack KGQA...")
    qa = FastNetHackKGQA('data/nethack_kg_full/nethack_kg_intermediate.tsv')
    
    test_questions = [
        "What is the Amulet of Yendor?",
        "What is Excalibur?",
        "Where is Sokoban?",
        "Is Sokoban part of NetHack?",
        "How many variants does NetHack have?",
        "What's the difference between blessed and cursed?",
        "What items are in Gehennom?",
        "Who owns the Amulet of Yendor?",
        "What is used for ascension?",
        "How are NetHack and Rogue related?"
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
    test_improved_qa()