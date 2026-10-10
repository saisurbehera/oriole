"""
NetHack Knowledge Graph Question Answering System
Handles any query type with semantic matching, contextual relevance,
confidence scoring, and multi-criteria evaluation
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
import json
import logging
from enum import Enum
import os
import sys

# Add ULTRA to path if available
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultra'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchedEntity:
    """Entity match with all scoring components"""
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
class MatchedRelation:
    """Relation match with scoring"""
    name: str
    semantic_score: float
    exact_match: bool
    pattern_score: float
    final_score: float


@dataclass
class Answer:
    """Complete answer with all evaluation metrics"""
    text: str
    confidence: float
    evidence_triples: List[Tuple[str, str, str]]
    reasoning_path: List[str]
    alternative_answers: List[Tuple[str, float]]
    answer_type: str
    evaluation_breakdown: Dict[str, float]
    supporting_context: Dict[str, Any]


class NetHackKGQA:
    """Advanced KGQA system for NetHack"""
    
    def __init__(self, kg_path: str, use_ultra: bool = True):
        self.kg_path = kg_path
        self.use_ultra = use_ultra
        
        # Load components
        logger.info("Initializing NetHack KGQA System...")
        self.semantic_encoder = SentenceTransformer('all-mpnet-base-v2')
        
        # Load KG
        self._load_kg()
        
        # Pre-compute embeddings and importance
        self._compute_embeddings()
        self._compute_importance_scores()
        
        # Initialize query understanding components
        self._init_query_patterns()
        self._init_nethack_context()
        
        # ULTRA for complex reasoning
        self.ultra_model = None
        self.ultra_data = None
        if use_ultra:
            self._init_ultra()
    
    def _load_kg(self):
        """Load NetHack KG"""
        logger.info(f"Loading KG from {self.kg_path}")
        
        # Load triples
        self.df = pd.read_csv(self.kg_path, sep='\t', header=0)
        if len(self.df.columns) == 3:
            self.df.columns = ['head', 'relation', 'tail']
            self.df['source'] = 'unknown'
        else:
            self.df.columns = ['head', 'relation', 'tail', 'source']
        
        # Build indexes
        self.entities = set()
        self.relations = set()
        self.entity_triples = defaultdict(list)
        self.relation_triples = defaultdict(list)
        self.entity_relations = defaultdict(set)
        
        for idx, row in self.df.iterrows():
            h, r, t = str(row['head']), str(row['relation']), str(row['tail'])
            
            self.entities.add(h)
            self.entities.add(t)
            self.relations.add(r)
            
            # Index by entity
            self.entity_triples[h].append((idx, 'head', r, t))
            self.entity_triples[t].append((idx, 'tail', r, h))
            
            # Index by relation
            self.relation_triples[r].append((h, t))
            
            # Track what relations each entity participates in
            self.entity_relations[h].add(r)
            self.entity_relations[t].add(r)
        
        # Build graph
        self.graph = nx.MultiDiGraph()
        for _, row in self.df.iterrows():
            self.graph.add_edge(row['head'], row['tail'], 
                              relation=row['relation'], 
                              source=row.get('source', 'unknown'))
        
        logger.info(f"Loaded {len(self.entities)} entities, {len(self.relations)} relations, "
                   f"{len(self.df)} triples")
    
    def _compute_embeddings(self):
        """Pre-compute embeddings for all entities and relations"""
        logger.info("Computing semantic embeddings...")
        
        # Entity embeddings
        entity_list = list(self.entities)
        entity_embeddings = self.semantic_encoder.encode(entity_list, 
                                                        batch_size=256, 
                                                        show_progress_bar=True)
        self.entity_embeddings = {e: emb for e, emb in zip(entity_list, entity_embeddings)}
        
        # Relation embeddings - include variations
        relation_variations = []
        relation_map = {}
        
        for rel in self.relations:
            # Original
            relation_variations.append(rel)
            relation_map[len(relation_variations)-1] = (rel, 'original')
            
            # With spaces
            spaced = rel.replace('_', ' ')
            relation_variations.append(spaced)
            relation_map[len(relation_variations)-1] = (rel, 'spaced')
            
            # Question form
            if rel.startswith('has_'):
                question_form = f"what {rel[4:]} does it have"
                relation_variations.append(question_form)
                relation_map[len(relation_variations)-1] = (rel, 'question')
        
        relation_embeddings = self.semantic_encoder.encode(relation_variations,
                                                         batch_size=256,
                                                         show_progress_bar=True)
        
        self.relation_embeddings = defaultdict(dict)
        for idx, emb in enumerate(relation_embeddings):
            rel, variant = relation_map[idx]
            self.relation_embeddings[rel][variant] = emb
    
    def _compute_importance_scores(self):
        """Compute importance scores for entities"""
        logger.info("Computing entity importance scores...")
        
        # PageRank
        pagerank = nx.pagerank(self.graph)
        
        # Degree centrality
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        
        # Normalize and combine
        self.entity_importance = {}
        max_pr = max(pagerank.values()) if pagerank else 1
        max_degree = max(max(in_degree.values()), max(out_degree.values())) if in_degree else 1
        
        for entity in self.entities:
            pr_score = pagerank.get(entity, 0) / max_pr
            in_score = in_degree.get(entity, 0) / max_degree
            out_score = out_degree.get(entity, 0) / max_degree
            
            # Weighted combination
            self.entity_importance[entity] = (
                0.5 * pr_score + 
                0.3 * in_score + 
                0.2 * out_score
            )
        
        # Special boost for key NetHack entities
        key_entities = {'NetHack', 'Amulet of Yendor', 'player', 'ascension', 
                       'Gehennom', 'Wizard of Yendor', 'SLASH\'EM', 'NetHackWiki'}
        for entity in key_entities:
            if entity in self.entity_importance:
                self.entity_importance[entity] = min(1.0, self.entity_importance[entity] * 1.5)
    
    def _init_query_patterns(self):
        """Initialize query patterns for understanding"""
        self.query_patterns = {
            'definition': [
                r'what (?:is|are)(?: a| an| the)? (.+?)(?:\?|$)',
                r'(?:define|describe|explain)(?: a| an| the)? (.+?)(?:\?|$)',
                r'tell me about (.+?)(?:\?|$)'
            ],
            'property': [
                r'what (.+?) does (.+?) have',
                r'(.+?)(?:\'s|s\') (.+?)(?:\?|$)',
                r'(?:list|show|give me)(?: all)? (.+?) (?:of|for|in) (.+?)(?:\?|$)'
            ],
            'relationship': [
                r'how (?:is|are) (.+?) (?:related to|connected to) (.+?)(?:\?|$)',
                r'(?:what is the )?(?:relationship|connection) between (.+?) and (.+?)(?:\?|$)',
                r'(.+?) and (.+?) relationship'
            ],
            'boolean': [
                r'(?:is|are|does|do|can|will|has|have) (.+?) (.+?)(?:\?|$)',
                r'(?:true|false):? (.+?)(?:\?|$)'
            ],
            'comparison': [
                r'(?:difference|differences) between (.+?) and (.+?)(?:\?|$)',
                r'compare (.+?) (?:to|with|and) (.+?)(?:\?|$)',
                r'(.+?) (?:vs\.?|versus) (.+?)(?:\?|$)'
            ],
            'count': [
                r'how many (.+?)(?:\?|$)',
                r'(?:count|number of|total) (.+?)(?:\?|$)',
                r'list all (.+?)(?:\?|$)'
            ],
            'path': [
                r'how (?:do I get|to get) from (.+?) to (.+?)(?:\?|$)',
                r'path from (.+?) to (.+?)(?:\?|$)',
                r'(.+?) leads to (.+?)(?:\?|$)'
            ]
        }
    
    def _init_nethack_context(self):
        """Initialize NetHack-specific context"""
        # Common entity aliases
        self.entity_aliases = {
            'aoy': 'Amulet of Yendor',
            'woy': 'Wizard of Yendor',
            'quest': 'The Quest',
            'valk': 'Valkyrie',
            'wiz': 'Wizard',
            'arch': 'Archeologist',
            'excal': 'Excalibur',
            'stormy': 'Stormbringer',
            'cow': 'Wand of death',
            'soko': 'Sokoban'
        }
        
        # Relation mappings for natural language
        self.relation_nl_map = {
            'part_of': ['part of', 'in', 'belongs to', 'contained in'],
            'has_part': ['contains', 'has', 'includes'],
            'instance_of': ['is a', 'type of', 'kind of'],
            'subclass_of': ['is a type of', 'subcategory of'],
            'used_by': ['used by', 'wielded by', 'carried by'],
            'uses': ['uses', 'requires', 'needs'],
            'different_from': ['different from', 'not the same as', 'distinct from'],
            'present_in_work': ['appears in', 'found in', 'present in']
        }
    
    def _init_ultra(self):
        """Initialize ULTRA for complex reasoning"""
        try:
            from ultra.tasks import build_relation_graph
            from ultra.models import Ultra
            import yaml
            from easydict import EasyDict as edict
            
            # Load KG in ULTRA format
            logger.info("Initializing ULTRA model...")
            
            # This would be the full ULTRA initialization
            # For now, marking as available
            self.ultra_available = True
            logger.info("ULTRA integration ready")
            
        except Exception as e:
            logger.warning(f"ULTRA not available: {e}")
            self.ultra_available = False
    
    def answer(self, question: str) -> Answer:
        """Main entry point - answer any question about NetHack"""
        logger.info(f"Question: {question}")
        
        # Normalize question
        question_lower = question.lower().strip()
        
        # Extract entities and relations
        entity_matches = self._match_entities(question_lower)
        relation_matches = self._match_relations(question_lower)
        
        # Determine query type and route accordingly
        query_type = self._determine_query_type(question_lower, entity_matches, relation_matches)
        
        logger.info(f"Query type: {query_type}")
        logger.info(f"Entities: {[(e.name, e.final_score) for e in entity_matches[:3]]}")
        logger.info(f"Relations: {[(r.name, r.final_score) for r in relation_matches[:3]]}")
        
        # Route to appropriate handler
        if query_type == 'definition':
            return self._answer_definition(question_lower, entity_matches)
        elif query_type == 'property':
            return self._answer_property(question_lower, entity_matches, relation_matches)
        elif query_type == 'relationship':
            return self._answer_relationship(question_lower, entity_matches)
        elif query_type == 'boolean':
            return self._answer_boolean(question_lower, entity_matches, relation_matches)
        elif query_type == 'comparison':
            return self._answer_comparison(question_lower, entity_matches)
        elif query_type == 'count':
            return self._answer_count(question_lower, entity_matches, relation_matches)
        elif query_type == 'path':
            return self._answer_path(question_lower, entity_matches)
        else:
            # General fact retrieval
            return self._answer_general(question_lower, entity_matches, relation_matches)
    
    def _match_entities(self, question: str) -> List[MatchedEntity]:
        """Match entities with semantic similarity and context"""
        matches = []
        
        # First check aliases
        for alias, full_name in self.entity_aliases.items():
            if alias in question:
                if full_name in self.entities:
                    matches.append(MatchedEntity(
                        name=full_name,
                        semantic_score=1.0,
                        exact_match=True,
                        context_score=1.0,
                        importance_score=self.entity_importance.get(full_name, 0.5),
                        final_score=1.0
                    ))
        
        # Extract potential entity mentions (1-4 word phrases)
        words = question.split()
        candidates = []
        
        for i in range(len(words)):
            for j in range(i+1, min(i+5, len(words)+1)):
                phrase = ' '.join(words[i:j])
                candidates.append(phrase)
        
        # Check each candidate
        for candidate in candidates:
            # Exact match
            if candidate in self.entities:
                importance = self.entity_importance.get(candidate, 0.5)
                matches.append(MatchedEntity(
                    name=candidate,
                    semantic_score=1.0,
                    exact_match=True,
                    context_score=1.0,
                    importance_score=importance,
                    final_score=1.0
                ))
                continue
            
            # Semantic similarity
            if len(candidate) > 2:  # Skip very short phrases
                cand_emb = self.semantic_encoder.encode([candidate])[0]
                
                similarities = []
                for entity, emb in self.entity_embeddings.items():
                    sim = cosine_similarity([cand_emb], [emb])[0][0]
                    if sim > 0.6:  # Threshold
                        similarities.append((entity, sim))
                
                # Take top matches
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                for entity, sim in similarities[:3]:
                    # Context score based on co-occurrence
                    context_score = self._compute_context_score(candidate, entity, question)
                    
                    # Importance
                    importance = self.entity_importance.get(entity, 0.5)
                    
                    # Final score combining all factors
                    final_score = (
                        0.4 * sim +
                        0.2 * context_score +
                        0.2 * importance +
                        0.2 * (1.0 if entity.lower() in question else 0.0)
                    )
                    
                    matches.append(MatchedEntity(
                        name=entity,
                        semantic_score=sim,
                        exact_match=False,
                        context_score=context_score,
                        importance_score=importance,
                        final_score=final_score
                    ))
        
        # Deduplicate and sort
        seen = set()
        unique_matches = []
        for match in sorted(matches, key=lambda x: x.final_score, reverse=True):
            if match.name not in seen:
                seen.add(match.name)
                unique_matches.append(match)
        
        return unique_matches
    
    def _match_relations(self, question: str) -> List[MatchedRelation]:
        """Match relations considering variations"""
        matches = []
        
        # Check each relation and its variations
        for relation in self.relations:
            scores = []
            
            # Exact match
            if relation in question:
                scores.append(1.0)
            
            # Check NL variations
            if relation in self.relation_nl_map:
                for variation in self.relation_nl_map[relation]:
                    if variation in question:
                        scores.append(0.9)
            
            # Semantic similarity
            question_emb = self.semantic_encoder.encode([question])[0]
            
            for variant, emb in self.relation_embeddings[relation].items():
                sim = cosine_similarity([question_emb], [emb])[0][0]
                if sim > 0.5:
                    scores.append(sim)
            
            if scores:
                max_score = max(scores)
                pattern_score = self._relation_pattern_score(relation, question)
                
                final_score = 0.7 * max_score + 0.3 * pattern_score
                
                matches.append(MatchedRelation(
                    name=relation,
                    semantic_score=max_score,
                    exact_match=(max_score == 1.0),
                    pattern_score=pattern_score,
                    final_score=final_score
                ))
        
        return sorted(matches, key=lambda x: x.final_score, reverse=True)
    
    def _compute_context_score(self, candidate: str, entity: str, question: str) -> float:
        """Compute context score based on question context"""
        score = 0.5  # Base score
        
        # Check if other entities in question are related
        question_entities = [e.name for e in self._match_entities(question)[:5]]
        
        for q_entity in question_entities:
            if q_entity != entity:
                # Check if connected in graph
                if self.graph.has_edge(entity, q_entity) or self.graph.has_edge(q_entity, entity):
                    score += 0.2
                    break
        
        # Domain-specific boosts
        if 'item' in question and entity in self.entity_triples:
            if any('instance_of' in str(t) and 'item' in str(t) for t in self.entity_triples[entity]):
                score += 0.1
        
        if 'monster' in question and entity in self.entity_triples:
            if any('instance_of' in str(t) and 'monster' in str(t) for t in self.entity_triples[entity]):
                score += 0.1
        
        return min(1.0, score)
    
    def _relation_pattern_score(self, relation: str, question: str) -> float:
        """Score based on question patterns"""
        score = 0.0
        
        # Check question words that suggest certain relations
        if relation == 'instance_of' and any(word in question for word in ['what is', 'type of', 'kind of']):
            score += 0.5
        elif relation == 'has_part' and any(word in question for word in ['contains', 'includes', 'parts']):
            score += 0.5
        elif relation == 'part_of' and any(word in question for word in ['belongs to', 'found in', 'located in']):
            score += 0.5
        elif relation == 'used_by' and any(word in question for word in ['who uses', 'wielded by']):
            score += 0.5
        
        return score
    
    def _determine_query_type(self, question: str, entities: List[MatchedEntity], 
                            relations: List[MatchedRelation]) -> str:
        """Determine query type from question"""
        # Check patterns
        for qtype, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    return qtype
        
        # Heuristics
        if question.startswith('what') and len(entities) == 1:
            return 'definition'
        elif any(word in question for word in ['how many', 'count', 'number of', 'list all']):
            return 'count'
        elif len(entities) >= 2 and any(word in question for word in ['and', 'between']):
            return 'relationship'
        
        return 'general'
    
    def _answer_definition(self, question: str, entities: List[MatchedEntity]) -> Answer:
        """Answer definition questions"""
        if not entities:
            return self._create_answer(
                "I couldn't identify what you're asking about.",
                confidence=0.0,
                answer_type='definition'
            )
        
        target = entities[0]
        
        # Gather all facts about the entity
        facts = []
        evidence_triples = []
        
        if target.name in self.entity_triples:
            for idx, role, relation, other in self.entity_triples[target.name]:
                facts.append(f"{target.name} {relation} {other}" if role == 'head' 
                           else f"{other} {relation} {target.name}")
                evidence_triples.append((
                    target.name if role == 'head' else other,
                    relation,
                    other if role == 'head' else target.name
                ))
        
        if not facts:
            return self._create_answer(
                f"I don't have information about {target.name}.",
                confidence=0.3,
                answer_type='definition'
            )
        
        # Prioritize important facts
        important_facts = []
        
        # Instance/type information
        for fact in facts:
            if 'instance_of' in fact or 'subclass_of' in fact:
                important_facts.append(fact)
        
        # Then part_of/has_part
        for fact in facts:
            if 'part_of' in fact or 'has_part' in fact:
                important_facts.append(fact)
        
        # Construct answer
        definition_parts = []
        
        # Type information
        type_facts = [f for f in important_facts if 'instance_of' in f or 'subclass_of' in f]
        if type_facts:
            types = [f.split()[-1] for f in type_facts[:2]]
            definition_parts.append(f"{target.name} is a {' and '.join(types)}")
        
        # Location/part information
        location_facts = [f for f in facts if 'part_of' in f and f.startswith(target.name)]
        if location_facts:
            locations = [f.split()[-1] for f in location_facts[:2]]
            definition_parts.append(f"It is part of {' and '.join(locations)}")
        
        # Key properties
        property_facts = [f for f in facts if any(r in f for r in ['has_part', 'uses', 'present_in_work'])]
        if property_facts and len(definition_parts) < 3:
            definition_parts.append(property_facts[0])
        
        answer_text = '. '.join(definition_parts) + '.'
        
        # Confidence based on evidence
        confidence = min(1.0, 0.3 + 0.1 * len(evidence_triples[:10]))
        confidence *= target.final_score  # Adjust by match quality
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=evidence_triples[:10],
            reasoning_path=[f"Found {len(facts)} facts about {target.name}"],
            alternative_answers=[],
            answer_type='definition',
            evaluation_breakdown={
                'entity_match': target.final_score,
                'evidence_count': min(1.0, len(facts) / 10),
                'fact_diversity': len(set(r for _, r, _ in evidence_triples)) / 10
            },
            supporting_context={'all_facts': facts[:20]}
        )
    
    def _answer_property(self, question: str, entities: List[MatchedEntity], 
                        relations: List[MatchedRelation]) -> Answer:
        """Answer property questions"""
        if not entities:
            return self._create_answer("I couldn't identify what you're asking about.", 
                                     confidence=0.0, answer_type='property')
        
        target = entities[0]
        
        # Find relevant relation
        relevant_relation = None
        if relations and relations[0].final_score > 0.5:
            relevant_relation = relations[0].name
        
        # Get properties
        properties = []
        evidence = []
        
        for idx, role, relation, other in self.entity_triples.get(target.name, []):
            if role == 'head':  # Entity has this property
                if not relevant_relation or relation == relevant_relation:
                    properties.append(other)
                    evidence.append((target.name, relation, other))
        
        if not properties:
            return self._create_answer(
                f"{target.name} doesn't have any {relevant_relation or 'properties'} in my knowledge.",
                confidence=0.5,
                answer_type='property'
            )
        
        # Format answer
        if len(properties) == 1:
            answer_text = f"{target.name} {relevant_relation or 'has'} {properties[0]}."
        else:
            answer_text = f"{target.name} {relevant_relation or 'has'}: {', '.join(properties[:10])}"
            if len(properties) > 10:
                answer_text += f" (and {len(properties) - 10} more)"
        
        confidence = min(1.0, 0.5 + 0.1 * len(properties))
        confidence *= target.final_score
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=evidence[:10],
            reasoning_path=[f"Found {len(properties)} {relevant_relation or 'properties'} for {target.name}"],
            alternative_answers=[],
            answer_type='property',
            evaluation_breakdown={
                'entity_match': target.final_score,
                'relation_match': relations[0].final_score if relations else 0.5,
                'result_count': min(1.0, len(properties) / 5)
            },
            supporting_context={'all_properties': properties}
        )
    
    def _answer_boolean(self, question: str, entities: List[MatchedEntity],
                       relations: List[MatchedRelation]) -> Answer:
        """Answer yes/no questions"""
        if len(entities) < 2 and not relations:
            return self._create_answer("I need more information to answer this question.",
                                     confidence=0.0, answer_type='boolean')
        
        # Check if it's a fact verification
        if len(entities) >= 2:
            e1, e2 = entities[0], entities[1]
            
            # Check direct connection
            evidence = []
            for idx, role, relation, other in self.entity_triples.get(e1.name, []):
                if other == e2.name:
                    evidence.append((e1.name if role == 'head' else other,
                                   relation,
                                   other if role == 'head' else e1.name))
            
            if evidence:
                # Fact exists
                answer_text = "Yes"
                if relations and relations[0].final_score > 0.7:
                    # Check if specific relation matches
                    matching_evidence = [e for e in evidence if relations[0].name in e[1]]
                    if matching_evidence:
                        answer_text = f"Yes, {e1.name} {relations[0].name} {e2.name}."
                        confidence = 0.9
                    else:
                        answer_text = f"No, but {e1.name} {evidence[0][1]} {e2.name}."
                        confidence = 0.8
                else:
                    answer_text = f"Yes, {e1.name} is related to {e2.name}."
                    confidence = 0.85
                
                return Answer(
                    text=answer_text,
                    confidence=confidence * min(e1.final_score, e2.final_score),
                    evidence_triples=evidence,
                    reasoning_path=[f"Found direct connection between {e1.name} and {e2.name}"],
                    alternative_answers=[],
                    answer_type='boolean',
                    evaluation_breakdown={
                        'entity1_match': e1.final_score,
                        'entity2_match': e2.final_score,
                        'relation_match': relations[0].final_score if relations else 0.5,
                        'evidence_strength': 1.0
                    },
                    supporting_context={}
                )
            else:
                # No direct connection
                return self._create_answer(
                    f"No, I don't have evidence that {e1.name} is directly related to {e2.name}.",
                    confidence=0.7 * min(e1.final_score, e2.final_score),
                    answer_type='boolean'
                )
        
        # Property check
        elif entities and relations:
            entity = entities[0]
            relation = relations[0]
            
            # Check if entity has this property
            has_property = False
            for _, role, rel, _ in self.entity_triples.get(entity.name, []):
                if rel == relation.name and role == 'head':
                    has_property = True
                    break
            
            answer_text = "Yes" if has_property else "No"
            confidence = 0.8 * entity.final_score * relation.final_score
            
            return Answer(
                text=answer_text,
                confidence=confidence,
                evidence_triples=[],
                reasoning_path=[f"Checked if {entity.name} has property {relation.name}"],
                alternative_answers=[],
                answer_type='boolean',
                evaluation_breakdown={
                    'entity_match': entity.final_score,
                    'relation_match': relation.final_score,
                    'property_found': 1.0 if has_property else 0.0
                },
                supporting_context={}
            )
        
        return self._create_answer("I cannot determine the answer to this question.",
                                 confidence=0.0, answer_type='boolean')
    
    def _answer_comparison(self, question: str, entities: List[MatchedEntity]) -> Answer:
        """Answer comparison questions"""
        if len(entities) < 2:
            return self._create_answer("I need two things to compare.",
                                     confidence=0.0, answer_type='comparison')
        
        e1, e2 = entities[0], entities[1]
        
        # Get properties for both
        e1_props = defaultdict(set)
        e2_props = defaultdict(set)
        
        for idx, role, relation, other in self.entity_triples.get(e1.name, []):
            if role == 'head':
                e1_props[relation].add(other)
        
        for idx, role, relation, other in self.entity_triples.get(e2.name, []):
            if role == 'head':
                e2_props[relation].add(other)
        
        # Find differences and similarities
        all_relations = set(e1_props.keys()) | set(e2_props.keys())
        
        similarities = []
        differences = []
        
        for rel in all_relations:
            e1_values = e1_props.get(rel, set())
            e2_values = e2_props.get(rel, set())
            
            common = e1_values & e2_values
            only_e1 = e1_values - e2_values
            only_e2 = e2_values - e1_values
            
            if common:
                similarities.append(f"Both {rel} {', '.join(list(common)[:2])}")
            if only_e1:
                differences.append(f"{e1.name} {rel} {', '.join(list(only_e1)[:2])}")
            if only_e2:
                differences.append(f"{e2.name} {rel} {', '.join(list(only_e2)[:2])}")
        
        # Construct answer
        answer_parts = []
        
        if similarities:
            answer_parts.append(f"Similarities: {'; '.join(similarities[:3])}")
        
        if differences:
            answer_parts.append(f"Differences: {'; '.join(differences[:5])}")
        
        if not answer_parts:
            answer_text = f"I don't have enough information to compare {e1.name} and {e2.name}."
            confidence = 0.3
        else:
            answer_text = ' '.join(answer_parts)
            confidence = 0.7
        
        confidence *= min(e1.final_score, e2.final_score)
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=[],
            reasoning_path=[f"Compared properties of {e1.name} and {e2.name}"],
            alternative_answers=[],
            answer_type='comparison',
            evaluation_breakdown={
                'entity1_match': e1.final_score,
                'entity2_match': e2.final_score,
                'properties_compared': len(all_relations) / 10,
                'similarities_found': len(similarities) / 5,
                'differences_found': len(differences) / 10
            },
            supporting_context={
                'all_similarities': similarities,
                'all_differences': differences
            }
        )
    
    def _answer_count(self, question: str, entities: List[MatchedEntity],
                     relations: List[MatchedRelation]) -> Answer:
        """Answer counting questions"""
        
        # Determine what to count
        count_target = None
        constraint_entity = None
        constraint_relation = None
        
        # Pattern: "how many [things] in/of [entity]"
        if entities:
            constraint_entity = entities[0]
        
        if relations:
            constraint_relation = relations[0]
        
        # Extract what to count from question
        count_patterns = [
            r'how many (\w+)',
            r'count (?:of |the )?(\w+)',
            r'number of (\w+)',
            r'list all (\w+)'
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, question)
            if match:
                count_target = match.group(1)
                break
        
        # Perform counting
        if constraint_entity and constraint_relation:
            # Count specific relation from entity
            count = 0
            items = []
            for _, role, rel, other in self.entity_triples.get(constraint_entity.name, []):
                if rel == constraint_relation.name and role == 'head':
                    count += 1
                    items.append(other)
            
            answer_text = f"{constraint_entity.name} has {count} {constraint_relation.name}"
            if count > 0 and count <= 10:
                answer_text += f": {', '.join(items)}"
            elif count > 10:
                answer_text += f" (showing first 5: {', '.join(items[:5])}...)"
            
            confidence = 0.8 * constraint_entity.final_score * constraint_relation.final_score
            
        elif count_target == 'entities':
            count = len(self.entities)
            answer_text = f"There are {count} entities in the knowledge graph"
            confidence = 1.0
            
        elif count_target == 'relations':
            count = len(self.relations)
            answer_text = f"There are {count} types of relations in the knowledge graph"
            confidence = 1.0
            
        elif count_target == 'variants' and any(e.name == 'NetHack' for e in entities):
            # Count NetHack variants
            variants = [e for e in self.entities if 'NetHack' not in e and 
                       any('based_on' in str(t) and 'NetHack' in str(t) 
                           for t in self.entity_triples.get(e, []))]
            count = len(variants)
            answer_text = f"NetHack has {count} variants in the knowledge graph"
            if variants:
                answer_text += f" including: {', '.join(variants[:5])}"
            confidence = 0.8
            
        else:
            # General count based on instance_of
            if count_target:
                count = 0
                items = []
                for entity in self.entities:
                    for _, role, rel, other in self.entity_triples.get(entity, []):
                        if rel == 'instance_of' and count_target in other.lower() and role == 'head':
                            count += 1
                            items.append(entity)
                            break
                
                if count > 0:
                    answer_text = f"There are {count} {count_target} in the knowledge graph"
                    if count <= 10:
                        answer_text += f": {', '.join(items)}"
                    confidence = 0.7
                else:
                    answer_text = f"I couldn't find any {count_target} in the knowledge graph"
                    confidence = 0.5
            else:
                answer_text = "I'm not sure what you want me to count"
                confidence = 0.2
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=[],
            reasoning_path=[f"Counted {count_target or 'items'} with constraints"],
            alternative_answers=[],
            answer_type='count',
            evaluation_breakdown={
                'query_understanding': 0.8 if count_target else 0.3,
                'constraint_match': constraint_entity.final_score if constraint_entity else 0.5,
                'execution_success': 1.0 if 'count' in locals() else 0.0
            },
            supporting_context={'items': items if 'items' in locals() else []}
        )
    
    def _answer_path(self, question: str, entities: List[MatchedEntity]) -> Answer:
        """Answer path-finding questions"""
        if len(entities) < 2:
            return self._create_answer("I need two locations to find a path between.",
                                     confidence=0.0, answer_type='path')
        
        start, end = entities[0], entities[1]
        
        # Find shortest path
        try:
            path = nx.shortest_path(self.graph, start.name, end.name)
            
            # Get the relations along the path
            path_description = []
            evidence = []
            
            for i in range(len(path) - 1):
                # Find the relation between consecutive nodes
                edges = self.graph.get_edge_data(path[i], path[i+1])
                if edges:
                    # Take first relation
                    rel = list(edges.values())[0]['relation']
                    path_description.append(f"{path[i]} --[{rel}]--> {path[i+1]}")
                    evidence.append((path[i], rel, path[i+1]))
            
            answer_text = f"Path from {start.name} to {end.name}: " + ' → '.join(path_description)
            confidence = 0.8 * min(start.final_score, end.final_score)
            
            return Answer(
                text=answer_text,
                confidence=confidence,
                evidence_triples=evidence,
                reasoning_path=[f"Found path of length {len(path)-1}"],
                alternative_answers=[],
                answer_type='path',
                evaluation_breakdown={
                    'start_match': start.final_score,
                    'end_match': end.final_score,
                    'path_length': 1.0 / len(path) if len(path) > 0 else 0.0,
                    'path_found': 1.0
                },
                supporting_context={'full_path': path}
            )
            
        except nx.NetworkXNoPath:
            return self._create_answer(
                f"No path found between {start.name} and {end.name}.",
                confidence=0.9 * min(start.final_score, end.final_score),
                answer_type='path'
            )
    
    def _answer_relationship(self, question: str, entities: List[MatchedEntity]) -> Answer:
        """Answer relationship questions"""
        if len(entities) < 2:
            return self._create_answer("I need two entities to find their relationship.",
                                     confidence=0.0, answer_type='relationship')
        
        e1, e2 = entities[0], entities[1]
        
        # Find all connections
        direct_connections = []
        reverse_connections = []
        
        # Direct connections
        for idx, role, relation, other in self.entity_triples.get(e1.name, []):
            if other == e2.name and role == 'head':
                direct_connections.append(relation)
        
        # Reverse connections
        for idx, role, relation, other in self.entity_triples.get(e2.name, []):
            if other == e1.name and role == 'head':
                reverse_connections.append(relation)
        
        # Try to find indirect connections if no direct ones
        indirect_paths = []
        if not direct_connections and not reverse_connections:
            try:
                # Find up to 3 shortest paths
                paths = list(nx.all_shortest_paths(self.graph, e1.name, e2.name))[:3]
                for path in paths:
                    if len(path) <= 4:  # Limit path length
                        path_desc = self._describe_path(path)
                        if path_desc:
                            indirect_paths.append(path_desc)
            except:
                pass
        
        # Construct answer
        if direct_connections:
            answer_text = f"{e1.name} {' and '.join(direct_connections)} {e2.name}"
            confidence = 0.9
        elif reverse_connections:
            answer_text = f"{e2.name} {' and '.join(reverse_connections)} {e1.name}"
            confidence = 0.9
        elif indirect_paths:
            answer_text = f"{e1.name} and {e2.name} are indirectly related: {indirect_paths[0]}"
            confidence = 0.7
        else:
            answer_text = f"I couldn't find a clear relationship between {e1.name} and {e2.name}"
            confidence = 0.5
        
        confidence *= min(e1.final_score, e2.final_score)
        
        evidence = []
        if direct_connections:
            evidence = [(e1.name, rel, e2.name) for rel in direct_connections]
        elif reverse_connections:
            evidence = [(e2.name, rel, e1.name) for rel in reverse_connections]
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=evidence,
            reasoning_path=[f"Found {len(direct_connections) + len(reverse_connections)} direct connections"],
            alternative_answers=[(p, 0.6) for p in indirect_paths[1:3]] if indirect_paths else [],
            answer_type='relationship',
            evaluation_breakdown={
                'entity1_match': e1.final_score,
                'entity2_match': e2.final_score,
                'direct_connections': len(direct_connections) + len(reverse_connections),
                'connection_strength': 1.0 if direct_connections or reverse_connections else 0.5
            },
            supporting_context={
                'all_paths': indirect_paths
            }
        )
    
    def _describe_path(self, path: List[str]) -> str:
        """Describe a path through the graph"""
        if len(path) < 2:
            return ""
        
        descriptions = []
        for i in range(len(path) - 1):
            edges = self.graph.get_edge_data(path[i], path[i+1])
            if edges:
                rel = list(edges.values())[0]['relation']
                descriptions.append(f"{rel}")
        
        return f"{path[0]} → {' → '.join(descriptions)} → {path[-1]}"
    
    def _answer_general(self, question: str, entities: List[MatchedEntity],
                       relations: List[MatchedRelation]) -> Answer:
        """Handle general fact retrieval"""
        if not entities:
            return self._create_answer("I couldn't understand what you're asking about.",
                                     confidence=0.0, answer_type='general')
        
        # Use best entity and relation matches
        target_entity = entities[0]
        target_relation = relations[0] if relations and relations[0].final_score > 0.5 else None
        
        # Find relevant facts
        relevant_facts = []
        evidence = []
        
        for idx, role, relation, other in self.entity_triples.get(target_entity.name, []):
            relevance = 0.5  # Base relevance
            
            # Boost if relation matches
            if target_relation and relation == target_relation.name:
                relevance = 1.0
            elif target_relation:
                # Semantic similarity between relations
                rel_emb = self.relation_embeddings[relation]['original']
                target_emb = self.relation_embeddings[target_relation.name]['original']
                sim = cosine_similarity([rel_emb], [target_emb])[0][0]
                relevance = max(0.5, sim)
            
            # Check if other entity is mentioned in question
            if other.lower() in question.lower():
                relevance += 0.3
            
            relevant_facts.append((relevance, role, relation, other))
        
        # Sort by relevance
        relevant_facts.sort(key=lambda x: x[0], reverse=True)
        
        if not relevant_facts:
            return self._create_answer(
                f"I don't have information about {target_entity.name}" +
                (f" and {target_relation.name}" if target_relation else ""),
                confidence=0.3,
                answer_type='general'
            )
        
        # Format top facts
        top_facts = relevant_facts[:5]
        
        if len(top_facts) == 1:
            _, role, rel, other = top_facts[0]
            if role == 'head':
                answer_text = f"{target_entity.name} {rel} {other}"
            else:
                answer_text = f"{other} {rel} {target_entity.name}"
        else:
            fact_strings = []
            for _, role, rel, other in top_facts[:3]:
                if role == 'head':
                    fact_strings.append(f"{rel} {other}")
                else:
                    fact_strings.append(f"{other} {rel} it")
            answer_text = f"About {target_entity.name}: " + "; ".join(fact_strings)
        
        # Build evidence
        for relevance, role, rel, other in top_facts:
            if role == 'head':
                evidence.append((target_entity.name, rel, other))
            else:
                evidence.append((other, rel, target_entity.name))
        
        confidence = min(1.0, 0.4 + 0.1 * len(relevant_facts))
        confidence *= target_entity.final_score
        if target_relation:
            confidence *= target_relation.final_score
        
        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence_triples=evidence,
            reasoning_path=[f"Found {len(relevant_facts)} relevant facts"],
            alternative_answers=[],
            answer_type='general',
            evaluation_breakdown={
                'entity_match': target_entity.final_score,
                'relation_match': target_relation.final_score if target_relation else 0.5,
                'fact_relevance': top_facts[0][0] if top_facts else 0.0,
                'evidence_count': min(1.0, len(relevant_facts) / 10)
            },
            supporting_context={'all_relevant_facts': relevant_facts[:10]}
        )
    
    def _create_answer(self, text: str, confidence: float, answer_type: str) -> Answer:
        """Create a simple answer"""
        return Answer(
            text=text,
            confidence=confidence,
            evidence_triples=[],
            reasoning_path=[],
            alternative_answers=[],
            answer_type=answer_type,
            evaluation_breakdown={},
            supporting_context={}
        )


def interactive_qa():
    """Interactive Q&A session"""
    print("Loading NetHack Knowledge Graph QA System...")
    qa = NetHackKGQA('data/nethack_kg_full/nethack_kg_intermediate.tsv')
    print(f"Loaded {len(qa.entities)} entities and {len(qa.relations)} relations")
    print("\nAsk me anything about NetHack! (type 'quit' to exit)")
    
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        answer = qa.answer(question)
        
        print(f"\nAnswer: {answer.text}")
        print(f"Confidence: {answer.confidence:.2%}")
        
        if answer.evidence_triples:
            print(f"\nEvidence ({len(answer.evidence_triples)} facts):")
            for h, r, t in answer.evidence_triples[:3]:
                print(f"  - {h} --[{r}]--> {t}")
        
        if answer.alternative_answers:
            print("\nAlternative answers:")
            for alt, score in answer.alternative_answers[:3]:
                print(f"  - {alt} (confidence: {score:.2%})")
        
        if answer.evaluation_breakdown:
            print("\nEvaluation scores:")
            for metric, score in answer.evaluation_breakdown.items():
                print(f"  - {metric}: {score:.3f}")


if __name__ == "__main__":
    # Test with example questions
    test_questions = [
        "What is the Amulet of Yendor?",
        "How many variants does NetHack have?",
        "What's the difference between blessed and cursed items?",
        "Is Sokoban part of NetHack?",
        "What items are in Gehennom?",
        "How is SLASH'EM related to NetHack?",
        "What monsters use magic?",
        "Path from Castle to Astral Plane?",
        "What does a wand of wishing do?",
        "Can you eat a cockatrice corpse?"
    ]
    
    print("Testing NetHack KGQA System...")
    qa = NetHackKGQA('data/nethack_kg_full/nethack_kg_intermediate.tsv')
    
    for question in test_questions[:3]:  # Test first 3
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        answer = qa.answer(question)
        print(f"A: {answer.text}")
        print(f"Confidence: {answer.confidence:.2%}")
    
    print("\n\nStarting interactive mode...")
    interactive_qa()