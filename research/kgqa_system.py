"""
Comprehensive Knowledge Graph Question Answering System
A generalizable KGQA system with semantic matching, contextual relevance, 
confidence scoring, and multi-criteria evaluation
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from abc import ABC, abstractmethod
import json
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle"""
    SINGLE_HOP = "single_hop"          # Direct fact retrieval
    MULTI_HOP = "multi_hop"            # Path queries
    AGGREGATION = "aggregation"        # Count/statistical queries
    COMPARISON = "comparison"          # Comparing entities
    BOOLEAN = "boolean"                # Yes/no questions
    DEFINITION = "definition"          # What is X?
    RELATION_QUERY = "relation_query"  # How are X and Y related?


@dataclass
class Entity:
    """Represents an entity in the KG"""
    id: int
    name: str
    embeddings: Optional[np.ndarray] = None
    aliases: List[str] = None
    context: Dict[str, Any] = None
    importance_score: float = 0.0
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.context is None:
            self.context = {}


@dataclass
class Relation:
    """Represents a relation in the KG"""
    id: int
    name: str
    inverse_name: Optional[str] = None
    embeddings: Optional[np.ndarray] = None
    semantic_type: Optional[str] = None
    confidence: float = 1.0


@dataclass
class Triple:
    """Represents a fact in the KG"""
    head: Entity
    relation: Relation
    tail: Entity
    confidence: float = 1.0
    source: Optional[str] = None
    temporal_info: Optional[Dict] = None


@dataclass
class QueryResult:
    """Result of a query with all evaluation metrics"""
    answer: Any
    confidence: float
    evidence: List[Triple]
    reasoning_path: List[str]
    alternative_answers: List[Tuple[Any, float]]
    query_type: QueryType
    evaluation_scores: Dict[str, float]


class SemanticMatcher:
    """Handles semantic similarity and entity/relation matching"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.entity_embeddings = {}
        self.relation_embeddings = {}
        
    def encode_entities(self, entities: Dict[str, Entity]) -> None:
        """Pre-compute embeddings for all entities"""
        logger.info(f"Encoding {len(entities)} entities...")
        
        # Batch encode for efficiency
        entity_names = list(entities.keys())
        embeddings = self.encoder.encode(entity_names, batch_size=256, show_progress_bar=True)
        
        for name, embedding in zip(entity_names, embeddings):
            entities[name].embeddings = embedding
            self.entity_embeddings[name] = embedding
    
    def encode_relations(self, relations: Dict[str, Relation]) -> None:
        """Pre-compute embeddings for all relations"""
        logger.info(f"Encoding {len(relations)} relations...")
        
        relation_names = list(relations.keys())
        embeddings = self.encoder.encode(relation_names, batch_size=256, show_progress_bar=True)
        
        for name, embedding in zip(relation_names, embeddings):
            relations[name].embeddings = embedding
            self.relation_embeddings[name] = embedding
    
    def find_similar_entities(self, query: str, top_k: int = 10, 
                            threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Find entities similar to query string"""
        query_embedding = self.encoder.encode([query])[0]
        
        similarities = []
        for name, embedding in self.entity_embeddings.items():
            sim = cosine_similarity([query_embedding], [embedding])[0][0]
            if sim >= threshold:
                similarities.append((name, float(sim)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_similar_relations(self, query: str, top_k: int = 5,
                             threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Find relations similar to query string"""
        query_embedding = self.encoder.encode([query])[0]
        
        similarities = []
        for name, embedding in self.relation_embeddings.items():
            sim = cosine_similarity([query_embedding], [embedding])[0][0]
            if sim >= threshold:
                similarities.append((name, float(sim)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def compute_contextual_similarity(self, entity1: Entity, entity2: Entity,
                                    context: Dict[str, Any]) -> float:
        """Compute similarity considering context"""
        base_sim = cosine_similarity([entity1.embeddings], [entity2.embeddings])[0][0]
        
        # Adjust based on context (e.g., same category, temporal proximity)
        context_boost = 0.0
        
        if 'category' in context:
            if entity1.context.get('category') == entity2.context.get('category'):
                context_boost += 0.1
        
        if 'temporal' in context:
            # Boost if entities are from similar time periods
            if abs(entity1.context.get('year', 0) - entity2.context.get('year', 0)) < 5:
                context_boost += 0.05
        
        return min(1.0, base_sim + context_boost)


class MultiCriteriaEvaluator:
    """Evaluates candidate answers using multiple criteria"""
    
    def __init__(self, kg_graph: nx.MultiDiGraph):
        self.graph = kg_graph
        self.pagerank_scores = None
        self.hub_scores = None
        self.authority_scores = None
        self._compute_graph_metrics()
    
    def _compute_graph_metrics(self):
        """Pre-compute graph-based importance metrics"""
        logger.info("Computing graph metrics...")
        
        # PageRank for overall importance
        self.pagerank_scores = nx.pagerank(self.graph)
        
        # HITS algorithm for hubs and authorities
        try:
            self.hub_scores, self.authority_scores = nx.hits(self.graph)
        except:
            self.hub_scores = defaultdict(float)
            self.authority_scores = defaultdict(float)
    
    def evaluate_answer(self, answer: Entity, query_context: Dict,
                       evidence: List[Triple]) -> Dict[str, float]:
        """Evaluate an answer using multiple criteria"""
        scores = {}
        
        # 1. Semantic relevance (from matcher)
        scores['semantic_relevance'] = query_context.get('semantic_score', 0.5)
        
        # 2. Graph importance
        scores['pagerank'] = self.pagerank_scores.get(answer.id, 0.0)
        scores['authority'] = self.authority_scores.get(answer.id, 0.0)
        
        # 3. Evidence strength
        if evidence:
            scores['evidence_strength'] = np.mean([t.confidence for t in evidence])
            scores['evidence_count'] = min(1.0, len(evidence) / 10.0)  # Normalize
        else:
            scores['evidence_strength'] = 0.0
            scores['evidence_count'] = 0.0
        
        # 4. Path length penalty (for multi-hop)
        path_length = query_context.get('path_length', 1)
        scores['path_penalty'] = 1.0 / path_length
        
        # 5. Relation type relevance
        relation_relevance = query_context.get('relation_relevance', 1.0)
        scores['relation_relevance'] = relation_relevance
        
        # 6. Temporal relevance (if applicable)
        if 'query_time' in query_context and answer.context.get('timestamp'):
            time_diff = abs(query_context['query_time'] - answer.context['timestamp'])
            scores['temporal_relevance'] = 1.0 / (1.0 + time_diff / 86400)  # Day decay
        else:
            scores['temporal_relevance'] = 1.0
        
        return scores
    
    def compute_final_score(self, scores: Dict[str, float],
                          weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted final score"""
        if weights is None:
            weights = {
                'semantic_relevance': 0.3,
                'pagerank': 0.1,
                'authority': 0.1,
                'evidence_strength': 0.2,
                'evidence_count': 0.1,
                'path_penalty': 0.1,
                'relation_relevance': 0.05,
                'temporal_relevance': 0.05
            }
        
        final_score = 0.0
        for criterion, score in scores.items():
            final_score += weights.get(criterion, 0.0) * score
        
        return final_score


class QueryProcessor:
    """Processes natural language queries to extract intent and entities"""
    
    def __init__(self, semantic_matcher: SemanticMatcher):
        self.matcher = semantic_matcher
        self.query_patterns = self._load_query_patterns()
    
    def _load_query_patterns(self) -> Dict[QueryType, List[str]]:
        """Load patterns for different query types"""
        return {
            QueryType.DEFINITION: [
                "what is", "what are", "define", "describe", "explain"
            ],
            QueryType.RELATION_QUERY: [
                "how are", "relationship between", "related to", "connection between"
            ],
            QueryType.BOOLEAN: [
                "is", "are", "does", "can", "will", "has", "have"
            ],
            QueryType.AGGREGATION: [
                "how many", "count", "number of", "total", "sum"
            ],
            QueryType.COMPARISON: [
                "difference between", "compare", "versus", "better than"
            ]
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query to extract structured information"""
        query_lower = query.lower()
        
        # Determine query type
        query_type = self._identify_query_type(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Extract relations
        relations = self._extract_relations(query)
        
        # Identify constraints
        constraints = self._extract_constraints(query)
        
        return {
            'original_query': query,
            'query_type': query_type,
            'entities': entities,
            'relations': relations,
            'constraints': constraints,
            'intent': self._extract_intent(query, query_type)
        }
    
    def _identify_query_type(self, query: str) -> QueryType:
        """Identify the type of query"""
        for qtype, patterns in self.query_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return qtype
        
        # Default to single-hop if no pattern matches
        return QueryType.SINGLE_HOP
    
    def _extract_entities(self, query: str) -> List[Tuple[str, float]]:
        """Extract potential entities from query"""
        # Use semantic matcher to find entities
        candidates = []
        
        # Split query into potential entity mentions
        words = query.split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):  # Max 3-word entities
                phrase = ' '.join(words[i:j])
                similar = self.matcher.find_similar_entities(phrase, top_k=3)
                candidates.extend(similar)
        
        # Deduplicate and sort by confidence
        seen = set()
        unique_candidates = []
        for entity, score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if entity not in seen:
                seen.add(entity)
                unique_candidates.append((entity, score))
        
        return unique_candidates[:5]
    
    def _extract_relations(self, query: str) -> List[Tuple[str, float]]:
        """Extract potential relations from query"""
        return self.matcher.find_similar_relations(query, top_k=5)
    
    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract constraints like time, location, etc."""
        constraints = {}
        
        # Extract temporal constraints
        # (Would use more sophisticated NER in production)
        if 'before' in query or 'after' in query or 'during' in query:
            constraints['temporal'] = True
        
        # Extract numerical constraints
        import re
        numbers = re.findall(r'\d+', query)
        if numbers:
            constraints['numerical'] = numbers
        
        return constraints
    
    def _extract_intent(self, query: str, query_type: QueryType) -> str:
        """Extract the primary intent of the query"""
        if query_type == QueryType.DEFINITION:
            return "seeking_definition"
        elif query_type == QueryType.RELATION_QUERY:
            return "exploring_relationships"
        elif query_type == QueryType.BOOLEAN:
            return "verifying_fact"
        elif query_type == QueryType.AGGREGATION:
            return "counting_or_statistics"
        else:
            return "information_retrieval"


class KGQASystem:
    """Main Knowledge Graph Question Answering System"""
    
    def __init__(self, kg_path: str, use_ultra: bool = False):
        self.kg_path = kg_path
        self.use_ultra = use_ultra
        
        # Core components
        self.semantic_matcher = SemanticMatcher()
        self.entities = {}
        self.relations = {}
        self.triples = []
        self.graph = nx.MultiDiGraph()
        
        # Load KG
        self._load_knowledge_graph()
        
        # Initialize components
        self.query_processor = QueryProcessor(self.semantic_matcher)
        self.evaluator = MultiCriteriaEvaluator(self.graph)
        
        # ULTRA integration (if available)
        self.ultra_model = None
        if use_ultra:
            self._initialize_ultra()
    
    def _load_knowledge_graph(self):
        """Load KG from file"""
        logger.info(f"Loading knowledge graph from {self.kg_path}")
        
        # Load triples
        df = pd.read_csv(self.kg_path, sep='\t', header=0, 
                        names=['head', 'relation', 'tail', 'source'])
        
        # Build entities and relations
        entity_id = 0
        relation_id = 0
        
        for _, row in df.iterrows():
            # Get or create entities
            if row['head'] not in self.entities:
                self.entities[row['head']] = Entity(
                    id=entity_id,
                    name=row['head']
                )
                entity_id += 1
            
            if row['tail'] not in self.entities:
                self.entities[row['tail']] = Entity(
                    id=entity_id,
                    name=row['tail']
                )
                entity_id += 1
            
            # Get or create relation
            if row['relation'] not in self.relations:
                self.relations[row['relation']] = Relation(
                    id=relation_id,
                    name=row['relation']
                )
                relation_id += 1
            
            # Create triple
            triple = Triple(
                head=self.entities[row['head']],
                relation=self.relations[row['relation']],
                tail=self.entities[row['tail']],
                source=row.get('source')
            )
            self.triples.append(triple)
            
            # Add to graph
            self.graph.add_edge(
                triple.head.id,
                triple.tail.id,
                relation=triple.relation.name,
                triple=triple
            )
        
        # Compute embeddings
        self.semantic_matcher.encode_entities(self.entities)
        self.semantic_matcher.encode_relations(self.relations)
        
        logger.info(f"Loaded {len(self.entities)} entities, {len(self.relations)} relations, "
                   f"{len(self.triples)} triples")
    
    def _initialize_ultra(self):
        """Initialize ULTRA model for complex reasoning"""
        # Implementation would load ULTRA model
        logger.info("ULTRA integration not implemented in this version")
    
    def answer_question(self, question: str) -> QueryResult:
        """Main entry point for answering questions"""
        logger.info(f"Processing question: {question}")
        
        # Process query
        processed_query = self.query_processor.process_query(question)
        
        # Route to appropriate handler based on query type
        if processed_query['query_type'] == QueryType.SINGLE_HOP:
            return self._handle_single_hop(processed_query)
        elif processed_query['query_type'] == QueryType.MULTI_HOP:
            return self._handle_multi_hop(processed_query)
        elif processed_query['query_type'] == QueryType.AGGREGATION:
            return self._handle_aggregation(processed_query)
        elif processed_query['query_type'] == QueryType.BOOLEAN:
            return self._handle_boolean(processed_query)
        elif processed_query['query_type'] == QueryType.DEFINITION:
            return self._handle_definition(processed_query)
        elif processed_query['query_type'] == QueryType.RELATION_QUERY:
            return self._handle_relation_query(processed_query)
        else:
            return self._handle_general_query(processed_query)
    
    def _handle_single_hop(self, query: Dict) -> QueryResult:
        """Handle single-hop queries"""
        # Get top entity candidates
        entity_candidates = query['entities']
        relation_candidates = query['relations']
        
        if not entity_candidates:
            return self._create_empty_result(QueryType.SINGLE_HOP, 
                                           "No entities found in query")
        
        # Find matching triples
        results = []
        for entity_name, entity_score in entity_candidates[:3]:
            entity = self.entities.get(entity_name)
            if not entity:
                continue
            
            # Find triples with this entity as head
            for triple in self.triples:
                if triple.head.id == entity.id:
                    # Check if relation matches
                    relation_score = 0.5  # Default
                    for rel_name, rel_score in relation_candidates:
                        if rel_name == triple.relation.name:
                            relation_score = rel_score
                            break
                    
                    # Evaluate this answer
                    context = {
                        'semantic_score': entity_score,
                        'relation_relevance': relation_score,
                        'path_length': 1
                    }
                    
                    eval_scores = self.evaluator.evaluate_answer(
                        triple.tail, context, [triple]
                    )
                    final_score = self.evaluator.compute_final_score(eval_scores)
                    
                    results.append((triple, final_score, eval_scores))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        if not results:
            return self._create_empty_result(QueryType.SINGLE_HOP,
                                           "No matching facts found")
        
        # Create result
        best_triple, best_score, eval_scores = results[0]
        alternatives = [(r[0].tail.name, r[1]) for r in results[1:6]]
        
        return QueryResult(
            answer=best_triple.tail.name,
            confidence=best_score,
            evidence=[best_triple],
            reasoning_path=[f"{best_triple.head.name} --[{best_triple.relation.name}]--> {best_triple.tail.name}"],
            alternative_answers=alternatives,
            query_type=QueryType.SINGLE_HOP,
            evaluation_scores=eval_scores
        )
    
    def _handle_multi_hop(self, query: Dict) -> QueryResult:
        """Handle multi-hop queries using path finding"""
        # Implementation would use BFS/DFS or neural methods
        pass
    
    def _handle_aggregation(self, query: Dict) -> QueryResult:
        """Handle counting and aggregation queries"""
        # Implementation would count entities/relations
        pass
    
    def _handle_boolean(self, query: Dict) -> QueryResult:
        """Handle yes/no questions"""
        # Implementation would verify facts
        pass
    
    def _handle_definition(self, query: Dict) -> QueryResult:
        """Handle definition queries"""
        # Implementation would gather all facts about an entity
        pass
    
    def _handle_relation_query(self, query: Dict) -> QueryResult:
        """Handle relationship exploration queries"""
        # Implementation would find paths between entities
        pass
    
    def _handle_general_query(self, query: Dict) -> QueryResult:
        """Handle general queries"""
        return self._handle_single_hop(query)  # Default to single-hop
    
    def _create_empty_result(self, query_type: QueryType, message: str) -> QueryResult:
        """Create an empty result with explanation"""
        return QueryResult(
            answer=None,
            confidence=0.0,
            evidence=[],
            reasoning_path=[message],
            alternative_answers=[],
            query_type=query_type,
            evaluation_scores={}
        )


def main():
    """Example usage"""
    # Initialize system
    kgqa = KGQASystem('data/nethack_kg_full/nethack_kg_intermediate.tsv')
    
    # Example queries
    queries = [
        "What is the Amulet of Yendor?",
        "How is NetHack related to Rogue?",
        "What items are part of NetHack?",
        "Is Sokoban part of NetHack?",
        "How many variants does NetHack have?",
        "What is the difference between blessed and cursed?"
    ]
    
    for query in queries:
        print(f"\nQuestion: {query}")
        result = kgqa.answer_question(query)
        print(f"Answer: {result.answer} (confidence: {result.confidence:.3f})")
        if result.alternative_answers:
            print("Alternative answers:")
            for alt, score in result.alternative_answers[:3]:
                print(f"  - {alt} ({score:.3f})")


if __name__ == "__main__":
    main()