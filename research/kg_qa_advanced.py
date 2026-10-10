"""
Advanced KG Question Answering with UltraQuery for complex queries
"""

import torch
from typing import List, Tuple, Dict, Optional, Set
from ultra.models import Ultra
from ultra.tasks import build_relation_graph
from ultra.query_utils import *
from torch_geometric.data import Data
import re


class AdvancedKGQA:
    """Advanced wrapper supporting complex logical queries using UltraQuery"""
    
    def __init__(self, checkpoint_path: str = "ultra/ckpts/ultraquery.pth", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.kg_data = None
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        self.checkpoint_path = checkpoint_path
        
    def parse_natural_query(self, query: str) -> Optional[Dict]:
        """
        Parse natural language query into logical query structure
        
        Supported patterns:
        - Single hop: "Find all X that have relation R with Y"
        - Path queries: "Find all X that R1 to something that R2 to Y"
        - Intersection: "Find all X that both R1 to Y and R2 to Z"
        - Union: "Find all X that either R1 to Y or R2 to Z"
        - Negation: "Find all X that R to Y but not R to Z"
        """
        query = query.lower().strip()
        
        # Single hop pattern
        if "find all" in query and "that" in query:
            # Extract pattern
            if "both" in query and "and" in query:
                # Intersection query
                return self._parse_intersection_query(query)
            elif "either" in query and "or" in query:
                # Union query
                return self._parse_union_query(query)
            elif "but not" in query:
                # Negation query
                return self._parse_negation_query(query)
            else:
                # Simple or path query
                return self._parse_path_query(query)
                
        return None
    
    def _parse_path_query(self, query: str) -> Optional[Dict]:
        """Parse path queries like 'Find all X that R1 to something that R2 to Y'"""
        # Extract entities and relations
        pattern = r"find all (\w+) that (\w+) to (?:something that (\w+) to )?(\w+)"
        match = re.search(pattern, query)
        
        if match:
            var = match.group(1)  # Variable to find
            rel1 = match.group(2)
            rel2 = match.group(3)
            target = match.group(4)
            
            if rel2:
                # 2-hop path query (2p)
                return {
                    "type": "2p",
                    "target_var": var,
                    "anchor": target,
                    "relations": [rel1, rel2]
                }
            else:
                # 1-hop query (1p)
                return {
                    "type": "1p", 
                    "target_var": var,
                    "anchor": target,
                    "relation": rel1
                }
                
        return None
    
    def _parse_intersection_query(self, query: str) -> Optional[Dict]:
        """Parse intersection queries"""
        # Pattern: find all X that both R1 to Y and R2 to Z
        pattern = r"find all (\w+) that both (\w+) to (\w+) and (\w+) to (\w+)"
        match = re.search(pattern, query)
        
        if match:
            return {
                "type": "2i",
                "target_var": match.group(1),
                "branches": [
                    {"relation": match.group(2), "anchor": match.group(3)},
                    {"relation": match.group(4), "anchor": match.group(5)}
                ]
            }
            
        return None
    
    def _parse_union_query(self, query: str) -> Optional[Dict]:
        """Parse union queries"""
        # Pattern: find all X that either R1 to Y or R2 to Z
        pattern = r"find all (\w+) that either (\w+) to (\w+) or (\w+) to (\w+)"
        match = re.search(pattern, query)
        
        if match:
            return {
                "type": "2u",
                "target_var": match.group(1),
                "branches": [
                    {"relation": match.group(2), "anchor": match.group(3)},
                    {"relation": match.group(4), "anchor": match.group(5)}
                ]
            }
            
        return None
    
    def _parse_negation_query(self, query: str) -> Optional[Dict]:
        """Parse negation queries"""
        # Pattern: find all X that R1 to Y but not R2 to Z
        pattern = r"find all (\w+) that (\w+) to (\w+) but not (\w+) to (\w+)"
        match = re.search(pattern, query)
        
        if match:
            return {
                "type": "2in",  # 2-branch intersection with negation
                "target_var": match.group(1),
                "positive": {"relation": match.group(2), "anchor": match.group(3)},
                "negative": {"relation": match.group(4), "anchor": match.group(5)}
            }
            
        return None
    
    def execute_logical_query(self, query_struct: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Execute a logical query structure and return top-k results
        
        Args:
            query_struct: Dictionary describing the query structure
            top_k: Number of top results to return
            
        Returns:
            List of (entity_name, score) tuples
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        query_type = query_struct.get("type")
        
        # Convert to model input format based on query type
        if query_type == "1p":
            # Single hop query
            anchor = self.entity2id.get(query_struct["anchor"])
            relation = self.relation2id.get(query_struct["relation"])
            
            if anchor is None or relation is None:
                return []
                
            # Run standard link prediction
            return self._run_link_prediction(anchor, relation, top_k)
            
        elif query_type == "2p":
            # 2-hop path query
            anchor = self.entity2id.get(query_struct["anchor"])
            relations = [self.relation2id.get(r) for r in query_struct["relations"]]
            
            if anchor is None or None in relations:
                return []
                
            return self._run_path_query(anchor, relations, top_k)
            
        elif query_type == "2i":
            # Intersection query
            return self._run_intersection_query(query_struct, top_k)
            
        elif query_type == "2u":
            # Union query
            return self._run_union_query(query_struct, top_k)
            
        elif query_type == "2in":
            # Negation query
            return self._run_negation_query(query_struct, top_k)
            
        return []
    
    def _run_link_prediction(self, head_idx: int, rel_idx: int, top_k: int) -> List[Tuple[str, float]]:
        """Run simple link prediction query"""
        with torch.no_grad():
            batch = torch.tensor([[head_idx, 0, rel_idx]], dtype=torch.long)
            pred = self.model(self.kg_data, batch.to(self.device))
            scores = pred.squeeze()
            
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
            
            results = []
            for idx, score in zip(top_indices.cpu(), top_scores.cpu()):
                if idx < len(self.id2entity):
                    results.append((self.id2entity[idx], float(score)))
                    
        return results
    
    def _run_path_query(self, anchor: int, relations: List[int], top_k: int) -> List[Tuple[str, float]]:
        """Execute multi-hop path query"""
        # For 2-hop: find intermediate nodes first, then final answers
        with torch.no_grad():
            # First hop
            batch1 = torch.tensor([[anchor, 0, relations[0]]], dtype=torch.long)
            pred1 = self.model(self.kg_data, batch1.to(self.device))
            
            # Get top intermediate nodes
            intermediate_scores, intermediate_nodes = torch.topk(pred1.squeeze(), min(50, len(pred1.squeeze())))
            
            # Second hop from each intermediate node
            final_scores = torch.zeros(len(self.entity2id))
            
            for node, score1 in zip(intermediate_nodes, intermediate_scores):
                batch2 = torch.tensor([[int(node), 0, relations[1]]], dtype=torch.long)
                pred2 = self.model(self.kg_data, batch2.to(self.device))
                
                # Combine scores (using fuzzy logic)
                final_scores += score1 * pred2.squeeze()
            
            # Get top-k final results
            top_scores, top_indices = torch.topk(final_scores, min(top_k, len(final_scores)))
            
            results = []
            for idx, score in zip(top_indices, top_scores):
                if idx < len(self.id2entity):
                    results.append((self.id2entity[int(idx)], float(score)))
                    
        return results
    
    def _run_intersection_query(self, query_struct: Dict, top_k: int) -> List[Tuple[str, float]]:
        """Execute intersection query (find entities satisfying both conditions)"""
        results1 = self._get_branch_results(query_struct["branches"][0])
        results2 = self._get_branch_results(query_struct["branches"][1])
        
        # Intersection using min (fuzzy AND)
        scores = {}
        for ent1, score1 in results1:
            for ent2, score2 in results2:
                if ent1 == ent2:
                    scores[ent1] = min(score1, score2)
        
        # Sort and return top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def _run_union_query(self, query_struct: Dict, top_k: int) -> List[Tuple[str, float]]:
        """Execute union query (find entities satisfying either condition)"""
        results1 = self._get_branch_results(query_struct["branches"][0])
        results2 = self._get_branch_results(query_struct["branches"][1])
        
        # Union using max (fuzzy OR)
        scores = {}
        for ent, score in results1:
            scores[ent] = score
        for ent, score in results2:
            scores[ent] = max(scores.get(ent, 0), score)
        
        # Sort and return top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def _run_negation_query(self, query_struct: Dict, top_k: int) -> List[Tuple[str, float]]:
        """Execute negation query (find entities satisfying positive but not negative)"""
        positive_results = self._get_branch_results(query_struct["positive"])
        negative_results = self._get_branch_results(query_struct["negative"])
        
        # Create set of negative entities
        negative_entities = {ent for ent, _ in negative_results}
        
        # Filter positive results
        filtered_results = [(ent, score) for ent, score in positive_results 
                          if ent not in negative_entities]
        
        return filtered_results[:top_k]
    
    def _get_branch_results(self, branch: Dict) -> List[Tuple[str, float]]:
        """Get results for a query branch"""
        anchor = self.entity2id.get(branch["anchor"])
        relation = self.relation2id.get(branch["relation"])
        
        if anchor is None or relation is None:
            return []
            
        return self._run_link_prediction(anchor, relation, top_k=100)


# Example natural language interface
def natural_language_qa(kg_qa: AdvancedKGQA, question: str) -> List[Tuple[str, float]]:
    """
    Answer natural language questions about the KG
    
    Examples:
    - "Find all people that work at Google"
    - "Find all people that friend of someone that works at Google"
    - "Find all people that both live in NYC and work at Google"
    - "Find all people that either live in NYC or work at Google"
    - "Find all people that work at Google but not live in SF"
    """
    # Parse natural language to logical query
    query_struct = kg_qa.parse_natural_query(question)
    
    if query_struct:
        # Execute logical query
        results = kg_qa.execute_logical_query(query_struct)
        return results
    else:
        # Fallback to simple pattern matching
        # Could be extended with more sophisticated NLP
        return []


if __name__ == "__main__":
    # Example usage
    qa = AdvancedKGQA()
    
    # Example queries
    print("Example natural language queries:")
    print("- Find all people that work at Google")
    print("- Find all people that friend of someone that works at Google")
    print("- Find all people that both live in NYC and work at Google")