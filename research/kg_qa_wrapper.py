"""
Lightweight wrapper for Knowledge Graph Question Answering using ULTRA
"""

import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
from ultra.datasets import TransductiveDataset
from ultra.models import Ultra
from ultra.tasks import build_relation_graph
from torch_geometric.data import Data
import yaml
from easydict import EasyDict as edict


class KnowledgeGraphQA:
    """Simple wrapper for KG parsing and question answering with ULTRA"""
    
    def __init__(self, checkpoint_path: str = "ultra/ckpts/ultra_4g.pth", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.kg_data = None
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        self.checkpoint_path = checkpoint_path
        
    def parse_kg_from_triples(self, triples: List[Tuple[str, str, str]]):
        """
        Parse KG from list of (head, relation, tail) triples
        
        Args:
            triples: List of (head, relation, tail) string tuples
        """
        # Build entity and relation vocabularies
        entities = set()
        relations = set()
        
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
        
        # Create mappings
        self.entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.relation2id = {r: i for i, r in enumerate(sorted(relations))}
        self.id2relation = {i: r for r, i in self.relation2id.items()}
        
        # Convert triples to indices
        edge_list = []
        edge_type = []
        
        for h, r, t in triples:
            edge_list.append([self.entity2id[h], self.entity2id[t]])
            edge_type.append(self.relation2id[r])
        
        # Create PyG Data object
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
        self.kg_data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(self.entity2id)
        )
        
        # Build relation graph
        self.kg_data = build_relation_graph(self.kg_data)
        
    def parse_kg_from_file(self, file_path: str, delimiter: str = "\t"):
        """Parse KG from TSV/CSV file with (head, relation, tail) format"""
        df = pd.read_csv(file_path, sep=delimiter, header=None, names=['head', 'relation', 'tail'])
        triples = [(row.head, row.relation, row.tail) for _, row in df.iterrows()]
        self.parse_kg_from_triples(triples)
        
    def load_model(self, config_path: str = "ultra/config/transductive/inference.yaml"):
        """Load pre-trained ULTRA model"""
        # Load config
        with open(config_path, 'r') as f:
            config = edict(yaml.safe_load(f))
        
        # Initialize model
        self.model = Ultra(
            config.model.relation_graph.aggregate_func,
            config.model.entity_graph.aggregate_func,
            config.model.entity_graph.num_layer,
            config.model.relation_graph.num_layer,
            config.model.entity_graph.hidden_dim,
            config.model.relation_graph.hidden_dim,
            num_relation=len(self.relation2id) if self.kg_data else 1
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def link_prediction(self, head: str, relation: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Predict tail entities for (head, relation, ?)
        
        Returns:
            List of (entity_name, score) tuples
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        with torch.no_grad():
            # Convert to indices
            h_idx = self.entity2id.get(head)
            r_idx = self.relation2id.get(relation)
            
            if h_idx is None or r_idx is None:
                return []
            
            # Create batch
            batch = torch.tensor([[h_idx, 0, r_idx]], dtype=torch.long)  # dummy tail
            
            # Run inference
            pred = self.model(self.kg_data, batch.to(self.device))
            scores = pred.squeeze()
            
            # Get top-k predictions
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
            
            results = []
            for idx, score in zip(top_indices.cpu(), top_scores.cpu()):
                if idx < len(self.id2entity):
                    results.append((self.id2entity[idx], float(score)))
                    
        return results
    
    def answer_simple_questions(self, question: str) -> List[Tuple[str, float]]:
        """
        Answer simple questions about the KG
        
        Supported patterns:
        - "What is related to X by relation Y?"
        - "Who/What has relation Y with X?"
        """
        question = question.lower().strip('?')
        
        # Pattern 1: What is related to X by relation Y?
        if "related to" in question and "by" in question:
            parts = question.split("related to")[1].split("by")
            if len(parts) == 2:
                entity = parts[0].strip()
                relation = parts[1].strip()
                return self.link_prediction(entity, relation)
        
        # Pattern 2: Who/What has relation Y with X?
        if "has" in question and "with" in question:
            parts = question.split("has")[1].split("with")
            if len(parts) == 2:
                relation = parts[0].strip()
                entity = parts[1].strip()
                # This would require inverse relation prediction
                # For now, return empty
                return []
        
        return []
    
    def get_entity_relations(self, entity: str) -> List[str]:
        """Get all relations connected to an entity"""
        if entity not in self.entity2id:
            return []
            
        e_idx = self.entity2id[entity]
        relations = set()
        
        edge_index = self.kg_data.edge_index
        edge_type = self.kg_data.edge_type
        
        # Find all edges where entity is head
        mask = edge_index[0] == e_idx
        rel_indices = edge_type[mask]
        
        for r_idx in rel_indices:
            relations.add(self.id2relation[int(r_idx)])
            
        return list(relations)


# Example usage
if __name__ == "__main__":
    # Initialize wrapper
    qa = KnowledgeGraphQA()
    
    # Example KG triples
    triples = [
        ("Alice", "friendOf", "Bob"),
        ("Bob", "worksAt", "Google"),
        ("Alice", "livesIn", "NYC"),
        ("Bob", "livesIn", "SF"),
        ("Google", "locatedIn", "SF"),
    ]
    
    # Parse KG
    qa.parse_kg_from_triples(triples)
    
    # Load model
    qa.load_model()
    
    # Make predictions
    results = qa.link_prediction("Alice", "friendOf", top_k=5)
    print(f"Who is Alice friends with? {results}")
    
    # Get entity relations
    relations = qa.get_entity_relations("Bob")
    print(f"Relations for Bob: {relations}")