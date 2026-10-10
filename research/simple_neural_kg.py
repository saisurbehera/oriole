#!/usr/bin/env python3
"""
Simple neural KG link prediction - train from scratch without ULTRA dependencies
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import time
from tqdm import tqdm

class SimpleKGDataset(Dataset):
    def __init__(self, kg_df, entity2id, relation2id, negative_sampling=True):
        self.triples = []
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.negative_sampling = negative_sampling
        
        # Build positive triples
        for _, row in kg_df.iterrows():
            if row['head'] in entity2id and row['tail'] in entity2id:
                h_id = entity2id[row['head']]
                r_id = relation2id[row['type']]
                t_id = entity2id[row['tail']]
                self.triples.append((h_id, r_id, t_id, 1))  # positive
        
        # Build negative triples
        if negative_sampling:
            self.true_triples = set([(h, r, t) for h, r, t, _ in self.triples])
            neg_triples = []
            
            for h, r, t, _ in self.triples:
                # Sample negative tail
                while True:
                    neg_t = np.random.randint(len(entity2id))
                    if (h, r, neg_t) not in self.true_triples:
                        neg_triples.append((h, r, neg_t, 0))
                        break
                
                # Sample negative head
                while True:
                    neg_h = np.random.randint(len(entity2id))
                    if (neg_h, r, t) not in self.true_triples:
                        neg_triples.append((neg_h, r, t, 0))
                        break
            
            self.triples.extend(neg_triples)
        
        print(f"Dataset: {len(self.triples)} triples")
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        h, r, t, label = self.triples[idx]
        return torch.tensor([h, r, t]), torch.tensor(label, dtype=torch.float)

class SimpleKGModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # MLP for scoring
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 1)
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, triples):
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        
        # Get embeddings
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        
        # Concatenate and score
        combined = torch.cat([h_emb, r_emb, t_emb], dim=1)
        score = self.mlp(combined)
        
        return score.squeeze()

class SimpleKGTrainer:
    def __init__(self, train_path="data/nethack_kg_train_proper.tsv"):
        print("🧠 Simple Neural KG Link Prediction")
        print("="*50)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data
        self.train_kg = pd.read_csv(train_path, sep='\t')
        print(f"Training set: {len(self.train_kg)} triples")
        
        # Build mappings
        self._build_mappings()
        
        # Create datasets
        self.train_dataset = SimpleKGDataset(self.train_kg, self.entity2id, self.relation2id)
        self.train_loader = DataLoader(self.train_dataset, batch_size=1024, shuffle=True)
        
        # Initialize model
        self.model = SimpleKGModel(len(self.entities), len(self.relations)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCEWithLogitsLoss()
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Build evaluation indices
        self._build_eval_indices()
    
    def _build_mappings(self):
        """Build entity and relation mappings"""
        entities = set()
        relations = set()
        
        for _, row in self.train_kg.iterrows():
            entities.add(row['head'])
            entities.add(row['tail'])
            relations.add(row['type'])
        
        self.entities = sorted(list(entities))
        self.relations = sorted(list(relations))
        
        self.entity2id = {e: i for i, e in enumerate(self.entities)}
        self.relation2id = {r: i for i, r in enumerate(self.relations)}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.id2relation = {i: r for r, i in self.relation2id.items()}
        
        print(f"Entities: {len(self.entities)}, Relations: {len(self.relations)}")
    
    def _build_eval_indices(self):
        """Build indices for evaluation"""
        self.relation_candidates = defaultdict(list)
        
        for _, row in self.train_kg.iterrows():
            if row['head'] in self.entity2id and row['tail'] in self.entity2id:
                rel = row['type']
                tail_id = self.entity2id[row['tail']]
                self.relation_candidates[rel].append(tail_id)
        
        # Remove duplicates
        for rel in self.relation_candidates:
            self.relation_candidates[rel] = list(set(self.relation_candidates[rel]))
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_triples, batch_labels in pbar:
            batch_triples = batch_triples.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            scores = self.model(batch_triples)
            loss = self.criterion(scores, batch_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def predict(self, head, relation, max_candidates=10):
        """Predict candidates for a query"""
        self.model.eval()
        
        if head not in self.entity2id or relation not in self.relation2id:
            return []
        
        head_id = self.entity2id[head]
        rel_id = self.relation2id[relation]
        
        # Get candidate entities
        if relation in self.relation_candidates:
            candidates = self.relation_candidates[relation][:100]
        else:
            candidates = list(range(min(len(self.entities), 100)))
        
        # Score candidates
        scores = []
        with torch.no_grad():
            for tail_id in candidates:
                triple = torch.tensor([[head_id, rel_id, tail_id]]).to(self.device)
                score = self.model(triple)
                scores.append((tail_id, score.item()))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        return [self.id2entity[tid] for tid, _ in scores[:max_candidates]]
    
    def evaluate(self, test_path, max_queries=50):
        """Evaluate on test set"""
        test_kg = pd.read_csv(test_path, sep='\t')
        test_sample = test_kg.sample(n=min(max_queries, len(test_kg)), random_state=42)
        
        mrr_scores = []
        hits_at_10 = []
        
        for _, row in test_sample.iterrows():
            head, rel, tail = row['head'], row['type'], row['tail']
            
            if head not in self.entity2id or tail not in self.entity2id:
                continue
            
            predictions = self.predict(head, rel, max_candidates=50)
            
            if tail in predictions:
                rank = predictions.index(tail) + 1
                mrr_scores.append(1.0 / rank)
                hits_at_10.append(1 if rank <= 10 else 0)
            else:
                mrr_scores.append(0.0)
                hits_at_10.append(0)
        
        return np.mean(mrr_scores), np.mean(hits_at_10)
    
    def train(self, num_epochs=10):
        """Main training loop"""
        print("\n" + "="*50)
        print("Training Neural KG Model")
        print("="*50)
        
        best_mrr = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            avg_loss = self.train_epoch(epoch)
            
            # Quick evaluation
            test_path = "data/nethack_kg_test_medium_proper.tsv"
            if os.path.exists(test_path):
                mrr, h10 = self.evaluate(test_path, max_queries=30)
                
                print(f"\nEpoch {epoch}:")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  MRR: {mrr:.3f}")
                print(f"  Hits@10: {h10:.3f}")
                
                if mrr > best_mrr:
                    best_mrr = mrr
                    torch.save(self.model.state_dict(), 'simple_kg_model.pt')
                    print("  ✓ New best model saved!")
        
        print(f"\nBest MRR: {best_mrr:.3f}")
        
        # Final evaluation on all test sets
        print("\nFinal Evaluation:")
        for difficulty in ['easy', 'medium', 'hard']:
            test_path = f"data/nethack_kg_test_{difficulty}_proper.tsv"
            if os.path.exists(test_path):
                mrr, h10 = self.evaluate(test_path, max_queries=50)
                print(f"  {difficulty}: MRR={mrr:.3f}, Hits@10={h10:.3f}")
        
        # Compare with heuristics
        print("\nComparison with Heuristics:")
        print("  Heuristics: Easy=0.555, Medium=0.404, Hard=0.115, Avg=0.358")
        print("  Our Neural Model: See results above")

def main():
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train model
    trainer = SimpleKGTrainer()
    trainer.train(num_epochs=10)
    
    print("\n✅ Neural KG training completed!")

if __name__ == "__main__":
    main()