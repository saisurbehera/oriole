#!/usr/bin/env python3
"""
Train ULTRA from scratch on NetHack KG
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from collections import defaultdict
from easydict import EasyDict as edict
import time
from tqdm import tqdm

# Add ULTRA path
sys.path.append('ultra')
from ultra.models import Ultra
from ultra import tasks

class ULTRATrainer:
    def __init__(self, train_path="data/nethack_kg_train_proper.tsv", 
                 valid_path="data/nethack_kg_test_medium_proper.tsv"):
        print("🚀 Training ULTRA from Scratch on NetHack KG")
        print("="*50)
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load data
        print("\nLoading data...")
        self.train_kg = pd.read_csv(train_path, sep='\t')
        self.valid_kg = pd.read_csv(valid_path, sep='\t')
        print(f"Training set: {len(self.train_kg)} triples")
        print(f"Validation set: {len(self.valid_kg)} triples")
        
        # Build mappings
        self._build_mappings()
        
        # Build PyG data
        self.train_data = self._build_pyg_data(self.train_kg)
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Build negative sampling index
        self._build_negative_sampling_index()
    
    def _build_mappings(self):
        """Build entity and relation mappings"""
        entities = set()
        relations = set()
        
        for df in [self.train_kg, self.valid_kg]:
            for _, row in df.iterrows():
                entities.add(row['head'])
                entities.add(row['tail'])
                relations.add(row['type'])
        
        self.entities = sorted(list(entities))
        self.relations = sorted(list(relations))
        
        self.entity2id = {e: i for i, e in enumerate(self.entities)}
        self.relation2id = {r: i for i, r in enumerate(self.relations)}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.id2relation = {i: r for r, i in self.relation2id.items()}
        
        print(f"\nEntities: {len(self.entities)}, Relations: {len(self.relations)}")
    
    def _build_pyg_data(self, kg_df):
        """Build PyG data from KG dataframe"""
        edge_list = []
        edge_type = []
        
        for _, row in kg_df.iterrows():
            if row['head'] in self.entity2id and row['tail'] in self.entity2id:
                h_id = self.entity2id[row['head']]
                r_id = self.relation2id[row['type']]
                t_id = self.entity2id[row['tail']]
                
                edge_list.append([h_id, t_id])
                edge_type.append(r_id)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
        
        data = Data(
            edge_index=edge_index,
            edge_type=edge_type_tensor,
            num_nodes=len(self.entities)
        )
        
        data.num_relations = len(self.relations)
        
        # Build relation graph
        data = tasks.build_relation_graph(data)
        
        return data.to(self.device)
    
    def _initialize_model(self):
        """Initialize ULTRA model"""
        print("\nInitializing model...")
        
        # Smaller config for faster training
        config = edict({
            'model': {
                'relation_graph': {
                    'class': 'RelNBFNet',
                    'input_dim': 32,
                    'hidden_dims': [32, 32],
                    'message_func': 'distmult',
                    'aggregate_func': 'sum',
                    'short_cut': True,
                    'layer_norm': True
                },
                'entity_graph': {
                    'class': 'EntityNBFNet',
                    'input_dim': 32,
                    'hidden_dims': [32, 32],
                    'message_func': 'distmult',
                    'aggregate_func': 'sum',
                    'short_cut': True,
                    'layer_norm': True
                }
            }
        })
        
        model = Ultra(
            rel_model_cfg=config.model.relation_graph,
            entity_model_cfg=config.model.entity_graph
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model.to(self.device)
    
    def _build_negative_sampling_index(self):
        """Build index for negative sampling"""
        self.true_triples = set()
        
        for _, row in self.train_kg.iterrows():
            if row['head'] in self.entity2id and row['tail'] in self.entity2id:
                h_id = self.entity2id[row['head']]
                r_id = self.relation2id[row['type']]
                t_id = self.entity2id[row['tail']]
                self.true_triples.add((h_id, r_id, t_id))
    
    def _sample_negative(self, head, rel, tail, num_neg=10):
        """Sample negative examples"""
        neg_tails = []
        neg_heads = []
        
        # Sample negative tails
        while len(neg_tails) < num_neg:
            neg_t = np.random.randint(len(self.entities))
            if (head, rel, neg_t) not in self.true_triples:
                neg_tails.append(neg_t)
        
        # Sample negative heads
        while len(neg_heads) < num_neg:
            neg_h = np.random.randint(len(self.entities))
            if (neg_h, rel, tail) not in self.true_triples:
                neg_heads.append(neg_h)
        
        return neg_heads, neg_tails
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        # Sample training triples
        train_triples = []
        for _, row in self.train_kg.iterrows():
            if row['head'] in self.entity2id and row['tail'] in self.entity2id:
                h_id = self.entity2id[row['head']]
                r_id = self.relation2id[row['type']]
                t_id = self.entity2id[row['tail']]
                train_triples.append((h_id, r_id, t_id))
        
        # Shuffle
        np.random.shuffle(train_triples)
        
        batch_size = 256
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(range(0, len(train_triples), batch_size), desc=f"Epoch {epoch}")
        
        for i in pbar:
            batch_triples = train_triples[i:i+batch_size]
            
            # Prepare batch
            pos_batch = []
            neg_batch = []
            
            for h, r, t in batch_triples:
                pos_batch.append([h, r, t])
                
                # Sample negatives
                neg_h, neg_t = self._sample_negative(h, r, t, num_neg=1)
                
                # Negative tail
                neg_batch.append([h, r, neg_t[0]])
                # Negative head
                neg_batch.append([neg_h[0], r, t])
            
            pos_batch = torch.tensor(pos_batch, dtype=torch.long).to(self.device)
            neg_batch = torch.tensor(neg_batch, dtype=torch.long).to(self.device)
            
            # Forward pass
            pos_score = self.model(self.train_data, pos_batch.unsqueeze(0))
            neg_score = self.model(self.train_data, neg_batch.unsqueeze(0))
            
            # Margin ranking loss
            margin = 1.0
            loss = torch.nn.functional.relu(margin - pos_score + neg_score).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def evaluate(self, test_path=None):
        """Evaluate model"""
        self.model.eval()
        
        if test_path:
            test_kg = pd.read_csv(test_path, sep='\t')
        else:
            test_kg = self.valid_kg
        
        # Sample queries
        test_sample = test_kg.sample(n=min(100, len(test_kg)), random_state=42)
        
        mrr_scores = []
        hits_at_10 = []
        
        with torch.no_grad():
            for _, row in test_sample.iterrows():
                head, rel, tail = row['head'], row['type'], row['tail']
                
                if head not in self.entity2id or tail not in self.entity2id:
                    continue
                
                h_id = self.entity2id[head]
                r_id = self.relation2id[rel]
                t_id = self.entity2id[tail]
                
                # Score all entities as tails
                all_candidates = list(range(len(self.entities)))
                batch_queries = [[h_id, r_id, c] for c in all_candidates]
                
                # Score in batches
                batch_size = 1000
                scores = []
                
                for i in range(0, len(batch_queries), batch_size):
                    batch = torch.tensor(batch_queries[i:i+batch_size], dtype=torch.long).to(self.device)
                    batch_scores = self.model(self.train_data, batch.unsqueeze(0))
                    scores.extend(batch_scores.squeeze(0).cpu().numpy().tolist())
                
                # Rank
                scores = np.array(scores)
                rank = (scores > scores[t_id]).sum() + 1
                
                mrr_scores.append(1.0 / rank)
                hits_at_10.append(1 if rank <= 10 else 0)
        
        return np.mean(mrr_scores), np.mean(hits_at_10)
    
    def train(self, num_epochs=10):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        best_mrr = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            avg_loss = self.train_epoch(epoch)
            
            # Evaluate
            mrr, h10 = self.evaluate()
            
            print(f"\nEpoch {epoch}:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  MRR: {mrr:.3f}")
            print(f"  Hits@10: {h10:.3f}")
            
            if mrr > best_mrr:
                best_mrr = mrr
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'mrr': mrr
                }, 'ultra_nethack_best.pt')
                print("  ✓ New best model saved!")
        
        print("\n" + "="*50)
        print(f"Training Complete! Best MRR: {best_mrr:.3f}")
        
        # Final evaluation on all test sets
        print("\nFinal Evaluation:")
        for difficulty in ['easy', 'medium', 'hard']:
            test_path = f"data/nethack_kg_test_{difficulty}_proper.tsv"
            if os.path.exists(test_path):
                mrr, h10 = self.evaluate(test_path)
                print(f"  {difficulty}: MRR={mrr:.3f}, Hits@10={h10:.3f}")

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize trainer
    trainer = ULTRATrainer()
    
    # Train model
    trainer.train(num_epochs=5)  # Start with 5 epochs for quick results
    
    print("\n✅ Training completed!")
    print("Model saved to: ultra_nethack_best.pt")

if __name__ == "__main__":
    main()