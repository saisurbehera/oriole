"""
Process full NetHack wiki to build comprehensive knowledge graph using REBEL
"""

import xml.etree.ElementTree as ET
import re
import json
import os
from typing import List, Dict
import csv
from tqdm import tqdm
from transformers import pipeline
import torch
import gc
import pandas as pd
from collections import defaultdict

class NetHackKGBuilder:
    def __init__(self, xml_path: str, output_dir: str = "data/nethack_kg_full"):
        self.xml_path = xml_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize REBEL model
        print("Initializing REBEL model...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.triplet_extractor = pipeline(
            'text2text-generation',
            model='Babelscape/rebel-large',
            tokenizer='Babelscape/rebel-large',
            device=self.device
        )
        print(f"REBEL model loaded on {self.device}")
        
    def clean_wiki_text(self, text: str) -> str:
        """Clean wiki markup from text"""
        if not text:
            return ""
            
        # Remove wiki templates {{...}}
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        
        # Remove wiki links [[...]]
        text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove wiki formatting
        text = re.sub(r"'''([^']+)'''", r'\1', text)
        text = re.sub(r"''([^']+)''", r'\1', text)
        
        # Remove references
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove categories
        text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_triplets(self, text: str) -> List[Dict]:
        """Parse REBEL output to extract triplets"""
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        
        if subject != '' and relation != '' and object_ != '':
            triplets.append({
                'head': subject.strip(),
                'type': relation.strip(),
                'tail': object_.strip()
            })
        
        return triplets
    
    def process_text_chunk(self, text: str, source: str) -> List[Dict]:
        """Process a text chunk with REBEL"""
        all_triplets = []
        
        try:
            # Skip empty or very short texts
            if not text or len(text.strip()) < 20:
                return []
            
            # First tokenize to check length
            tokenizer = self.triplet_extractor.tokenizer
            
            # Handle extremely long texts by pre-chunking at character level
            if len(text) > 50000:  # Very long text
                char_chunks = []
                chunk_size = 10000
                for i in range(0, len(text), chunk_size - 500):  # Overlap
                    char_chunks.append(text[i:i + chunk_size])
                
                for char_chunk in char_chunks:
                    chunk_triplets = self.process_text_chunk(char_chunk, source)
                    all_triplets.extend(chunk_triplets)
                
                return all_triplets
            
            tokens = tokenizer(text, truncation=False, return_tensors="pt", add_special_tokens=False)
            
            # If too long, process in chunks
            if tokens['input_ids'].shape[1] > 1024:
                # Split into overlapping chunks
                chunk_size = 800  # Smaller to be safe
                stride = 600      # Good overlap
                
                token_ids = tokens['input_ids'][0]
                num_chunks = 0
                
                for i in range(0, len(token_ids), stride):
                    chunk_tokens = token_ids[i:i + chunk_size]
                    
                    # Skip if chunk is too small
                    if len(chunk_tokens) < 50:
                        continue
                        
                    chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    
                    # Process this chunk
                    triplets = self._process_single_chunk(chunk_text, source)
                    all_triplets.extend(triplets)
                    
                    num_chunks += 1
                    # Limit chunks per page to avoid explosion
                    if num_chunks > 20:
                        break
                    
                    # Stop if we've reached the end
                    if i + chunk_size >= len(token_ids):
                        break
            else:
                # Process normally if short enough
                all_triplets = self._process_single_chunk(text, source)
                
            return all_triplets
            
        except Exception as e:
            print(f"Error processing text from {source}: {str(e)[:200]}")
            return []
    
    def _process_single_chunk(self, text: str, source: str) -> List[Dict]:
        """Process a single chunk of text"""
        try:
            # Extract with REBEL
            extraction = self.triplet_extractor(
                text,
                return_tensors=True,
                return_text=False,
                max_length=256,
                truncation=True
            )
            
            # Decode output
            extracted_text = self.triplet_extractor.tokenizer.batch_decode(
                [extraction[0]["generated_token_ids"]]
            )
            
            # Parse triplets
            triplets = self.extract_triplets(extracted_text[0])
            
            # Add source
            for triplet in triplets:
                triplet['source'] = source
                
            return triplets
            
        except Exception as e:
            print(f"Error processing chunk from {source}: {e}")
            return []
    
    def process_wiki_full(self, batch_size: int = 100, max_pages: int = None, resume: bool = True):
        """Process full wiki in batches"""
        
        print(f"Processing wiki from: {self.xml_path}")
        
        # Load existing data if resuming
        all_triplets = []
        unique_triplets = set()
        processed_pages = set()
        
        if resume:
            intermediate_path = os.path.join(self.output_dir, "nethack_kg_intermediate.tsv")
            if os.path.exists(intermediate_path):
                print(f"Resuming from existing file: {intermediate_path}")
                
                # Try to read with source column first
                try:
                    existing_df = pd.read_csv(intermediate_path, sep='\t', header=0)
                    has_source = 'source' in existing_df.columns
                except:
                    # Fallback to 3 columns
                    existing_df = pd.read_csv(intermediate_path, sep='\t', header=0, names=['head', 'relation', 'tail'])
                    has_source = False
                
                for _, row in existing_df.iterrows():
                    triplet = {
                        'head': str(row['head']),
                        'type': str(row['relation']),
                        'tail': str(row['tail']),
                        'source': str(row['source']) if has_source and pd.notna(row['source']) else ''
                    }
                    all_triplets.append(triplet)
                    unique_triplets.add((str(row['head']), str(row['relation']), str(row['tail'])))
                    if has_source and pd.notna(row['source']):
                        processed_pages.add(str(row['source']))
                
                print(f"Loaded {len(all_triplets)} existing triplets")
                if processed_pages:
                    print(f"Found {len(processed_pages)} already processed pages")
        
        # Parse XML
        context = ET.iterparse(self.xml_path, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        page_count = len(processed_pages)
        batch_texts = []
        batch_sources = []
        
        current_page = {}
        
        # Priority pages to definitely include
        priority_titles = {
            "Amulet of Yendor", "Gehennom", "Astral Plane", "Quest",
            "Elbereth", "Oracle", "Sokoban", "Castle", "Medusa",
            "Wizard of Yendor", "Vlad the Impaler", "Altar", "Prayer",
            "Wand of wishing", "Bag of holding", "Artifact", "Ascension",
            "Excalibur", "Mjollnir", "Stormbringer", "Grayswandir",
            "Valkyrie", "Wizard", "Samurai", "Monk", "Knight", "Priest",
            "Dragon", "Demon", "Lich", "Vampire", "Mind flayer",
            "Polymorph", "Teleportation", "Invisibility", "Genocide",
            "Armor class", "Hit points", "Experience", "Dungeon",
            "Alignment", "Luck", "Shop", "Vault", "Throne room",
            "Wand", "Scroll", "Potion", "Ring", "Spellbook",
            "Food", "Corpse", "Intrinsic", "Resistance"
        }
        
        for event, elem in tqdm(context, desc="Parsing wiki"):
            if event == 'start' and elem.tag.endswith('page'):
                current_page = {}
                
            elif event == 'end':
                if elem.tag.endswith('title'):
                    if elem.text:
                        current_page['title'] = elem.text
                        
                elif elem.tag.endswith('text'):
                    if elem.text:
                        current_page['text'] = elem.text
                        
                elif elem.tag.endswith('page'):
                    if current_page.get('title') and current_page.get('text'):
                        title = current_page['title']
                        
                        # Skip if already processed
                        if resume and title in processed_pages:
                            elem.clear()
                            root.clear()
                            continue
                        
                        # Skip meta pages
                        if not any(prefix in title for prefix in ['File:', 'Template:', 'Category:', 'User:', 'Talk:', 'MediaWiki:']):
                            
                            # Clean text - process ENTIRE page content
                            text = self.clean_wiki_text(current_page['text'])
                            
                            # Process ALL content, no truncation
                            if len(text) > 50:  # Skip very short pages
                                batch_texts.append(text)
                                batch_sources.append(title)
                                page_count += 1
                                
                                # Process batch
                                if len(batch_texts) >= batch_size:
                                    print(f"\nProcessing batch of {len(batch_texts)} pages (total processed: {page_count}, existing: {len(processed_pages)})...")
                                    
                                    # Process in smaller chunks to avoid memory issues
                                    chunk_results = []
                                    for text, source in zip(batch_texts, batch_sources):
                                        try:
                                            triplets = self.process_text_chunk(text, source)
                                            chunk_results.extend(triplets)
                                        except Exception as e:
                                            print(f"Error processing {source}: {str(e)[:100]}")
                                            continue
                                    
                                    # Add unique triplets
                                    for triplet in chunk_results:
                                        key = (triplet['head'], triplet['type'], triplet['tail'])
                                        if key not in unique_triplets:
                                            unique_triplets.add(key)
                                            all_triplets.append(triplet)
                                    
                                    print(f"Total unique triplets so far: {len(all_triplets)}")
                                    
                                    # Save intermediate results
                                    self.save_triplets(all_triplets, suffix="_intermediate")
                                    
                                    # Clear batch
                                    batch_texts = []
                                    batch_sources = []
                                    
                                    # Clear GPU memory
                                    if self.device != "cpu":
                                        torch.cuda.empty_cache()
                                        gc.collect()
                                
                                if max_pages and page_count >= max_pages:
                                    break
                    
                    # Clear memory
                    elem.clear()
                    root.clear()
        
        # Process remaining batch
        if batch_texts:
            print(f"\nProcessing final batch of {len(batch_texts)} pages...")
            for text, source in zip(batch_texts, batch_sources):
                triplets = self.process_text_chunk(text, source)
                
                for triplet in triplets:
                    key = (triplet['head'], triplet['type'], triplet['tail'])
                    if key not in unique_triplets:
                        unique_triplets.add(key)
                        all_triplets.append(triplet)
        
        print(f"\nTotal pages processed: {page_count}")
        print(f"Total unique triplets extracted: {len(all_triplets)}")
        
        # Save final results
        self.save_triplets(all_triplets)
        
        return all_triplets
    
    def save_triplets(self, triplets: List[Dict], suffix: str = ""):
        """Save triplets in multiple formats"""
        
        # Save as TSV
        tsv_file = os.path.join(self.output_dir, f'nethack_kg{suffix}.tsv')
        with open(tsv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            # Write header
            writer.writerow(['head', 'relation', 'tail'])
            for triplet in triplets:
                writer.writerow([
                    triplet['head'],
                    triplet['type'],
                    triplet['tail']
                ])
        
        print(f"Saved TSV to: {tsv_file}")
        
        # Save full JSON with metadata
        json_file = os.path.join(self.output_dir, f'nethack_kg{suffix}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(triplets, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats = self.compute_statistics(triplets)
        stats_file = os.path.join(self.output_dir, f'statistics{suffix}.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved statistics to: {stats_file}")
    
    def compute_statistics(self, triplets: List[Dict]) -> Dict:
        """Compute KG statistics"""
        entities = set()
        relations = set()
        sources = set()
        
        relation_counts = {}
        entity_counts = {}
        
        for triplet in triplets:
            # Collect unique items
            entities.add(triplet['head'])
            entities.add(triplet['tail'])
            relations.add(triplet['type'])
            sources.add(triplet['source'])
            
            # Count relations
            rel = triplet['type']
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
            
            # Count entity occurrences
            for ent in [triplet['head'], triplet['tail']]:
                entity_counts[ent] = entity_counts.get(ent, 0) + 1
        
        # Sort by frequency
        top_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        
        stats = {
            'total_triplets': len(triplets),
            'unique_entities': len(entities),
            'unique_relations': len(relations),
            'unique_sources': len(sources),
            'top_relations': top_relations,
            'top_entities': top_entities,
            'avg_triplets_per_source': len(triplets) / len(sources) if sources else 0
        }
        
        return stats


def main():
    """Main function with options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build NetHack knowledge graph from wiki')
    parser.add_argument('--mode', choices=['sample', 'medium', 'full'], default='medium',
                        help='Processing mode: sample (100 pages), medium (1000 pages), full (all pages)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for processing')
    parser.add_argument('--xml-path', default='data/nethackwiki_current.xml',
                        help='Path to wiki XML file')
    parser.add_argument('--output-dir', default='data/nethack_kg_full',
                        help='Output directory')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh instead of resuming from existing data')
    
    args = parser.parse_args()
    
    # Set max pages based on mode
    max_pages = {
        'sample': 100,
        'medium': 1000,
        'full': None
    }[args.mode]
    
    print(f"Building NetHack KG in {args.mode} mode...")
    print(f"Max pages: {max_pages or 'all'}")
    print(f"Batch size: {args.batch_size}")
    
    # Build KG
    builder = NetHackKGBuilder(args.xml_path, args.output_dir)
    triplets = builder.process_wiki_full(
        batch_size=args.batch_size,
        max_pages=max_pages,
        resume=not args.no_resume
    )
    
    print("\n" + "="*50)
    print("Knowledge graph building completed!")
    print(f"Output saved to: {args.output_dir}")
    
    # Print sample triplets
    print("\nSample triplets:")
    for i, triplet in enumerate(triplets[:10]):
        print(f"{i+1}. {triplet['head']} --[{triplet['type']}]--> {triplet['tail']}")


if __name__ == "__main__":
    main()