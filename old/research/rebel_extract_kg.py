"""
Extract knowledge graph triples from NetHack text using REBEL model
"""

import json
import os
from typing import List, Dict
import csv

# Mock the transformers import for testing without GPU
USE_MOCK = False  # Set to False when running with actual model

if not USE_MOCK:
    from transformers import pipeline
else:
    # Mock for testing the pipeline
    class MockPipeline:
        def __init__(self, *args, **kwargs):
            pass
        
        def tokenizer(self):
            return self
        
        def batch_decode(self, inputs):
            # Return mock REBEL-formatted output
            return ['<s><triplet> NetHack <subj> Amulet of Yendor <obj> primary objective </s>']
        
        def __call__(self, text, **kwargs):
            # Return mock token IDs
            return [{"generated_token_ids": [1, 2, 3]}]
    
    def pipeline(*args, **kwargs):
        return MockPipeline()


def extract_triplets(text):
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


def process_nethack_chunks(input_file, output_dir):
    """Process NetHack chunks and extract knowledge graph"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize REBEL model
    print("Initializing REBEL model...")
    if USE_MOCK:
        print("(Using mock pipeline for testing)")
        triplet_extractor = pipeline('text2text-generation')
    else:
        triplet_extractor = pipeline(
            'text2text-generation',
            model='Babelscape/rebel-large',
            tokenizer='Babelscape/rebel-large',
            device="cuda:0"  # Change to "cpu" if no GPU
        )
    
    # Load sample chunks
    with open(input_file, 'r') as f:
        chunks = json.load(f)
    
    print(f"Processing {len(chunks)} text chunks...")
    
    all_triplets = []
    unique_triplets = set()
    
    for i, chunk in enumerate(chunks):
        title = chunk['page_title']
        text = chunk['chunk_text']
        
        print(f"\nProcessing chunk {i+1}/{len(chunks)}: {title}")
        print(f"Text preview: {text[:100]}...")
        
        try:
            # Extract triplets using REBEL
            if USE_MOCK:
                # For testing, create manual triplets based on content
                if "Amulet of Yendor" in text:
                    mock_triplets = [
                        {'head': 'Amulet of Yendor', 'type': 'is', 'tail': 'primary objective'},
                        {'head': 'Amulet of Yendor', 'type': 'hidden in', 'tail': 'Gehennom'},
                        {'head': 'Moloch', 'type': 'stole', 'tail': 'Amulet of Yendor'}
                    ]
                elif "NetHack" in text:
                    mock_triplets = [
                        {'head': 'NetHack', 'type': 'is', 'tail': 'roguelike game'},
                        {'head': 'player', 'type': 'must retrieve', 'tail': 'Amulet of Yendor'},
                        {'head': 'NetHack', 'type': 'features', 'tail': 'permadeath'}
                    ]
                elif "Elbereth" in text:
                    mock_triplets = [
                        {'head': 'Elbereth', 'type': 'is', 'tail': 'powerful word'},
                        {'head': 'Elbereth', 'type': 'can be engraved on', 'tail': 'floor'},
                        {'head': 'monsters', 'type': 'scared by', 'tail': 'Elbereth'}
                    ]
                elif "Sokoban" in text:
                    mock_triplets = [
                        {'head': 'Sokoban', 'type': 'is', 'tail': 'special branch'},
                        {'head': 'Sokoban', 'type': 'features', 'tail': 'puzzle levels'},
                        {'head': 'Sokoban', 'type': 'rewards', 'tail': 'bag of holding'}
                    ]
                else:
                    # Generic triplets
                    mock_triplets = [
                        {'head': title.replace('_', ' '), 'type': 'related to', 'tail': 'NetHack'}
                    ]
                
                triplets = mock_triplets
            else:
                # Use actual REBEL model
                extraction = triplet_extractor(
                    text,
                    return_tensors=True,
                    return_text=False
                )
                
                # Decode the output
                extracted_text = triplet_extractor.tokenizer.batch_decode(
                    [extraction[0]["generated_token_ids"]]
                )
                
                # Parse triplets
                triplets = extract_triplets(extracted_text[0])
            
            # Add page context to triplets
            for triplet in triplets:
                triplet['source'] = title
                
                # Create unique key
                key = (triplet['head'], triplet['type'], triplet['tail'])
                
                if key not in unique_triplets:
                    unique_triplets.add(key)
                    all_triplets.append(triplet)
            
            print(f"Extracted {len(triplets)} triplets")
            
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue
    
    print(f"\nTotal unique triplets extracted: {len(all_triplets)}")
    
    # Save as TSV
    tsv_file = os.path.join(output_dir, 'nethack_kg.tsv')
    with open(tsv_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for triplet in all_triplets:
            writer.writerow([
                triplet['head'],
                triplet['type'],
                triplet['tail']
            ])
    
    print(f"Saved knowledge graph to: {tsv_file}")
    
    # Also save with metadata
    json_file = os.path.join(output_dir, 'nethack_kg_full.json')
    with open(json_file, 'w') as f:
        json.dump(all_triplets, f, indent=2)
    
    print(f"Saved full data to: {json_file}")
    
    # Print sample triplets
    print("\nSample triplets:")
    for triplet in all_triplets[:10]:
        print(f"  {triplet['head']} --[{triplet['type']}]--> {triplet['tail']}")
    
    return all_triplets


if __name__ == "__main__":
    # Process the sample chunks
    input_file = "data/nethack_samples/sample_chunks.json"
    output_dir = "data/nethack_kg"
    
    if os.path.exists(input_file):
        triplets = process_nethack_chunks(input_file, output_dir)
    else:
        print(f"Input file not found: {input_file}")
        print("Please run extract_nethack_samples.py first")