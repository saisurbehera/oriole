#!/usr/bin/env python3
"""
Process full NetHack wiki with optimized extraction
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the optimized extractor
from optimized_extraction import OptimizedNetHackExtractor


def process_full_wiki():
    """Process the full NetHack wiki"""
    
    # Input and output paths
    xml_path = "data/nethackwiki_current.xml"
    articles_path = "data/nethack_wiki_full.json"
    output_path = "data/nethack_kg_full.tsv"
    checkpoint_path = "data/extraction_checkpoint.json"
    
    # Check if we need to extract articles first
    if not os.path.exists(articles_path):
        logger.info("Articles JSON not found. Extracting from XML...")
        
        # Use the XML processor to extract articles
        from process_nethackwiki import process_xml_to_articles
        process_xml_to_articles(xml_path, articles_path)
    
    # Load articles
    logger.info(f"Loading articles from {articles_path}")
    with open(articles_path, 'r') as f:
        articles = json.load(f)
    
    logger.info(f"Found {len(articles)} total articles")
    
    # Filter out meta pages
    filtered_articles = []
    for article in articles:
        title = article.get('title', '')
        if not any(skip in title for skip in [':', 'MediaWiki', 'Template', 'Talk:', 'User:', 'Category:']):
            if len(article.get('text', '')) > 50:
                filtered_articles.append(article)
    
    logger.info(f"Filtered to {len(filtered_articles)} content articles")
    
    # Check for checkpoint
    start_idx = 0
    all_triples = []
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        start_idx = checkpoint['last_processed_idx'] + 1
        all_triples = checkpoint['triples']
        logger.info(f"Resuming from article {start_idx} with {len(all_triples)} existing triples")
    
    # Initialize extractor
    extractor = OptimizedNetHackExtractor()
    
    # Process articles with checkpointing
    batch_size = 100
    save_interval = 500  # Save checkpoint every 500 articles
    
    for idx in tqdm(range(start_idx, len(filtered_articles)), 
                    initial=start_idx, 
                    total=len(filtered_articles),
                    desc="Processing articles"):
        
        article = filtered_articles[idx]
        title = article.get('title', '')
        text = article.get('text', '')
        
        try:
            # Extract triples
            triples = extractor.extract_from_article(title, text)
            all_triples.extend(triples)
        except Exception as e:
            logger.error(f"Error processing {title}: {e}")
            continue
        
        # Save checkpoint periodically
        if (idx + 1) % save_interval == 0:
            checkpoint = {
                'last_processed_idx': idx,
                'triples': all_triples,
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f)
            logger.info(f"Checkpoint saved at article {idx + 1}, {len(all_triples)} triples extracted")
    
    # Final save
    logger.info(f"Extraction complete! Total triples: {len(all_triples)}")
    
    # Save to TSV
    df = pd.DataFrame(all_triples)
    df = df[['head', 'type', 'tail', 'source']]
    df.to_csv(output_path, sep='\t', index=False)
    logger.info(f"Saved {len(all_triples)} triples to {output_path}")
    
    # Generate statistics
    stats = {
        'total_articles': len(filtered_articles),
        'total_triples': len(all_triples),
        'unique_entities': len(set([t['head'] for t in all_triples] + [t['tail'] for t in all_triples])),
        'unique_relations': len(set([t['type'] for t in all_triples])),
        'relation_distribution': defaultdict(int),
        'source_distribution': defaultdict(int)
    }
    
    for triple in all_triples:
        stats['relation_distribution'][triple['type']] += 1
        stats['source_distribution'][triple['source']] += 1
    
    # Sort distributions
    stats['relation_distribution'] = dict(sorted(
        stats['relation_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:50])  # Top 50 relations
    
    stats['source_distribution'] = dict(sorted(
        stats['source_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:50])  # Top 50 sources
    
    # Save stats
    stats_path = output_path.replace('.tsv', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to {stats_path}")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("Checkpoint file removed")
    
    return stats


def main():
    """Main entry point"""
    stats = process_full_wiki()
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total triples: {stats['total_triples']:,}")
    print(f"Unique entities: {stats['unique_entities']:,}")
    print(f"Unique relations: {stats['unique_relations']}")
    print(f"\nTop 10 relations:")
    for rel, count in list(stats['relation_distribution'].items())[:10]:
        print(f"  {rel}: {count:,}")
    print(f"\nTop 10 source articles:")
    for src, count in list(stats['source_distribution'].items())[:10]:
        print(f"  {src}: {count:,}")


if __name__ == "__main__":
    main()