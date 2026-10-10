#!/usr/bin/env python3
"""
Extract first 100 pages from NetHack wiki for experimentation
"""

import xml.etree.ElementTree as ET
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_first_n_pages(xml_path: str, n: int = 100, output_path: str = "data/nethack_wiki_first_100.json"):
    """Extract first N pages from wiki XML"""
    
    logger.info(f"Parsing XML from {xml_path}")
    
    articles = []
    count = 0
    
    # Parse XML iteratively to handle large files
    context = ET.iterparse(xml_path, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    for event, elem in context:
        if event == 'end' and elem.tag.endswith('page'):
            # Extract page data - handle both namespace versions
            title_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}title')
            if title_elem is None:
                title_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.11/}title')
            
            text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}text')
            if text_elem is None:
                text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.11/}text')
            
            if title_elem is not None and text_elem is not None:
                title = title_elem.text
                text = text_elem.text or ""
                
                # Skip redirects and special pages
                if text.startswith('#REDIRECT') or title.startswith('Special:'):
                    elem.clear()
                    root.clear()
                    continue
                
                # Add article
                articles.append({
                    'title': title,
                    'text': text[:5000]  # Limit text length for experiments
                })
                
                count += 1
                if count >= n:
                    break
            
            # Clear elements to save memory
            elem.clear()
            root.clear()
    
    logger.info(f"Extracted {len(articles)} articles")
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(articles, f, indent=2)
    
    logger.info(f"Saved to {output_path}")
    
    # Show some statistics
    print("\nSample articles extracted:")
    for i, article in enumerate(articles[:10]):
        print(f"{i+1}. {article['title']} (text length: {len(article['text'])})")
    
    return output_path

def main():
    xml_path = "data/nethackwiki_current.xml"
    output_path = "data/nethack_wiki_first_100.json"
    
    extract_first_n_pages(xml_path, n=100, output_path=output_path)

if __name__ == "__main__":
    main()