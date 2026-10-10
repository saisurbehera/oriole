#!/usr/bin/env python3
"""
Extract articles from NetHack wiki XML
"""

import xml.etree.ElementTree as ET
import json
import re
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_xml_to_articles(xml_path: str, output_path: str):
    """Extract all articles from NetHack wiki XML"""
    
    logger.info(f"Processing XML file: {xml_path}")
    
    articles = []
    
    # Parse XML with iterparse for memory efficiency
    context = ET.iterparse(xml_path, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    # Namespace handling
    ns = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}
    
    page_count = 0
    
    for event, elem in tqdm(context, desc="Parsing XML"):
        if event == 'end' and elem.tag.endswith('page'):
            # Extract page data - handle namespace
            ns = '{http://www.mediawiki.org/xml/export-0.10/}'
            title_elem = elem.find(f'.//{ns}title')
            text_elem = elem.find(f'.//{ns}revision/{ns}text')
            
            if title_elem is not None and text_elem is not None:
                title = title_elem.text
                text = text_elem.text or ""
                
                # Store article
                articles.append({
                    'title': title,
                    'text': text
                })
                
                page_count += 1
                if page_count % 1000 == 0:
                    logger.info(f"Processed {page_count} pages...")
            
            # Clear element to save memory
            elem.clear()
            root.clear()
    
    logger.info(f"Extracted {len(articles)} articles")
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(articles, f)
    
    logger.info(f"Saved articles to {output_path}")
    
    return len(articles)


if __name__ == "__main__":
    xml_path = "data/nethackwiki_current.xml"
    output_path = "data/nethack_wiki_full.json"
    process_xml_to_articles(xml_path, output_path)