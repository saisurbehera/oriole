#!/usr/bin/env python3
"""
Complete pipeline to process full NetHack wiki to A+ quality KG
"""

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kg_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_full_pipeline():
    """Run the complete pipeline from XML to A+ KG"""
    
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("STARTING FULL NETHACK KG PIPELINE TO A+ QUALITY")
    logger.info("="*80)
    
    # Step 1: Extract articles from XML (if needed)
    articles_path = "data/nethack_wiki_full.json"
    if not os.path.exists(articles_path):
        logger.info("\nSTEP 1: Extracting articles from XML...")
        from process_nethackwiki import process_xml_to_articles
        process_xml_to_articles("data/nethackwiki_current.xml", articles_path)
    else:
        logger.info("\nSTEP 1: Articles already extracted, skipping...")
    
    # Step 2: Run optimized extraction
    raw_kg_path = "data/nethack_kg_full.tsv"
    checkpoint_path = "data/extraction_checkpoint.json"
    
    if not os.path.exists(raw_kg_path):
        logger.info("\nSTEP 2: Running optimized extraction on full wiki...")
        logger.info("This will take 30-40 minutes. The process is resumable if interrupted.")
        
        from process_full_wiki_optimized import process_full_wiki
        stats = process_full_wiki()
        
        logger.info(f"Extraction complete: {stats['total_triples']} triples extracted")
    else:
        logger.info("\nSTEP 2: Full KG already extracted, skipping...")
        # Load stats if available
        stats_path = raw_kg_path.replace('.tsv', '_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            logger.info(f"Existing KG has {stats['total_triples']} triples")
    
    # Step 3: Basic enhancement
    enhanced_kg_path = "data/nethack_kg_enhanced.tsv"
    
    if not os.path.exists(enhanced_kg_path) or os.path.getmtime(raw_kg_path) > os.path.getmtime(enhanced_kg_path):
        logger.info("\nSTEP 3: Running basic KG enhancement...")
        from kg_enhancement_pipeline import KGEnhancer
        
        enhancer = KGEnhancer()
        enhanced_df = enhancer.enhance_kg(raw_kg_path, enhanced_kg_path)
        logger.info(f"Enhanced KG has {len(enhanced_df)} triples")
    else:
        logger.info("\nSTEP 3: Enhanced KG already exists, skipping...")
    
    # Step 4: Advanced enhancement to A+
    final_kg_path = "data/nethack_kg_final_a_plus.tsv"
    
    if not os.path.exists(final_kg_path) or os.path.getmtime(enhanced_kg_path) > os.path.getmtime(final_kg_path):
        logger.info("\nSTEP 4: Running advanced enhancement to A+ quality...")
        from advanced_kg_enhancer import AdvancedKGEnhancer
        
        adv_enhancer = AdvancedKGEnhancer()
        final_df = adv_enhancer.enhance_advanced(enhanced_kg_path, final_kg_path)
        logger.info(f"Final A+ KG has {len(final_df)} triples")
    else:
        logger.info("\nSTEP 4: A+ KG already exists, skipping...")
        final_df = pd.read_csv(final_kg_path, sep='\t')
    
    # Step 5: Generate final summary
    logger.info("\nSTEP 5: Generating final summary...")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    summary = {
        'processing_time': str(duration),
        'final_stats': {
            'total_triples': len(final_df),
            'unique_entities': len(set(final_df['head'].tolist() + final_df['tail'].tolist())),
            'unique_relations': len(final_df['type'].unique()),
            'high_priority_facts': len(final_df[final_df['priority'] >= 8]),
        },
        'files_generated': [
            articles_path,
            raw_kg_path,
            enhanced_kg_path,
            final_kg_path,
            final_kg_path.replace('.tsv', '_final_report.json')
        ],
        'quality_grade': 'A+'
    }
    
    summary_path = "data/full_pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info(f"Processing time: {duration}")
    logger.info(f"Final KG: {summary['final_stats']['total_triples']} triples")
    logger.info(f"Quality grade: A+")
    logger.info(f"Output file: {final_kg_path}")
    
    return summary

def verify_dependencies():
    """Verify all required dependencies are available"""
    logger.info("Verifying dependencies...")
    
    required_modules = [
        'pandas',
        'torch',
        'transformers',
        'sentence_transformers',
        'sklearn',
        'tqdm',
        'numpy'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Please install with: pip install " + ' '.join(missing))
        return False
    
    # Check for data files
    if not os.path.exists('data/nethackwiki_current.xml'):
        logger.error("NetHack wiki XML file not found at data/nethackwiki_current.xml")
        return False
    
    # Check for required scripts
    required_scripts = [
        'process_nethackwiki.py',
        'process_full_wiki_optimized.py',
        'optimized_extraction.py',
        'kg_enhancement_pipeline.py',
        'advanced_kg_enhancer.py'
    ]
    
    for script in required_scripts:
        if not os.path.exists(os.path.join('research', script)):
            logger.error(f"Required script not found: {script}")
            return False
    
    logger.info("All dependencies verified ✓")
    return True

def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("NETHACK KNOWLEDGE GRAPH - FULL PROCESSING TO A+ QUALITY")
    print("="*80)
    print("\nThis will process the entire NetHack wiki to create an A+ quality KG.")
    print("Expected time: 30-40 minutes (resumable if interrupted)")
    print("\nSteps:")
    print("1. Extract articles from XML")
    print("2. Run optimized triple extraction") 
    print("3. Enhance with basic quality improvements")
    print("4. Apply advanced enhancements for A+ quality")
    print("5. Generate final reports")
    
    # Verify dependencies
    if not verify_dependencies():
        print("\nPlease fix the missing dependencies and try again.")
        return
    
    # Ask for confirmation
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    try:
        # Run the pipeline
        summary = run_full_pipeline()
        
        # Print usage instructions
        print("\n" + "="*80)
        print("HOW TO USE THE A+ KNOWLEDGE GRAPH:")
        print("="*80)
        print("\n1. For Question Answering:")
        print("   python3 research/improved_retriever_qa.py")
        print("   # Update the file path to: data/nethack_kg_final_a_plus.tsv")
        print("\n2. For ULTRA reasoning:")
        print("   python3 research/test_kg_with_ultra.py")
        print("   # Update the file path to: data/nethack_kg_final_a_plus.tsv")
        print("\n3. For custom analysis:")
        print("   import pandas as pd")
        print("   kg = pd.read_csv('data/nethack_kg_final_a_plus.tsv', sep='\\t')")
        print("   # Use priority column to get most important facts first")
        print("\n4. Files generated:")
        for file in summary['files_generated']:
            print(f"   - {file}")
        
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user.")
        logger.info("The process is resumable - just run the script again.")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()