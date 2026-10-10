#!/bin/bash
# Complete NetHack KG creation script

echo "=================================="
echo "NetHack A+ Knowledge Graph Creator"
echo "=================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Check if XML file exists
if [ ! -f "data/nethackwiki_current.xml" ]; then
    echo "Error: NetHack wiki XML not found at data/nethackwiki_current.xml"
    echo "Please download it from: https://nethackwiki.com/wiki/Special:Export"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q pandas torch transformers sentence-transformers scikit-learn tqdm numpy

# Run the full pipeline
echo "Starting full pipeline..."
python3 research/process_full_wiki_to_a_plus.py

echo ""
echo "Done! Your A+ quality KG is ready at: data/nethack_kg_final_a_plus.tsv"