#!/bin/bash

echo "NetHack Knowledge Graph Extraction Options:"
echo "=========================================="
echo "1. Sample mode (100 pages, ~5 minutes)"
echo "2. Medium mode (1000 pages, ~30 minutes)"  
echo "3. Full mode (all pages, ~2-3 hours)"
echo "4. Process more samples (uses research/process_more_nethack.py)"
echo ""

# Check if research/rebel_extract_kg.py has USE_MOCK=False
if grep -q "USE_MOCK = True" research/rebel_extract_kg.py; then
    echo "WARNING: REBEL mock mode is enabled. Switching to real model..."
    sed -i 's/USE_MOCK = True/USE_MOCK = False/' research/rebel_extract_kg.py
fi

read -p "Select mode (1-4): " mode

case $mode in
    1)
        echo "Running sample extraction (100 pages)..."
        python3 research/process_full_nethack_kg.py --mode sample --batch-size 25
        ;;
    2)
        echo "Running medium extraction (1000 pages)..."
        python3 research/process_full_nethack_kg.py --mode medium --batch-size 50
        ;;
    3)
        echo "Running full extraction (all pages)..."
        echo "This will take 2-3 hours and generate a large KG."
        read -p "Are you sure? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            python3 research/process_full_nethack_kg.py --mode full --batch-size 100
        fi
        ;;
    4)
        echo "Processing more samples..."
        python3 research/process_more_nethack.py
        ;;
    *)
        echo "Invalid selection"
        exit 1
        ;;
esac

echo ""
echo "Extraction complete! Check the output directory for results."
echo "Files generated:"
echo "- nethack_kg.tsv: The knowledge graph in TSV format"
echo "- nethack_kg.json: Full data with metadata"
echo "- statistics.json: KG statistics"