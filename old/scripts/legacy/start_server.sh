#!/bin/bash
# Start NetHack KGQA API Server

echo "🎮 Starting NetHack Knowledge Graph QA API Server"
echo "================================================"

# Install requirements if needed
echo "Installing server dependencies..."
pip install fastapi uvicorn pandas requests

# Check if KG file exists
if [ ! -f "data/nethack_kg_final_a_plus.tsv" ]; then
    echo "❌ Error: Knowledge graph file not found!"
    echo "Please run the pipeline first to generate: data/nethack_kg_final_a_plus.tsv"
    exit 1
fi

# Check if VLLM is running
echo "Checking VLLM server..."
if curl -s http://localhost:8080/v1/models > /dev/null; then
    echo "✅ VLLM server is running"
else
    echo "⚠️  Warning: VLLM server not detected at localhost:8080"
    echo "   The API will still start but answer generation may fail"
fi

# Start the server
echo "Starting FastAPI server on port 8000..."
python3 research/nethack_qa_server.py