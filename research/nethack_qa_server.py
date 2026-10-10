#!/usr/bin/env python3
"""
NetHack KGQA API Server
FastAPI server for NetHack knowledge graph question answering
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import requests
import uvicorn
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    entity: Optional[str] = None
    max_facts: Optional[int] = 5
    include_predictions: Optional[bool] = True

class Fact(BaseModel):
    head: str
    relation: str
    tail: str
    priority: int
    confidence: str

class QueryResponse(BaseModel):
    question: str
    entity: str
    facts_used: List[Fact]
    answer: str
    status: str

# Initialize FastAPI
app = FastAPI(
    title="NetHack Knowledge Graph QA API",
    description="Answer NetHack questions using knowledge graph and VLLM",
    version="1.0.0"
)

# Global variables
kg = None
entity_facts = None
vllm_url = "http://localhost:8080"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

@app.on_event("startup")
async def startup_event():
    """Load KG on startup"""
    global kg, entity_facts
    
    logger.info("Loading NetHack Knowledge Graph...")
    
    try:
        kg = pd.read_csv("data/nethack_kg_final_a_plus.tsv", sep='\t')
        logger.info(f"Loaded {len(kg)} triples from KG")
        
        # Build entity index
        entity_facts = defaultdict(list)
        for _, row in kg.iterrows():
            entity_facts[str(row['head']).lower()].append({
                'head': row['head'],
                'relation': row['type'],
                'tail': row['tail'],
                'priority': row.get('priority', 5)
            })
        
        logger.info(f"Indexed {len(entity_facts)} entities")
        
        # Test VLLM connection
        resp = requests.get(f"{vllm_url}/v1/models", timeout=5)
        if resp.status_code == 200:
            logger.info("✓ VLLM server connected")
        else:
            logger.warning(f"VLLM server status: {resp.status_code}")
            
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise

def extract_entity(question: str) -> str:
    """Extract entity from question"""
    question_lower = question.lower()
    
    # Check for entities in the question
    for entity in entity_facts.keys():
        if entity in question_lower:
            # Return the original cased version
            for _, row in kg.iterrows():
                if str(row['head']).lower() == entity:
                    return row['head']
    
    return "NetHack"  # Default fallback

def get_facts(entity: str, max_facts: int = 5) -> List[Dict]:
    """Get facts about entity"""
    entity_lower = entity.lower()
    
    if entity_lower not in entity_facts:
        return []
    
    facts = entity_facts[entity_lower]
    
    # Sort by priority
    facts_sorted = sorted(facts, key=lambda x: x['priority'], reverse=True)
    
    return facts_sorted[:max_facts]

def predict_facts(entity: str, max_predictions: int = 2) -> List[Dict]:
    """Simple fact prediction based on similar entities"""
    predictions = []
    
    # Get existing facts for this entity
    existing_facts = get_facts(entity)
    existing_relations = {f['relation'] for f in existing_facts}
    
    # Find similar entities (those sharing relations)
    similar_entities = []
    for other_entity, other_facts in entity_facts.items():
        if other_entity != entity.lower():
            other_relations = {f['relation'] for f in other_facts}
            overlap = len(existing_relations & other_relations)
            if overlap > 0:
                similar_entities.append((other_entity, overlap))
    
    # Sort by similarity
    similar_entities.sort(key=lambda x: x[1], reverse=True)
    
    # Get common relations from similar entities
    relation_scores = defaultdict(int)
    for similar_entity, _ in similar_entities[:5]:
        for fact in entity_facts[similar_entity]:
            if fact['relation'] not in existing_relations:
                relation_scores[fact['relation']] += 1
    
    # Create predictions
    for relation, score in sorted(relation_scores.items(), key=lambda x: x[1], reverse=True)[:max_predictions]:
        predictions.append({
            'head': entity,
            'relation': relation,
            'tail': f"<predicted>",
            'priority': score,
            'confidence': 'predicted'
        })
    
    return predictions

def generate_answer(question: str, facts: List[Dict]) -> str:
    """Generate answer using VLLM"""
    
    # Separate known vs predicted facts
    known_facts = [f for f in facts if f.get('confidence', 'known') == 'known']
    predicted_facts = [f for f in facts if f.get('confidence') == 'predicted']
    
    # Format context
    context_parts = []
    
    if known_facts:
        known_lines = [f"- {f['head']} {f['relation']} {f['tail']}" for f in known_facts]
        context_parts.append("Known facts:\n" + "\n".join(known_lines))
    
    if predicted_facts:
        pred_lines = [f"- {f['head']} might have {f['relation']}" for f in predicted_facts]
        context_parts.append("Possible attributes:\n" + "\n".join(pred_lines))
    
    context = "\n\n".join(context_parts) if context_parts else "No specific facts available."
    
    # Create prompt
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful NetHack expert. Answer questions using the provided facts. Be concise and informative.<|eot_id|><|start_header_id|>user<|end_header_id|>

{context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    try:
        response = requests.post(
            f"{vllm_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": 150,
                "temperature": 0.7,
                "stop": ["<|eot_id|>", "\n\n", "Question:"]
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['text'].strip()
        else:
            return f"VLLM Error: {response.status_code}"
            
    except Exception as e:
        return f"Generation Error: {str(e)}"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "NetHack Knowledge Graph QA API",
        "version": "1.0.0",
        "status": "running",
        "total_facts": len(kg) if kg is not None else 0,
        "entities": len(entity_facts) if entity_facts is not None else 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test VLLM connection
        resp = requests.get(f"{vllm_url}/v1/models", timeout=3)
        vllm_status = resp.status_code == 200
    except:
        vllm_status = False
    
    return {
        "kg_loaded": kg is not None,
        "vllm_connected": vllm_status,
        "status": "healthy" if (kg is not None and vllm_status) else "degraded"
    }

@app.post("/query", response_model=QueryResponse)
async def query_qa(request: QueryRequest):
    """Main QA endpoint"""
    
    if kg is None:
        raise HTTPException(status_code=500, detail="Knowledge graph not loaded")
    
    try:
        # Extract entity if not provided
        entity = request.entity if request.entity else extract_entity(request.question)
        
        # Get facts
        facts = get_facts(entity, request.max_facts)
        
        # Add predictions if requested
        if request.include_predictions and len(facts) < request.max_facts:
            predictions = predict_facts(entity, max_predictions=2)
            facts.extend(predictions)
        
        # Mark confidence for known facts
        for fact in facts:
            if 'confidence' not in fact:
                fact['confidence'] = 'known'
        
        # Generate answer
        answer = generate_answer(request.question, facts)
        
        # Format response
        fact_objects = [
            Fact(
                head=f['head'],
                relation=f['relation'], 
                tail=f['tail'],
                priority=f['priority'],
                confidence=f['confidence']
            ) for f in facts[:request.max_facts]
        ]
        
        return QueryResponse(
            question=request.question,
            entity=entity,
            facts_used=fact_objects,
            answer=answer,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/entities")
async def list_entities():
    """List available entities"""
    if entity_facts is None:
        raise HTTPException(status_code=500, detail="Knowledge graph not loaded")
    
    # Get entities with most facts
    entity_counts = [(entity, len(facts)) for entity, facts in entity_facts.items()]
    entity_counts.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "total_entities": len(entity_counts),
        "top_entities": entity_counts[:20]
    }

@app.get("/relations")
async def list_relations():
    """List available relation types"""
    if kg is None:
        raise HTTPException(status_code=500, detail="Knowledge graph not loaded")
    
    relation_counts = kg['type'].value_counts().to_dict()
    
    return {
        "total_relations": len(relation_counts),
        "relations": relation_counts
    }

@app.get("/entity/{entity_name}")
async def get_entity_info(entity_name: str):
    """Get information about a specific entity"""
    facts = get_facts(entity_name, max_facts=10)
    
    if not facts:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")
    
    return {
        "entity": entity_name,
        "fact_count": len(facts),
        "facts": facts
    }

if __name__ == "__main__":
    print("🎮 Starting NetHack KGQA API Server")
    print("="*50)
    print("Endpoints:")
    print("  GET  /           - Service info")
    print("  GET  /health     - Health check") 
    print("  POST /query      - Ask questions")
    print("  GET  /entities   - List entities")
    print("  GET  /relations  - List relations")
    print("  GET  /entity/{name} - Entity info")
    print("\nStarting server on http://localhost:8000")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )