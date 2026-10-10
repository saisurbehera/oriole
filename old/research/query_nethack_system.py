#!/usr/bin/env python3
"""
Query the NetHack QA system with various questions
"""

import requests
import pandas as pd
import json

# Load KG to find interesting entities
kg = pd.read_csv("data/nethack_kg_final_a_plus.tsv", sep='\t')

# Function to make a query
def query_vllm(question, entity_facts):
    """Query VLLM with context"""
    
    # Format facts
    context = "\n".join([f"- {row['head']} {row['type']} {row['tail']}" 
                        for _, row in entity_facts.iterrows()])
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a NetHack expert. Answer using the provided facts.<|eot_id|><|start_header_id|>user<|end_header_id|>

Facts:
{context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    response = requests.post(
        "http://localhost:8080/v1/completions",
        json={
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7,
            "stop": ["<|eot_id|>", "\n\n"]
        }
    )
    
    if response.status_code == 200:
        return response.json()['choices'][0]['text'].strip()
    else:
        return f"Error: {response.status_code}"

# Interesting queries
print("🎮 NetHack Knowledge Graph Queries")
print("="*60)

# Query 1: Complex monster strategy
print("\n1. MONSTER STRATEGY QUERY")
print("-"*40)

# Get facts about multiple dangerous monsters
dangerous_monsters = ['Medusa', 'Lich', 'Dragon', 'Mind flayer', 'Wizard of Yendor']
all_facts = []
for monster in dangerous_monsters:
    facts = kg[kg['head'] == monster].head(3)
    all_facts.append(facts)

combined_facts = pd.concat(all_facts)

question = "What are the most dangerous monsters in NetHack and how should I prepare for them?"
print(f"Q: {question}")
answer = query_vllm(question, combined_facts)
print(f"A: {answer}")

# Query 2: Resistance planning
print("\n\n2. RESISTANCE PLANNING QUERY")
print("-"*40)

resistance_facts = kg[kg['type'].str.contains('resistance', case=False)].head(20)
question = "What resistances are most important in NetHack and how can I get them?"
print(f"Q: {question}")
answer = query_vllm(question, resistance_facts)
print(f"A: {answer}")

# Query 3: Speed comparison
print("\n\n3. SPEED TACTICS QUERY")
print("-"*40)

speed_facts = kg[kg['type'].isin(['speed', 'speed_category'])].head(15)
question = "Which creatures are very fast and how does speed affect combat?"
print(f"Q: {question}")
answer = query_vllm(question, speed_facts)
print(f"A: {answer}")

# Query 4: Eating strategy
print("\n\n4. EATING STRATEGY QUERY")
print("-"*40)

eating_facts = kg[kg['type'] == 'eating_effect'].head(10)
question = "What corpses should I eat for beneficial effects and which should I avoid?"
print(f"Q: {question}")
print("\nRelevant eating effects found:")
for _, row in eating_facts.iterrows():
    print(f"  • {row['head']}: {row['tail']}")
answer = query_vllm(question, eating_facts)
print(f"\nA: {answer}")

# Query 5: Combat ratings
print("\n\n5. COMBAT ANALYSIS QUERY")
print("-"*40)

combat_facts = kg[kg['type'].isin(['combat_rating', 'attack_damage', 'armor_class'])].sort_values('priority', ascending=False).head(15)
question = "Which monsters have the highest combat ratings and what makes them dangerous?"
print(f"Q: {question}")
answer = query_vllm(question, combat_facts)
print(f"A: {answer}")

# Query 6: Artifact knowledge
print("\n\n6. ARTIFACT QUERY")
print("-"*40)

artifact_facts = kg[(kg['head'].str.contains('Excalibur|Mjollnir|Stormbringer|Grayswandir', case=False)) | 
                   (kg['tail'].str.contains('artifact', case=False))].head(10)
question = "What are the most powerful artifacts and their properties?"
print(f"Q: {question}")
answer = query_vllm(question, artifact_facts)
print(f"A: {answer}")

# Interactive query
print("\n\n7. CUSTOM QUERY")
print("-"*40)
print("Enter your own NetHack question (or press Enter to skip):")
custom_q = input("> ").strip()

if custom_q:
    # Find relevant facts based on keywords
    keywords = custom_q.lower().split()
    relevant_facts = kg[kg.apply(lambda row: any(kw in str(row).lower() for kw in keywords), axis=1)].head(10)
    
    if len(relevant_facts) > 0:
        print(f"\nFound {len(relevant_facts)} relevant facts")
        answer = query_vllm(custom_q, relevant_facts)
        print(f"A: {answer}")
    else:
        print("No relevant facts found in KG")

print("\n" + "="*60)
print("Query session complete!")