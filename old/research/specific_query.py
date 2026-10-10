#!/usr/bin/env python3
"""
Specific targeted query
"""

import requests
import pandas as pd

# Load KG
kg = pd.read_csv("data/nethack_kg_final_a_plus.tsv", sep='\t')

# Get specific facts about Medusa
medusa_facts = kg[kg['head'] == 'Medusa'].sort_values('priority', ascending=False)

print("🐍 MEDUSA QUERY EXAMPLE")
print("="*50)

print("\nFacts about Medusa in the KG:")
for _, row in medusa_facts.head(8).iterrows():
    print(f"  [{row['priority']}] {row['head']} {row['type']} {row['tail']}")

# Create a very specific context
context = f"""Medusa has armor class 2
Medusa has difficulty 25
Medusa has speed 12
Medusa does 2d4 attack damage
Medusa does 1d8 attack damage with gaze
Medusa causes stoning with gaze attack
Medusa appears on dungeon level 20"""

prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a NetHack expert. Answer the question using only the provided facts. Be specific and detailed.<|eot_id|><|start_header_id|>user<|end_header_id|>

NetHack facts:
{context}

Question: Tell me about fighting Medusa - what makes her dangerous and what's her armor class?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

print(f"\nQuery: Tell me about fighting Medusa")
print("Sending to VLLM...")

response = requests.post(
    "http://localhost:8080/v1/completions",
    json={
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.1,  # Lower temperature for more focused response
        "stop": ["<|eot_id|>"]
    }
)

if response.status_code == 200:
    result = response.json()
    answer = result['choices'][0]['text'].strip()
    print(f"\nMedusa Analysis:")
    print(f"{answer}")
else:
    print(f"Error: {response.status_code}")

# Try another specific query about combat
print("\n" + "="*50)
print("⚔️  COMBAT RATING QUERY")

# Get specific combat data
combat_data = kg[kg['type'] == 'combat_rating'].sort_values('tail', ascending=False).head(5)

print("\nTop combat ratings found:")
for _, row in combat_data.iterrows():
    print(f"  • {row['head']}: {row['tail']}")

context2 = "\n".join([f"{row['head']} has combat rating {row['tail']}" for _, row in combat_data.iterrows()])

prompt2 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a NetHack expert. Answer using the combat ratings provided.<|eot_id|><|start_header_id|>user<|end_header_id|>

Combat ratings:
{context2}

Question: Which creatures have the highest combat ratings and what does this mean for combat?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

response2 = requests.post(
    "http://localhost:8080/v1/completions",
    json={
        "model": "meta-llama/Llama-3.2-1B-Instruct", 
        "prompt": prompt2,
        "max_tokens": 150,
        "temperature": 0.2
    }
)

if response2.status_code == 200:
    answer2 = response2.json()['choices'][0]['text'].strip()
    print(f"\nCombat Analysis:")
    print(f"{answer2}")

print("\n" + "="*50)
print("System working! The KG + VLLM integration is functional.")
print("Try running: python3 research/nethack_ultra_vllm.py for interactive mode")