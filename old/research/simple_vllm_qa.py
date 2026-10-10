#!/usr/bin/env python3
"""
Simple NetHack QA with VLLM
"""

import requests
import pandas as pd

# Test VLLM
print("Testing VLLM...")
resp = requests.get("http://localhost:8080/v1/models")
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    print(f"Model: {resp.json()['data'][0]['id']}")

# Load KG
print("\nLoading KG...")
kg = pd.read_csv("data/nethack_kg_final_a_plus.tsv", sep='\t')
print(f"Loaded {len(kg)} facts")

# Simple test
entity = "Medusa"
facts = kg[kg['head'] == entity].head(5)

context = "\n".join([f"{row['head']} {row['type']} {row['tail']}" 
                    for _, row in facts.iterrows()])

prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a NetHack expert.<|eot_id|><|start_header_id|>user<|end_header_id|>
Facts: {context}
Question: What is Medusa's armor class?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

response = requests.post(
    "http://localhost:8080/v1/completions",
    json={
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.7
    }
)

print(f"\nResponse: {response.json()['choices'][0]['text']}")