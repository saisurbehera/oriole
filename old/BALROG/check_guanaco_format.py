#!/usr/bin/env python3
"""
Check openassistant-guanaco dataset format
"""

from datasets import load_dataset

def check_guanaco():
    """Check how openassistant-guanaco dataset is formatted"""
    print("🔍 Loading openassistant-guanaco dataset...")
    
    try:
        # Load just a small sample
        dataset = load_dataset("timdettmers/openassistant-guanaco", split="train[:10]")
        
        print(f"📊 Dataset info:")
        print(f"   - Samples: {len(dataset)}")
        print(f"   - Columns: {dataset.column_names}")
        
        # Check first example
        example = dataset[0]
        print(f"\n🔍 First example:")
        for key, value in example.items():
            print(f"   - {key}: {type(value)} - {str(value)[:200]}...")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Let's create a simple test dataset instead...")
        
        # Create a simple test
        simple_data = {
            "text": [
                "### Human: What is 2+2?### Assistant: 2+2 equals 4.",
                "### Human: Tell me about cats.### Assistant: Cats are small mammals.",
            ]
        }
        
        from datasets import Dataset
        simple_dataset = Dataset.from_dict(simple_data)
        
        print(f"\n📊 Simple test dataset:")
        print(f"   - Samples: {len(simple_dataset)}")
        print(f"   - Columns: {simple_dataset.column_names}")
        print(f"   - Example: {simple_dataset[0]['text']}")

if __name__ == "__main__":
    check_guanaco()