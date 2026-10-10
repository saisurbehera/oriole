#!/usr/bin/env python3
"""
Basic Training Test
===================

Test basic training functionality without all the advanced features.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_training():
    """Test basic training setup"""
    
    print("🎮 Testing Basic NetHack SFT Training")
    print("=" * 40)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️  No GPU available")
        return
    
    # Load dataset
    print("📚 Loading dataset...")
    dataset = load_from_disk("enhanced_nethack_dataset")
    print(f"   - Train: {len(dataset['train'])}")
    print(f"   - Validation: {len(dataset['validation'])}")
    
    # Take a small sample for testing
    small_train = dataset['train'].select(range(10))
    small_val = dataset['validation'].select(range(5))
    
    print(f"📊 Sample data for testing:")
    sample = small_train[0]
    print(f"   - Instruction length: {len(sample['instruction'])}")
    print(f"   - Output: {sample['output']}")
    print(f"   - Subgoal: {sample['subgoal']}")
    
    # Test tokenizer loading
    print("🤖 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded")
    except Exception as e:
        print(f"❌ Tokenizer failed: {e}")
        return
    
    # Test model loading
    print("🧠 Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-small",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Model loaded")
        
        # Check memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"📊 GPU Memory used: {allocated:.2f} GB")
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Test tokenization
    print("🔤 Testing tokenization...")
    try:
        def format_example(example):
            instruction = example['instruction']
            response = example['output']
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            return {"text": text}
        
        def tokenize_function(examples):
            formatted = [format_example(ex) for ex in examples]
            texts = [ex["text"] for ex in formatted]
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=512,  # Shorter for testing
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Test on small sample
        processed = small_train.map(
            tokenize_function,
            batched=True,
            remove_columns=small_train.column_names,
        )
        
        print(f"✅ Tokenization successful")
        print(f"   - Processed {len(processed)} examples")
        print(f"   - Sample token length: {len(processed[0]['input_ids'])}")
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return
    
    # Test training arguments
    print("⚙️  Testing training setup...")
    try:
        training_args = TrainingArguments(
            output_dir="./test-training",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            logging_steps=1,
            eval_steps=2,
            save_steps=5,
            evaluation_strategy="steps",
            save_strategy="steps",
            fp16=True,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for testing
        )
        
        print("✅ Training arguments created")
        
    except Exception as e:
        print(f"❌ Training args failed: {e}")
        return
    
    print(f"\n🎯 Basic setup test completed successfully!")
    print(f"💡 Ready for full training with:")
    print(f"   - Model: microsoft/DialoGPT-small")
    print(f"   - Dataset: {len(dataset['train'])} examples")
    print(f"   - GPU: RTX 4090 (24GB)")
    print(f"   - Memory usage: {allocated:.2f} GB")


if __name__ == "__main__":
    test_basic_training()