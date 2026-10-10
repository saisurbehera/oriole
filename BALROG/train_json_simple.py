#!/usr/bin/env python3
"""
Simple training script for JSON format dataset (no LoRA to avoid compatibility issues)
"""

import os
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb completely

import torch
import logging
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_from_disk
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedDataCollator(DataCollatorForLanguageModeling):
    """Fixed data collator for variable length sequences"""
    
    def torch_call(self, examples):
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        
        if "labels" in batch:
            batch["labels"] = batch["input_ids"].clone()
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch

@dataclass
class SimpleJsonTrainingConfig:
    """Simple training configuration for JSON format"""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"
    model_max_length: int = 512
    
    # Training settings (no LoRA)
    output_dir: str = "./simple-json-nethack-sft"
    num_train_epochs: int = 5  # More epochs for better learning
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Memory optimizations
    gradient_checkpointing: bool = True
    fp16: bool = False  # Disable FP16 to avoid gradient scaling issues
    dataloader_num_workers: int = 0
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    
    # Logging
    logging_steps: int = 10
    report_to: list = field(default_factory=list)  # Empty list to disable all reporting
    run_name: str = f"simple-json-nethack-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Data settings
    dataset_path: str = "./json_no_reasoning_nethack_dataset"
    max_samples: Optional[int] = None  # Use full dataset

class SimpleJsonNetHackTrainer:
    """Simple NetHack trainer for JSON format"""
    
    def __init__(self, config: SimpleJsonTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(42)
        
        logger.info(f"🚀 Simple JSON NetHack Training")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Model: {config.model_name}")
        logger.info(f"   - Dataset: {config.dataset_path}")
    
    def load_and_prepare_data(self):
        """Load JSON format dataset"""
        logger.info("📚 Loading JSON format dataset...")
        dataset = load_from_disk(self.config.dataset_path)
        
        if self.config.max_samples:
            train_samples = min(self.config.max_samples, len(dataset['train']))
            val_samples = min(self.config.max_samples // 10, len(dataset['validation']))
            
            dataset['train'] = dataset['train'].select(range(train_samples))
            dataset['validation'] = dataset['validation'].select(range(val_samples))
        
        logger.info(f"   - Training samples: {len(dataset['train'])}")
        logger.info(f"   - Validation samples: {len(dataset['validation'])}")
        
        return dataset
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer (no LoRA)"""
        logger.info("🤖 Loading model and tokenizer...")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens for Llama-style chat format
        special_tokens = {
            "additional_special_tokens": [
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>"
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # Model (no quantization, no LoRA)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Use float32 to avoid FP16 issues
        )
        
        # Resize embeddings for new tokens
        model.resize_token_embeddings(len(tokenizer))
        
        # Move to GPU
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        logger.info("✅ Model loaded (no LoRA)")
        
        return model, tokenizer
    
    def preprocess_dataset(self, dataset, tokenizer):
        """Preprocess dataset"""
        logger.info("🔄 Preprocessing dataset...")
        
        def tokenize_function(examples):
            """Already in chat format, just tokenize"""
            # The instruction already contains the full chat format
            tokenized = tokenizer(
                examples['instruction'],
                truncation=True,
                padding=False,
                max_length=self.config.model_max_length,
                return_tensors=None,
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        processed_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset",
        )
        
        return processed_dataset
    
    def train(self):
        """Train the model"""
        logger.info("🎯 Starting training...")
        
        try:
            # Load everything
            dataset = self.load_and_prepare_data()
            model, tokenizer = self.load_model_and_tokenizer()
            processed_dataset = self.preprocess_dataset(dataset, tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                
                gradient_checkpointing=self.config.gradient_checkpointing,
                fp16=self.config.fp16,
                dataloader_num_workers=self.config.dataloader_num_workers,
                
                eval_strategy=self.config.evaluation_strategy,
                eval_steps=self.config.eval_steps,
                save_strategy=self.config.save_strategy,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                load_best_model_at_end=self.config.load_best_model_at_end,
                
                logging_steps=self.config.logging_steps,
                report_to=self.config.report_to,
                run_name=self.config.run_name,
                
                remove_unused_columns=False,
                prediction_loss_only=True,
                metric_for_best_model="eval_loss",
            )
            
            # Data collator
            data_collator = FixedDataCollator(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=processed_dataset["train"],
                eval_dataset=processed_dataset["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            # Memory check
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"🔍 GPU Memory before training: {allocated:.2f} GB")
            
            # Train!
            logger.info("🚂 Starting training...")
            trainer.train()
            
            # Save
            logger.info("💾 Saving final model...")
            trainer.save_model()
            tokenizer.save_pretrained(self.config.output_dir)
            
            # Save config
            with open(f"{self.config.output_dir}/training_config.json", "w") as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
            
            logger.info("✅ Training completed successfully!")
            
            # Test the model
            self.test_model(model, tokenizer)
            
            return trainer
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def test_model(self, model, tokenizer):
        """Quick test of the trained model"""
        logger.info("🧪 Testing trained model...")
        
        # Simple test prompt
        test_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert NetHack player.<|eot_id|><|start_header_id|>user<|end_header_id|>

Player is on dungeon level 1. There are stairs visible. Choose action:

Respond with JSON:
{"action": "chosen_action"}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Tokenize
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        
        logger.info(f"🎯 Test response: {response}")
        
        # Check if it's valid JSON
        try:
            parsed = json.loads(response.strip())
            logger.info(f"✅ Valid JSON: {parsed}")
        except:
            logger.info(f"⚠️  Not valid JSON, but model responded")

def main():
    """Main training function"""
    print("🎮 Simple JSON NetHack Training")
    print("=" * 50)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️  Warning: No GPU available")
    
    # Configuration
    config = SimpleJsonTrainingConfig()
    
    print(f"\n📋 Configuration:")
    print(f"   - Model: {config.model_name}")
    print(f"   - Dataset: {config.dataset_path}")
    print(f"   - Max samples: {config.max_samples}")
    print(f"   - Batch size: {config.per_device_train_batch_size}")
    print(f"   - Epochs: {config.num_train_epochs}")
    print(f"   - No LoRA (for compatibility)")
    print(f"   - Output: {config.output_dir}")
    
    # Train
    trainer = SimpleJsonNetHackTrainer(config)
    trainer.train()
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {config.output_dir}")

if __name__ == "__main__":
    main()