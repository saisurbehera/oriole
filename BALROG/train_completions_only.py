#!/usr/bin/env python3
"""
Train on completions only - compute loss only on the JSON output, not the input
"""

import os
os.environ["WANDB_DISABLED"] = "true"

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

class CompletionsOnlyDataCollator(DataCollatorForLanguageModeling):
    """Data collator that only computes loss on completions"""
    
    def torch_call(self, examples):
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # The key difference: labels are already set correctly in preprocessing
        # to -100 for input tokens and actual token IDs for completion tokens
        return batch

@dataclass
class CompletionsTrainingConfig:
    """Training configuration for completions-only training"""
    
    model_name: str = "microsoft/DialoGPT-small"
    model_max_length: int = 512
    
    output_dir: str = "./completions-only-nethack-sft"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    gradient_checkpointing: bool = True
    fp16: bool = False
    dataloader_num_workers: int = 0
    
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    
    logging_steps: int = 10
    report_to: list = field(default_factory=list)
    run_name: str = f"completions-only-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    dataset_path: str = "./json_no_reasoning_nethack_dataset"
    max_samples: Optional[int] = None

class CompletionsOnlyTrainer:
    """Trainer that only computes loss on completions"""
    
    def __init__(self, config: CompletionsTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(42)
        
        logger.info(f"🚀 Completions-Only Training")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Model: {config.model_name}")
    
    def load_and_prepare_data(self):
        """Load dataset"""
        logger.info("📚 Loading dataset...")
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
        """Load model and tokenizer"""
        logger.info("🤖 Loading model and tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>"
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
        )
        
        model.resize_token_embeddings(len(tokenizer))
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        logger.info("✅ Model loaded")
        return model, tokenizer
    
    def preprocess_dataset(self, dataset, tokenizer):
        """Preprocess dataset for completions-only training"""
        logger.info("🔄 Preprocessing for completions-only training...")
        
        def tokenize_completions_only(examples):
            """Tokenize and set labels to only compute loss on completions"""
            
            input_ids_list = []
            labels_list = []
            
            for i in range(len(examples['instruction'])):
                # Get the full text (instruction already contains the chat format)
                full_text = examples['instruction'][i]
                
                # Find where the assistant response starts
                assistant_start = full_text.find("<|start_header_id|>assistant<|end_header_id|>\n\n")
                
                if assistant_start == -1:
                    # Fallback: just use the output
                    completion = examples['output'][i]
                    prompt = full_text
                else:
                    # Split at assistant start
                    prompt = full_text[:assistant_start + len("<|start_header_id|>assistant<|end_header_id|>\n\n")]
                    completion = examples['output'][i]
                
                # Tokenize prompt and completion separately
                prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
                completion_tokens = tokenizer(completion, add_special_tokens=False)['input_ids']
                
                # Combine
                input_ids = prompt_tokens + completion_tokens
                
                # Labels: -100 for prompt tokens (ignored in loss), actual IDs for completion
                labels = [-100] * len(prompt_tokens) + completion_tokens
                
                # Truncate if too long
                if len(input_ids) > self.config.model_max_length:
                    input_ids = input_ids[:self.config.model_max_length]
                    labels = labels[:self.config.model_max_length]
                
                input_ids_list.append(input_ids)
                labels_list.append(labels)
            
            return {
                'input_ids': input_ids_list,
                'labels': labels_list
            }
        
        processed_dataset = dataset.map(
            tokenize_completions_only,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing for completions-only",
        )
        
        return processed_dataset
    
    def train(self):
        """Train the model"""
        logger.info("🎯 Starting completions-only training...")
        
        try:
            dataset = self.load_and_prepare_data()
            model, tokenizer = self.load_model_and_tokenizer()
            processed_dataset = self.preprocess_dataset(dataset, tokenizer)
            
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
                
                eval_strategy=self.config.eval_strategy,
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
            
            data_collator = CompletionsOnlyDataCollator(
                tokenizer=tokenizer,
                mlm=False,
            )
            
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
                logger.info(f"🔍 GPU Memory: {allocated:.2f} GB")
            
            logger.info("🚂 Starting training (loss computed on completions only)...")
            trainer.train()
            
            logger.info("💾 Saving model...")
            trainer.save_model()
            tokenizer.save_pretrained(self.config.output_dir)
            
            with open(f"{self.config.output_dir}/training_config.json", "w") as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
            
            logger.info("✅ Completions-only training completed!")
            
            # Test the model
            self.test_model(model, tokenizer)
            
            return trainer
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def test_model(self, model, tokenizer):
        """Test the trained model"""
        logger.info("🧪 Testing completions-only model...")
        
        test_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert NetHack player.<|eot_id|><|start_header_id|>user<|end_header_id|>

Player is on dungeon level 1. Stairs are visible to the south. Choose action:

Respond with JSON:
{"action": "chosen_action"}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        
        logger.info(f"🎯 Test response: {response}")
        
        try:
            parsed = json.loads(response.strip())
            logger.info(f"✅ Valid JSON: {parsed}")
        except:
            logger.info(f"⚠️  Not valid JSON")

def main():
    print("🎮 Completions-Only NetHack Training")
    print("=" * 50)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    config = CompletionsTrainingConfig()
    
    print(f"\n📋 Configuration:")
    print(f"   - Model: {config.model_name}")
    print(f"   - Loss computed on: COMPLETIONS ONLY")
    print(f"   - Epochs: {config.num_train_epochs}")
    print(f"   - Output: {config.output_dir}")
    
    trainer = CompletionsOnlyTrainer(config)
    trainer.train()
    
    print(f"\n🎉 Completions-only training completed!")
    print(f"📁 Model saved to: {config.output_dir}")

if __name__ == "__main__":
    main()