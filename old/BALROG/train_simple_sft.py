#!/usr/bin/env python3
"""
Simple NetHack SFT Training
============================

Simplified training without quantization for compatibility.
"""

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
from peft import LoraConfig, TaskType, get_peft_model
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleTrainingConfig:
    """Simple training configuration"""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"
    model_max_length: int = 1024  # Reduced for compatibility
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 8  # Smaller for compatibility
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: ["c_attn", "c_proj"])
    
    # Training settings
    output_dir: str = "./simple-nethack-sft"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Memory optimizations
    gradient_checkpointing: bool = True
    fp16: bool = True
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 2
    
    # Logging
    logging_steps: int = 10
    report_to: str = None
    run_name: str = f"simple-nethack-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Data settings
    dataset_path: str = "./enhanced_nethack_dataset"
    max_samples: Optional[int] = 200  # Small sample for testing

class SimpleNetHackTrainer:
    """Simple NetHack trainer"""
    
    def __init__(self, config: SimpleTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(42)
        
        logger.info(f"🚀 Simple NetHack SFT Training")
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
        
        logger.info(f"   - Training: {len(dataset['train'])}")
        logger.info(f"   - Validation: {len(dataset['validation'])}")
        
        return dataset
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        logger.info("🤖 Loading model and tokenizer...")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
        )
        
        # Move to GPU
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # LoRA
        if self.config.use_lora:
            logger.info("⚡ Setting up LoRA...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        return model, tokenizer
    
    def preprocess_dataset(self, dataset, tokenizer):
        """Preprocess dataset"""
        logger.info("🔄 Preprocessing dataset...")
        
        def tokenize_function(examples):
            """Tokenize examples"""
            texts = []
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                response = examples['output'][i]
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
                texts.append(text)
            
            tokenized = tokenizer(
                texts,
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
            desc="Tokenizing",
        )
        
        return processed_dataset
    
    def train(self):
        """Train the model"""
        logger.info("🎯 Starting training...")
        
        # Skip wandb for simplicity
        
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
                
                evaluation_strategy=self.config.evaluation_strategy,
                eval_steps=self.config.eval_steps,
                save_strategy=self.config.save_strategy,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                
                logging_steps=self.config.logging_steps,
                report_to=self.config.report_to,
                run_name=self.config.run_name,
                
                remove_unused_columns=False,
                prediction_loss_only=True,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
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
            
            # Check memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"🔍 GPU Memory: {allocated:.2f} GB")
            
            # Train
            logger.info("🚂 Starting training...")
            trainer.train()
            
            # Save
            logger.info("💾 Saving model...")
            trainer.save_model()
            tokenizer.save_pretrained(self.config.output_dir)
            
            with open(f"{self.config.output_dir}/config.json", "w") as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
            
            logger.info("✅ Training completed!")
            
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"🔍 Final GPU Memory: {final_memory:.2f} GB")
            
            return trainer
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise


def main():
    """Main training"""
    print("🎮 Simple NetHack SFT Training")
    print("=" * 40)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️  No GPU available")
    
    # Configuration
    config = SimpleTrainingConfig()
    
    print(f"\n📋 Configuration:")
    print(f"   - Model: {config.model_name}")
    print(f"   - Batch size: {config.per_device_train_batch_size}")
    print(f"   - Max samples: {config.max_samples}")
    print(f"   - Epochs: {config.num_train_epochs}")
    print(f"   - LoRA: {config.use_lora}")
    print(f"   - Output: {config.output_dir}")
    
    # Train
    trainer = SimpleNetHackTrainer(config)
    trainer.train()
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()