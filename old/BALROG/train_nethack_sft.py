#!/usr/bin/env python3
"""
NetHack Expert SFT Training Script
==================================

Optimized HuggingFace SFT training for single RTX 4090 (24GB VRAM).
Uses LoRA, gradient checkpointing, and other memory optimizations.
"""

import os
import torch
import wandb
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NetHackTrainingConfig:
    """Training configuration for NetHack SFT"""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"  # Start with small model for testing
    model_max_length: int = 2048
    
    # LoRA settings (for memory efficiency)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: ["c_attn", "c_proj"])
    
    # Quantization settings (4-bit for 4090)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # Training settings
    output_dir: str = "./nethack-sft-model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2  # Conservative for 4090
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective batch size = 2*8 = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # Memory optimizations
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Logging
    logging_steps: int = 10
    report_to: str = "wandb"
    run_name: str = f"nethack-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Data settings
    dataset_path: str = "/home/sai/Desktop/oriole/BALROG/enhanced_nethack_dataset"
    max_samples: Optional[int] = None  # Use all available data
    
    # Instruction format
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


class NetHackSFTTrainer:
    """NetHack SFT trainer with 4090 optimizations"""
    
    def __init__(self, config: NetHackTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        set_seed(42)
        
        logger.info(f"🚀 Initializing NetHack SFT Training")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Model: {config.model_name}")
        logger.info(f"   - LoRA: {config.use_lora}")
        logger.info(f"   - 4-bit quantization: {config.use_4bit}")
        
    def load_and_prepare_data(self):
        """Load and prepare the NetHack dataset"""
        logger.info("📚 Loading NetHack expert dataset...")
        
        # Load dataset
        dataset = load_from_disk(self.config.dataset_path)
        
        # Limit samples if specified
        if self.config.max_samples:
            train_samples = min(self.config.max_samples, len(dataset['train']))
            val_samples = min(self.config.max_samples // 10, len(dataset['validation']))
            
            dataset['train'] = dataset['train'].select(range(train_samples))
            dataset['validation'] = dataset['validation'].select(range(val_samples))
        
        logger.info(f"   - Training samples: {len(dataset['train'])}")
        logger.info(f"   - Validation samples: {len(dataset['validation'])}")
        
        return dataset
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with optimizations"""
        logger.info("🤖 Loading model and tokenizer...")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side="right",
            use_fast=True,
        )
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Quantization config for 4090
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
        else:
            quantization_config = None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Prepare for training
        if self.config.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Setup LoRA
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
        """Preprocess dataset for training"""
        logger.info("🔄 Preprocessing dataset...")
        
        def format_example(example):
            """Format example using instruction template"""
            instruction = example['instruction']
            response = example['output']
            
            # Format with template
            text = self.config.instruction_template.format(
                instruction=instruction,
                response=response
            )
            
            return {"text": text}
        
        def tokenize_function(examples):
            """Tokenize examples"""
            # Handle batched format from datasets
            texts = []
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                response = examples['output'][i]
                text = self.config.instruction_template.format(
                    instruction=instruction,
                    response=response
                )
                texts.append(text)
            
            # Tokenize
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config.model_max_length,
                return_tensors=None,
            )
            
            # Set labels for language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Apply preprocessing
        processed_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset",
        )
        
        return processed_dataset
    
    def setup_training_arguments(self):
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            
            # Memory optimizations
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            remove_unused_columns=self.config.remove_unused_columns,
            
            # Evaluation and saving
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            
            # Logging
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
            
            # Performance
            fp16=True,  # Use FP16 for 4090
            dataloader_num_workers=4,
            
            # Disable some features to save memory
            prediction_loss_only=True,
        )
    
    def train(self):
        """Main training loop"""
        logger.info("🎯 Starting NetHack SFT training...")
        
        # Initialize wandb
        if self.config.report_to == "wandb":
            wandb.init(
                project="nethack-sft",
                name=self.config.run_name,
                config=self.config.__dict__
            )
        
        try:
            # Load data
            dataset = self.load_and_prepare_data()
            
            # Load model and tokenizer
            model, tokenizer = self.load_model_and_tokenizer()
            
            # Preprocess dataset
            processed_dataset = self.preprocess_dataset(dataset, tokenizer)
            
            # Setup training arguments
            training_args = self.setup_training_arguments()
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=processed_dataset["train"],
                eval_dataset=processed_dataset["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            # Check memory before training
            if torch.cuda.is_available():
                logger.info(f"🔍 GPU Memory before training:")
                logger.info(f"   - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                logger.info(f"   - Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            # Train the model
            logger.info("🚂 Starting training...")
            trainer.train()
            
            # Save the final model
            logger.info("💾 Saving final model...")
            trainer.save_model()
            tokenizer.save_pretrained(self.config.output_dir)
            
            # Save training config
            with open(f"{self.config.output_dir}/training_config.json", "w") as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
            
            logger.info("✅ Training completed successfully!")
            
            # Final memory check
            if torch.cuda.is_available():
                logger.info(f"🔍 GPU Memory after training:")
                logger.info(f"   - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                logger.info(f"   - Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            return trainer
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        finally:
            # Cleanup
            if self.config.report_to == "wandb":
                wandb.finish()


def main():
    """Main training script"""
    print("🎮 NetHack Expert SFT Training")
    print("=" * 50)
    
    # Configuration
    config = NetHackTrainingConfig()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, using CPU (will be very slow)")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Adjust settings based on GPU memory
        if gpu_memory < 12:  # Less than 12GB
            config.per_device_train_batch_size = 1
            config.gradient_accumulation_steps = 16
            config.use_4bit = True
            print("🔧 Adjusted settings for <12GB GPU")
        elif gpu_memory >= 24:  # 24GB+ (like 4090)
            config.per_device_train_batch_size = 2
            config.gradient_accumulation_steps = 8
            print("🔧 Optimized settings for 24GB+ GPU")
    
    # Print configuration
    print(f"\n📋 Training Configuration:")
    print(f"   - Model: {config.model_name}")
    print(f"   - Batch size: {config.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   - Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Epochs: {config.num_train_epochs}")
    print(f"   - LoRA: {config.use_lora}")
    print(f"   - 4-bit quantization: {config.use_4bit}")
    print(f"   - Output dir: {config.output_dir}")
    
    # Initialize trainer
    trainer = NetHackSFTTrainer(config)
    
    # Start training
    trained_model = trainer.train()
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {config.output_dir}")
    print(f"🧪 Next steps:")
    print(f"   1. Test the model with NetHack prompts")
    print(f"   2. Integrate with MCTS agent")
    print(f"   3. Run evaluation on NetHack environment")


if __name__ == "__main__":
    main()