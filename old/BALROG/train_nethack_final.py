#!/usr/bin/env python3
"""
Final NetHack SFT Training - FIXED VERSION
==========================================

Working SFT training for RTX 4090 with proper data collation.
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

class FixedDataCollator(DataCollatorForLanguageModeling):
    """Fixed data collator for variable length sequences"""
    
    def torch_call(self, examples):
        # Simple fix: ensure consistent tensor shapes
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            max_length=512,  # Fixed max length
            return_tensors="pt",
        )
        
        # Handle labels properly
        if "labels" in batch:
            batch["labels"] = batch["input_ids"].clone()
            # Mask padding tokens in labels
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch

@dataclass
class FinalTrainingConfig:
    """Final working training configuration"""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"
    model_max_length: int = 512  # Conservative for stability
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: ["c_attn", "c_proj"])
    
    # Training settings
    output_dir: str = "./final-nethack-sft"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Memory optimizations
    gradient_checkpointing: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 0  # Avoid multiprocessing issues
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    
    # Logging
    logging_steps: int = 10
    report_to: str = None
    run_name: str = f"final-nethack-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Data settings
    dataset_path: str = "./enhanced_nethack_dataset"
    max_samples: Optional[int] = 500  # Manageable size for initial training

class FinalNetHackTrainer:
    """Final working NetHack trainer"""
    
    def __init__(self, config: FinalTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(42)
        
        logger.info(f"🚀 Final NetHack SFT Training")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Model: {config.model_name}")
        logger.info(f"   - LoRA: {config.use_lora}")
    
    def load_and_prepare_data(self):
        """Load dataset"""
        logger.info("📚 Loading NetHack expert dataset...")
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
        
        # LoRA setup
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
        """Preprocess dataset with fixed tokenization"""
        logger.info("🔄 Preprocessing dataset...")
        
        def tokenize_function(examples):
            """Fixed tokenization function"""
            texts = []
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                response = examples['output'][i]
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{tokenizer.eos_token}"
                texts.append(text)
            
            # Fixed tokenization with proper padding
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,  # Don't pad here, do it in data collator
                max_length=self.config.model_max_length,
                return_tensors=None,
            )
            
            # Set labels (will be handled properly by data collator)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Process dataset
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
        
        # Skip wandb for now
        
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
            
            # Fixed data collator
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
            
            # Final memory check
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"🔍 Final GPU Memory: {final_memory:.2f} GB")
            
            return trainer
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise


def main():
    """Main training function"""
    print("🎮 Final NetHack SFT Training")
    print("=" * 40)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️  Warning: No GPU available")
        return
    
    # Configuration
    config = FinalTrainingConfig()
    
    print(f"\n📋 Training Configuration:")
    print(f"   - Model: {config.model_name}")
    print(f"   - Batch size: {config.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   - Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"   - Max samples: {config.max_samples}")
    print(f"   - Epochs: {config.num_train_epochs}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - LoRA: {config.use_lora}")
    print(f"   - Output: {config.output_dir}")
    
    # Train
    trainer = FinalNetHackTrainer(config)
    trainer.train()
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {config.output_dir}")
    print(f"📊 Check training progress at: https://wandb.ai")
    print(f"\n🧪 Next steps:")
    print(f"   1. Test model: python test_sft_model.py")
    print(f"   2. Integrate with MCTS: python integrate_sft_mcts.py")
    print(f"   3. Run evaluation on NetHack environment")


if __name__ == "__main__":
    main()