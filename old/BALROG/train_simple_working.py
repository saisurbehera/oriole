#!/usr/bin/env python3
"""
Simple working SFT training based on guanaco format
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import torch
import logging
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    set_seed
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_from_disk, Dataset, DatasetDict
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleWorkingConfig:
    """Simple working configuration"""
    
    model_name: str = "microsoft/DialoGPT-small"
    model_max_length: int = 2048  # Increase to handle our long strategic prompts
    
    output_dir: str = "./simple-working-sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    
    fp16: bool = False
    
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    
    report_to: list = field(default_factory=list)
    
    dataset_path: str = "./json_no_reasoning_nethack_dataset"
    max_samples: Optional[int] = 100  # Small test first

class SimpleWorkingTrainer:
    """Simple working trainer"""
    
    def __init__(self, config: SimpleWorkingConfig):
        self.config = config
        set_seed(42)
        
        logger.info(f"🚀 Simple Working SFT Training")
        logger.info(f"   - Model: {config.model_name}")
        logger.info(f"   - Max samples: {config.max_samples}")
    
    def prepare_dataset(self):
        """Prepare dataset in guanaco format"""
        logger.info("📚 Loading and formatting dataset...")
        
        # Load original dataset
        dataset = load_from_disk(self.config.dataset_path)
        
        # Limit samples for testing
        if self.config.max_samples:
            train_samples = min(self.config.max_samples, len(dataset['train']))
            val_samples = min(self.config.max_samples // 10, len(dataset['validation']))
            
            dataset['train'] = dataset['train'].select(range(train_samples))
            dataset['validation'] = dataset['validation'].select(range(val_samples))
        
        logger.info(f"   - Training samples: {len(dataset['train'])}")
        logger.info(f"   - Validation samples: {len(dataset['validation'])}")
        
        # Format exactly like guanaco dataset
        def format_to_guanaco(examples):
            texts = []
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                output = examples['output'][i]
                
                # instruction already ends with assistant header, just add output once
                text = instruction + output
                texts.append(text)
            
            return {"text": texts}
        
        # Apply formatting and remove all other columns
        formatted_dataset = dataset.map(
            format_to_guanaco, 
            batched=True, 
            remove_columns=dataset['train'].column_names,
            desc="Formatting dataset"
        )
        
        # Verify format
        logger.info(f"📋 Formatted dataset:")
        logger.info(f"   - Columns: {formatted_dataset['train'].column_names}")
        example_text = formatted_dataset['train'][0]['text']
        logger.info(f"   - Example length: {len(example_text)}")
        logger.info(f"   - Example ending: ...{example_text[-100:]}")
        
        return formatted_dataset
    
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
        
        logger.info("✅ Model and tokenizer loaded")
        return model, tokenizer
    
    def train(self):
        """Train using SFTTrainer"""
        logger.info("🎯 Starting training...")
        
        try:
            # Load everything
            dataset = self.prepare_dataset()
            model, tokenizer = self.load_model_and_tokenizer()
            
            # SFTConfig
            sft_config = SFTConfig(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                fp16=self.config.fp16,
                eval_strategy=self.config.eval_strategy,
                eval_steps=self.config.eval_steps,
                save_steps=self.config.save_steps,
                logging_steps=self.config.logging_steps,
                report_to=self.config.report_to,
                max_seq_length=self.config.model_max_length,
                dataset_text_field="text",
                packing=False,
            )
            
            # Data collator for completion-only training
            instruction_template = "<|start_header_id|>user<|end_header_id|>"
            response_template = "<|start_header_id|>assistant<|end_header_id|>"
            collator = DataCollatorForCompletionOnlyLM(
                instruction_template=instruction_template,
                response_template=response_template,
                tokenizer=tokenizer,
                mlm=False
            )
            
            # SFTTrainer
            trainer = SFTTrainer(
                model,
                args=sft_config,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                data_collator=collator,
            )
            
            logger.info("🚂 Starting training...")
            trainer.train()
            
            logger.info("💾 Saving model...")
            trainer.save_model()
            tokenizer.save_pretrained(self.config.output_dir)
            
            logger.info("✅ Training completed!")
            
            # Quick test
            self.test_model(model, tokenizer)
            
            return trainer
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def test_model(self, model, tokenizer):
        """Quick test"""
        logger.info("🧪 Testing model...")
        
        test_prompt = """<|start_header_id|>user<|end_header_id|>

Player is on level 1. Stairs visible. Choose action:

Respond with JSON:
{"action": "chosen_action"}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=15,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        logger.info(f"🎯 Test response: '{response}'")

def main():
    print("🎮 Simple Working SFT Training")
    print("=" * 40)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🔥 GPU: {gpu_name}")
    
    config = SimpleWorkingConfig()
    trainer = SimpleWorkingTrainer(config)
    trainer.train()
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {config.output_dir}")

if __name__ == "__main__":
    main()