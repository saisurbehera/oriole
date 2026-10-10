#!/usr/bin/env python3
"""
Minimal working SFT training - simplest possible approach
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import torch
import logging
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    set_seed
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_from_disk, Dataset, DatasetDict
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MinimalConfig:
    """Minimal configuration"""
    
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Much smaller model, supports 4096 tokens
    model_max_length: int = 8192  # Extended context window
    
    output_dir: str = "./minimal-working-sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # Very small for 4096 tokens
    gradient_accumulation_steps: int = 16  # Larger accumulation for effective batch size
    learning_rate: float = 5e-5
    
    fp16: bool = False  # Disable FP16 to avoid gradient scaling issues
    gradient_checkpointing: bool = True  # Enable gradient checkpointing
    
    eval_strategy: str = "no"  # Disable eval for simplicity
    save_steps: int = 100
    logging_steps: int = 10
    
    report_to: list = field(default_factory=list)
    
    dataset_path: str = "./json_no_reasoning_nethack_dataset"
    max_samples: Optional[int] = 500  # More examples with smaller model

class MinimalTrainer:
    """Minimal trainer"""
    
    def __init__(self, config: MinimalConfig):
        self.config = config
        set_seed(42)
        
        logger.info(f"🚀 Minimal Working SFT Training")
        logger.info(f"   - Model: {config.model_name}")
        logger.info(f"   - Max samples: {config.max_samples}")
    
    def prepare_simple_dataset(self):
        """Prepare dataset in simplest possible format"""
        logger.info("📚 Creating simple dataset...")
        
        # Load original dataset
        dataset = load_from_disk(self.config.dataset_path)
        
        # Limit samples
        if self.config.max_samples:
            train_samples = min(self.config.max_samples, len(dataset['train']))
            dataset['train'] = dataset['train'].select(range(train_samples))
        
        logger.info(f"   - Training samples: {len(dataset['train'])}")
        
        # Create format with game context but simple templates
        def create_simple_format(examples):
            texts = []
            for i in range(len(examples['instruction'])):
                # Extract the strategic context from the original instruction
                original_instruction = examples['instruction'][i]
                
                # Find the strategic gameplay part (skip the chat headers)
                start_marker = "=== NetHack Strategic Gameplay ==="
                if start_marker in original_instruction:
                    # Extract from strategic gameplay onwards
                    strategic_part = original_instruction[original_instruction.find(start_marker):]
                    # Remove the chat format ending AND any duplicate JSON
                    strategic_part = strategic_part.replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "")
                    
                    # Remove duplicate JSON that appears in the instruction
                    # Look for pattern: Respond with JSON: {"action": "chosen_action"}{"action": "something"}
                    if '{"action": "chosen_action"}' in strategic_part:
                        # Find the end of the clean instruction part
                        clean_end = strategic_part.find('{"action": "chosen_action"}')
                        strategic_part = strategic_part[:clean_end].rstrip()
                else:
                    # Fallback to simple instruction
                    strategic_part = "Choose NetHack action based on game state."
                
                # Create simple format with game context
                instruction_part = f"### Instruction:\n{strategic_part}\n\n"
                response_part = f"### Response:\n{examples['output'][i]}"
                
                text = instruction_part + response_part
                texts.append(text)
            
            return {"text": texts}
        
        # Apply formatting
        simple_dataset = dataset.map(
            create_simple_format, 
            batched=True, 
            remove_columns=dataset['train'].column_names,
            desc="Creating simple format"
        )
        
        # Create a dummy validation set
        val_size = max(1, len(simple_dataset['train']) // 10)
        train_dataset = simple_dataset['train'].select(range(len(simple_dataset['train']) - val_size))
        val_dataset = simple_dataset['train'].select(range(len(simple_dataset['train']) - val_size, len(simple_dataset['train'])))
        
        formatted_dataset = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        # Verify format
        logger.info(f"📋 Simple dataset:")
        logger.info(f"   - Columns: {formatted_dataset['train'].column_names}")
        example_text = formatted_dataset['train'][0]['text']
        logger.info(f"   - Example length: {len(example_text)}")
        logger.info(f"   - Example: {example_text}")
        
        return formatted_dataset
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer without special tokens"""
        logger.info("🤖 Loading model and tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load config and fix rope_scaling issue
        config = AutoConfig.from_pretrained(self.config.model_name)
        
        # Fix rope_scaling format if it exists
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            if isinstance(config.rope_scaling, dict) and 'rope_type' in config.rope_scaling:
                # Keep all original fields but fix the format
                rope_scaling = config.rope_scaling.copy()
                rope_scaling['type'] = rope_scaling.pop('rope_type', 'linear')
                # Keep all other fields as they are
                config.rope_scaling = rope_scaling
        
        # Load model with fixed config and memory optimizations
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=config,
            torch_dtype=torch.float32,  # Use FP32 to avoid issues
            trust_remote_code=True,
            attn_implementation="eager",
            device_map="auto",  # Automatic device placement
        )
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        logger.info("✅ Model and tokenizer loaded (no modifications)")
        return model, tokenizer
    
    def train(self):
        """Train using SFTTrainer"""
        logger.info("🎯 Starting minimal training...")
        
        try:
            # Load everything
            dataset = self.prepare_simple_dataset()
            model, tokenizer = self.load_model_and_tokenizer()
            
            # Simple SFTConfig
            sft_config = SFTConfig(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                fp16=self.config.fp16,
                eval_strategy=self.config.eval_strategy,
                save_steps=self.config.save_steps,
                logging_steps=self.config.logging_steps,
                report_to=self.config.report_to,
                max_seq_length=self.config.model_max_length,
                dataset_text_field="text",
                packing=False,
            )
            
            # Simple completion-only training
            instruction_template = "### Instruction:"
            response_template = "### Response:"
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
                eval_dataset=dataset["validation"] if self.config.eval_strategy != "no" else None,
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
        
        test_prompt = """### Instruction:
Choose NetHack action based on game state.

### Response:
"""
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        logger.info(f"🎯 Test response: '{response}'")

def main():
    print("🎮 Minimal Working SFT Training")
    print("=" * 40)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🔥 GPU: {gpu_name}")
    
    config = MinimalConfig()
    trainer = MinimalTrainer(config)
    trainer.train()
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {config.output_dir}")

if __name__ == "__main__":
    main()