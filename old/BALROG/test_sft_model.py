#!/usr/bin/env python3
"""
Test NetHack SFT Model
======================

Quick testing script for the trained NetHack SFT model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetHackModelTester:
    """Test the trained NetHack SFT model"""
    
    def __init__(self, model_path: str = "./nethack-sft-model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🔍 Loading model from: {model_path}")
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load base model and adapter
            base_model_name = "microsoft/DialoGPT-small"  # Should match training
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            logger.info("✅ Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_response(self, instruction: str, max_length: int = 100) -> str:
        """Generate response for a NetHack instruction"""
        
        # Format prompt
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "### Response:\n" in response:
            response = response.split("### Response:\n")[1].strip()
        
        return response
    
    def test_nethack_scenarios(self):
        """Test various NetHack scenarios"""
        
        test_scenarios = [
            {
                "name": "Early Exploration",
                "instruction": """You are a novice NetHack player. Current subgoal: find_stairs

Game State:
Map:
-----------
|..@......|
|.........|
|....#....|
-----------

Status:
Agent the Samurai
HP:16(16) Pw:5(5) AC:10 Xp:1/0 T:1

Choose the best action from: north, south, east, west, search, pickup, wield, eat

Action:"""
            },
            {
                "name": "Item Collection",
                "instruction": """You are a novice NetHack player. Current subgoal: collect_equipment

Game State:
Map:
-----------
|..@).....|
|.........|
|....#....|
-----------

Status:
Agent the Samurai
HP:16(16) Pw:5(5) AC:10 Xp:1/0 T:15
You see here a long sword.

Choose the best action:

Action:"""
            },
            {
                "name": "Stairs Found",
                "instruction": """You are a novice NetHack player. Current subgoal: level_progression

Game State:
Map:
-----------
|..@.....<|
|.........|
|....#....|
-----------

Status:
Agent the Samurai
HP:16(16) Pw:5(5) AC:10 Xp:1/20 T:100

Choose the best action:

Action:"""
            }
        ]
        
        print("🎮 Testing NetHack SFT Model")
        print("=" * 50)
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🧪 Test {i}: {scenario['name']}")
            print("-" * 30)
            
            try:
                response = self.generate_response(scenario['instruction'])
                print(f"🤖 Model Response: {response}")
                
                # Try to extract just the action
                lines = response.split('\n')
                for line in lines:
                    line = line.strip().lower()
                    if any(action in line for action in ['north', 'south', 'east', 'west', 'search', 'pickup', 'wield', 'eat', 'down']):
                        print(f"🎯 Extracted Action: {line}")
                        break
                        
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n✅ Testing completed!")
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n🎮 Interactive NetHack Testing Mode")
        print("Enter NetHack scenarios or 'quit' to exit")
        print("-" * 40)
        
        while True:
            try:
                instruction = input("\n📝 Enter instruction: ")
                if instruction.lower() in ['quit', 'exit', 'q']:
                    break
                
                if instruction.strip():
                    response = self.generate_response(instruction)
                    print(f"🤖 Response: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Goodbye!")


def main():
    """Main testing function"""
    try:
        # Initialize tester
        tester = NetHackModelTester()
        
        # Run automated tests
        tester.test_nethack_scenarios()
        
        # Optional: Interactive testing
        response = input("\n🤔 Run interactive testing? (y/n): ")
        if response.lower().startswith('y'):
            tester.interactive_test()
            
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        print(f"\n💡 Make sure you've trained the model first:")
        print(f"   python train_nethack_sft.py")


if __name__ == "__main__":
    main()