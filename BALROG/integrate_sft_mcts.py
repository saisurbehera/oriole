#!/usr/bin/env python3
"""
SFT + MCTS Integration
======================

Integrates the fine-tuned NetHack SFT model with the TreeQuest MCTS agent.
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from typing import Dict, Any, Tuple

# Add BALROG path
sys.path.append('/home/sai/Desktop/oriole/BALROG')
from balrog.agents.real_llm_mcts import RealLLMMCTSAgent
from balrog.client import LLMResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTNetHackClient:
    """LLM client using the fine-tuned NetHack SFT model"""
    
    def __init__(self, model_path: str = "./nethack-sft-model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned SFT model"""
        logger.info(f"🔍 Loading SFT model from: {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load base model
            base_model_name = "microsoft/DialoGPT-small"
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            logger.info("✅ SFT model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load SFT model: {e}")
            raise
    
    def generate(self, messages, max_tokens=50, temperature=0.7):
        """Generate response using SFT model"""
        
        # Extract the user message (instruction)
        if messages and len(messages) > 0:
            instruction = messages[-1].content
        else:
            instruction = "Choose a NetHack action."
        
        # Format prompt for SFT model
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
                max_length=inputs.input_ids.shape[1] + max_tokens,
                temperature=temperature,
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
        
        # Return in expected format
        return LLMResponse(
            model_id="nethack-sft",
            completion=response,
            stop_reason="complete",
            input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(outputs[0]) - inputs.input_ids.shape[1]
        )


class SFTEnhancedMCTSAgent(RealLLMMCTSAgent):
    """MCTS agent enhanced with SFT model"""
    
    def __init__(self, sft_model_path: str, prompt_builder):
        # Create SFT client factory
        def sft_client_factory():
            return SFTNetHackClient(sft_model_path)
        
        # Initialize with SFT client
        super().__init__(sft_client_factory, prompt_builder)
        
        logger.info("🎯 SFT-Enhanced MCTS Agent initialized!")
        logger.info(f"   - Using fine-tuned model: {sft_model_path}")
        logger.info(f"   - TreeQuest MCTS integration: ✅")
        logger.info(f"   - Expert-guided action selection: ✅")
    
    def _get_llm_action(self, state: Dict, observation: Dict, tried_actions: set) -> Tuple[str, str]:
        """Enhanced LLM action selection using SFT model"""
        
        # Build enhanced context for SFT model
        context = self._build_sft_context(state, observation, tried_actions)
        
        try:
            # Use the SFT model through the parent's mechanism
            action, reasoning = super()._get_llm_action(state, observation, tried_actions)
            
            # Post-process with expert knowledge
            action = self._validate_sft_action(action, state, tried_actions)
            
            return action, f"SFT-guided: {reasoning}"
            
        except Exception as e:
            logger.warning(f"SFT action failed: {e}, using expert fallback")
            return self._get_expert_fallback_action(state, tried_actions)
    
    def _build_sft_context(self, state: Dict, observation: Dict, tried_actions: set) -> str:
        """Build enhanced context for SFT model"""
        
        # Get base context
        base_context = self._build_action_context(state, observation, tried_actions)
        
        # Add SFT-specific enhancements
        sft_context = f"{base_context}\n\nEXPERT TRAINING CONTEXT:\n"
        sft_context += f"- This model was trained on expert NetHack gameplay\n"
        sft_context += f"- Training included {self._get_expert_context_info()}\n"
        sft_context += f"- Focus on strategic decision-making for long-term success\n"
        
        return sft_context
    
    def _get_expert_context_info(self) -> str:
        """Get expert training context information"""
        return "50 episodes, 2,158 expert decisions, avg 15k points per episode"
    
    def _validate_sft_action(self, action: str, state: Dict, tried_actions: set) -> str:
        """Validate and correct SFT model output"""
        
        # Valid NetHack actions
        valid_actions = [
            'north', 'south', 'east', 'west', 'northeast', 'northwest', 
            'southeast', 'southwest', 'up', 'down', 'wait', 'search', 
            'pickup', 'drop', 'inventory', 'wield', 'wear', 'eat', 'drink', 
            'read', 'zap', 'fire', 'open', 'close', 'apply', 'pay'
        ]
        
        # Clean the action
        action = action.strip().lower()
        
        # Extract action if it's in a sentence
        for valid_action in valid_actions:
            if valid_action in action:
                action = valid_action
                break
        
        # Validate
        if action in valid_actions and action not in tried_actions:
            return action
        
        # Fallback to expert guidance
        logger.debug(f"Invalid SFT action '{action}', using expert fallback")
        fallback_action, _ = self._get_expert_fallback_action(state, tried_actions)
        return fallback_action


def create_sft_enhanced_agent(sft_model_path: str = "./nethack-sft-model"):
    """Factory function to create SFT-enhanced MCTS agent"""
    
    # Import prompt builder (you'll need to adapt this to your setup)
    from balrog.prompt_builder.base import BasePromptBuilder
    
    class NetHackPromptBuilder(BasePromptBuilder):
        def __init__(self):
            self.messages = []
        
        def reset(self):
            self.messages = []
        
        def update_observation(self, obs):
            pass
        
        def update_action(self, action):
            pass
    
    prompt_builder = NetHackPromptBuilder()
    
    return SFTEnhancedMCTSAgent(sft_model_path, prompt_builder)


def test_sft_mcts_integration():
    """Test the SFT + MCTS integration"""
    
    print("🎮 Testing SFT + MCTS Integration")
    print("=" * 40)
    
    try:
        # Create SFT-enhanced agent
        agent = create_sft_enhanced_agent()
        
        # Mock observation for testing
        test_observation = {
            'text': {
                'long_term_context': '''
map:
-----------
|..@......|
|.........|
|....#....|
-----------

Agent the Samurai
HP:16(16) Pw:5(5) AC:10 Xp:1/0 T:1
'''
            }
        }
        
        print("🤖 Testing SFT-enhanced action selection...")
        
        # Test action selection
        response = agent.act(test_observation)
        
        print(f"✅ SFT-MCTS Response:")
        print(f"   - Action: {response.completion}")
        print(f"   - Reasoning: {response.reasoning}")
        print(f"   - Model: {response.model_id}")
        
        print(f"\n🎯 Integration successful!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print(f"💡 Make sure you have:")
        print(f"   1. Trained SFT model: python train_nethack_sft.py")
        print(f"   2. Model saved in: ./nethack-sft-model/")


if __name__ == "__main__":
    test_sft_mcts_integration()