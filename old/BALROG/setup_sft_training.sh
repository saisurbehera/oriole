#!/bin/bash
# NetHack SFT Training Setup Script
# Optimized for RTX 4090

echo "🎮 NetHack SFT Training Setup"
echo "=============================="

# Check for CUDA
echo "🔍 Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected. Training will be very slow on CPU."
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements_sft.txt

# Verify installations
echo ""
echo "🔍 Verifying key installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"

# Check dataset
echo ""
echo "📊 Checking dataset..."
if [ -d "enhanced_nethack_dataset" ]; then
    echo "✅ Enhanced NetHack dataset found"
    python -c "from datasets import load_from_disk; ds = load_from_disk('enhanced_nethack_dataset'); print(f'Train: {len(ds[\"train\"])}, Val: {len(ds[\"validation\"])}')"
else
    echo "❌ Dataset not found. Please run enhanced_expert_data_processor.py first"
    exit 1
fi

# Setup wandb (optional)
echo ""
echo "📊 Setting up Weights & Biases (optional)..."
read -p "Do you want to setup wandb logging? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v wandb &> /dev/null; then
        wandb login
    else
        echo "⚠️  wandb not installed. Installing..."
        pip install wandb
        wandb login
    fi
else
    echo "⏭️  Skipping wandb setup"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Ready to start training:"
echo "   python train_nethack_sft.py"
echo ""
echo "📋 Training will:"
echo "   - Use LoRA for memory efficiency"
echo "   - Apply 4-bit quantization for RTX 4090"
echo "   - Train for 3 epochs with cosine LR schedule"
echo "   - Save checkpoints every 200 steps"
echo "   - Log to wandb (if configured)"
echo ""
echo "💾 After training, test with:"
echo "   python test_sft_model.py"