#!/bin/bash
# Run DreamCoder agent on NetHack only

# Number of episodes to run
EPISODES=50

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Error: Not in a virtual environment. Please activate the BALROG environment first."
    echo "Try: source .venv/bin/activate or conda activate balrog"
    exit 1
fi

echo "Starting DreamCoder evaluation on NetHack..."
echo "This will run $EPISODES episodes to track learning progress"
echo "Using Python from: $(which python3)"

python3 eval.py \
  agent.type=dreamcoder \
  agent.max_image_history=0 \
  agent.max_text_history=16 \
  eval.num_workers=1 \
  eval.num_episodes.nle=$EPISODES \
  eval.num_episodes.minihack=0 \
  eval.num_episodes.babyai=0 \
  eval.num_episodes.crafter=0 \
  eval.num_episodes.textworld=0 \
  eval.num_episodes.babaisai=0 \
  envs.names=nle \
  tasks.nle_tasks=["NetHackChallenge-v0"] \
  client.client_name=vllm \
  client.model_id=meta-llama/Llama-3.2-1B-Instruct \
  client.base_url=http://0.0.0.0:8080/v1

echo "Evaluation complete! Analyzing results..."

# Run analysis
python3 test_dreamcoder.py --analyze
