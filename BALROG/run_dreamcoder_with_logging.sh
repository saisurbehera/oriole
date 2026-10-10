#!/bin/bash
# Run DreamCoder with full logging

EPISODES=10
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="dreamcoder_run_${TIMESTAMP}.log"

echo "Starting DreamCoder evaluation with logging..."
echo "Logs will be saved to: $LOG_FILE"
echo ""

# Run with all output captured
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
  tasks.nle_tasks='["NetHackChallenge-v0"]' \
  client.client_name=vllm \
  client.model_id=meta-llama/Llama-3.2-1B-Instruct \
  client.base_url=http://0.0.0.0:8080/v1 \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  hydra.job.chdir=False \
  2>&1 | tee $LOG_FILE

echo ""
echo "Run complete! Check $LOG_FILE for details"

# Show summary
echo ""
echo "=== Quick Summary ==="
grep -E "\[DreamCoder\]|Score:|episode_return" $LOG_FILE | tail -20