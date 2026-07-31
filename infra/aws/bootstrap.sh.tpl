#!/bin/bash
# Boots the training container on first instance start.
# Templated by Terraform (main.tf) — ${git_repo_url}, ${git_ref}, ${wandb_api_key}.
set -euxo pipefail

git clone --branch "${git_ref}" --depth 1 "${git_repo_url}" /opt/sequential-muzero
cd /opt/sequential-muzero

docker build -t sequential-muzero:gpu .

WANDB_MODE="disabled"
if [ -n "${wandb_api_key}" ]; then
  WANDB_MODE="online"
fi

docker run -d \
  --gpus all \
  --network host \
  -e WANDB_API_KEY="${wandb_api_key}" \
  -v /opt/sequential-muzero/checkpoints:/workspace/checkpoints \
  sequential-muzero:gpu \
  model=smax mcts=joint train.wandb_mode=$WANDB_MODE
