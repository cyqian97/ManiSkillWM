#!/bin/bash

# Set log file (default to exp.log if not provided)
LOG_FILE="MyTestEnv_ppo_0.log"

# Set CUDA device (default to 0 if not provided)
CUDA_DEVICE="1"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# Set W&B API key (optional: set this to your API key to skip wandb login)
MY_WANDB_API_KEY="${1:-your_default_wandb_api_key}"

# Start tmux session and run the command in window 0, logging output to the specified file
tmux new-session -d -s ppo -x 200 -y 50

tmux send-keys -t ppo:0 "export WANDB_API_KEY=$MY_WANDB_API_KEY" Enter

# Build the command with line breaks for readability
CMD="conda activate mnsk && python examples/baselines/ppo/ppo_rgb.py \
  --env_id=\"PutSpoonOnTableClothInSceneReward-v0\" \
  --num_envs=512 \
  --control_mode=\"arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos\" \
  --update_epochs=4 \
  --num_minibatches=8 \
  --total_timesteps=20_000_000 \
  --eval_freq=10 \
  --num-steps=20 \
  --exp_name=\"PutSpoon-ppo_rgb-slower\" \
  --wandb_entity=\"chad_qian_tamu\" \
  --wandb_project_name=\"ManiSkill-PPO\" \
  --wandb_group=\"PutSpoon-Experiments\" \
  --track"

tmux send-keys -t ppo:0 "$CMD" Enter

echo "Command started in tmux session 'ppo' window 0"
echo "Output is being logged to $LOG_FILE"

# attach to the tmux session with:
# tmux attach -t ppo