#!/bin/bash

# Set log file (default to exp.log if not provided)
LOG_FILE="MyTestEnv_ppo_0.log"

# Set CUDA device (default to 0 if not provided)
CUDA_DEVICE="1"
# export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# Set W&B API key (optional: set this to your API key to skip wandb login)
MY_WANDB_API_KEY="${1:-your_default_wandb_api_key}"

# PPO Training Parameters
ENV_ID="PutSpoonOnTableClothInSceneReward-v0"
CONTROL_MODE="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
NUM_ENVS=512
NUM_STEPS=25
NUM_EVAL_STEPS=125
UPDATE_EPOCHS=8
NUM_MINIBATCHES=8
TOTAL_TIMESTEPS=10_000_000
GAMMA=0.95
CHECKPOINT="runs/PutSpoon-ppo_rgb-orient_corrected-contact_check/final_ckpt.pt"
EXP_NAME="PutSpoon-ppo_rgb-orient_corrected-contact_check-second_round"
WANDB_ENTITY="chad_qian_tamu"
WANDB_PROJECT_NAME="ManiSkill-PPO"
WANDB_GROUP="PutSpoon-Experiments"



# Check if experiment folder already exists
if [ -d "./runs/$EXP_NAME" ]; then
    echo "Aborted: Experiment folder './runs/$EXP_NAME' already exists!"
    exit 1
fi

# Start tmux session and run the command in window 0, logging output to the specified file
tmux new-session -d -s ppo -x 200 -y 50


tmux send-keys -t ppo:0 "export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE" Enter
tmux send-keys -t ppo:0 "export WANDB_API_KEY=$MY_WANDB_API_KEY" Enter
# Build the command with line breaks for readability
CMD="conda activate mnsk && python examples/baselines/ppo/ppo_rgb.py \
  --env_id=\"$ENV_ID\" \
  --control_mode=\"$CONTROL_MODE\" \
  --num_envs=$NUM_ENVS \
  --num_steps=$NUM_STEPS \
  --num_eval_steps=$NUM_EVAL_STEPS \
  --update_epochs=$UPDATE_EPOCHS \
  --num_minibatches=$NUM_MINIBATCHES \
  --total_timesteps=$TOTAL_TIMESTEPS \
  --gamma=$GAMMA \
  --checkpoint=\"$CHECKPOINT\" \
  --exp_name=\"$EXP_NAME\" \
  --wandb_entity=\"$WANDB_ENTITY\" \
  --wandb_project_name=\"$WANDB_PROJECT_NAME\" \
  --wandb_group=\"$WANDB_GROUP\" \
  --track"

tmux send-keys -t ppo:0 "$CMD" Enter

echo "Command started in tmux session 'ppo' window 0"
echo "Output is being logged to $LOG_FILE"

# attach to the tmux session with:
# tmux attach -t ppo1