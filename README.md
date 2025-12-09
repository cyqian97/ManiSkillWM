# Video World Model for RL Sample Efficiency

This repository explores how video world models can improve sample efficiency in reinforcement learning for robotic manipulation tasks. Built on top of [ManiSkill](https://github.com/haosulab/ManiSkill), it provides environments and training scripts for vision-based manipulation using Proximal Policy Optimization (PPO).

## Repository Structure

```
ManiSkillWM/
├── mani_skill/                    # Extended ManiSkill environments
│   └── envs/tasks/digital_twins/  # Custom manipulation tasks
├── examples/baselines/ppo/        # PPO implementation
│   └── ppo_rgb.py                 # Vision-based PPO trainer
├── experiments/                   # Training scripts
│   ├── run_ppo_can.sh            # Train coke can grasping
│   ├── run_ppo_spoon.sh          # Train spoon picking
│   ├── run_ppo_carrot.sh         # Train carrot picking
│   └── run_ppo_eggplant.sh       # Train eggplant picking
└── test_simplerenv.py            # Environment testing script
```


## Installation
Maniskill image rendering requires Vulkan. The experiements are conducted within a docker container.
To reproduce the experiment results, set up the enviroment as follows:
```
docker compose up
# In the docker container
cd ~
apt install tmux -y
https://github.com/cyqian97/ManiSkillWM.git
cd ManiSkillWM
conda create -n mnsk python=3.12 -y
conda activate mnsk
pip install -e .
```

## Test Environments

Run the test script to verify environment setup and visualize observations:

```bash
python test_simplerenv.py
```

This will:
- Create a `GraspSingleOpenedCokeCanInScene-v0` environment with 2 parallel instances
- Execute random actions and save camera observations to `test_results/`
- Record episode trajectories (.h5) and videos (.mp4)

## Train Policies

Launch training using the experiment scripts in `experiments/`:

```bash
# Train coke can grasping 
bash experiments/run_ppo_can.sh YOUR_WANDB_API_KEY

# Train carrot picking 
bash experiments/run_ppo_carrot.sh YOUR_WANDB_API_KEY

# Train eggplant picking 
bash experiments/run_ppo_eggplant.sh YOUR_WANDB_API_KEY

# Train spoon picking 
bash experiments/run_ppo_spoon.sh YOUR_WANDB_API_KEY
```

Each script:
- Runs training in a detached tmux session for long-running experiments
- Logs metrics to Weights & Biases
- Saves checkpoints to `runs/{experiment_name}/`
- Records evaluation videos periodically

**Training Configuration:**
- 512 parallel environments with GPU acceleration
- 10M total timesteps (~781 iterations)
- RGB observations
- Evaluation every 25 iterations

## Training results
The final checkpoints of the trainings are upload to [google drrive](https://drive.google.com/drive/folders/1ttrF8IOLcERmnzvxaIr4aXjTO-_UCouS?usp=drive_link).
To evaluate them, use the following command:
```
conda activate mnsk && python examples/baselines/ppo/ppo_rgb.py --env_id="Env_Name" \
   --evaluate --checkpoint=path/to/model.pt \
   --num_eval_envs=1 --num-eval-steps=1000
```