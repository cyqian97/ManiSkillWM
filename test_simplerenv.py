#!/usr/bin/env python3
"""Test script for GraspSingleOpenedCokeCanInScene-v0 environment"""

import gymnasium as gym
import numpy as np
import imageio
from mani_skill.utils.wrappers import CPUGymWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

test_cpu = False

def get_image_from_maniskill3_obs_dict(env, obs, camera_name=None):
    import torch
    # obtain image from observation dictionary returned by ManiSkill environment
    if camera_name is None:
        robot_name = env.unwrapped.robot_uids if isinstance(env.unwrapped.robot_uids, str) else env.unwrapped.robot_uids.uid
        if "google" in robot_name:
            camera_name = "overhead_camera"
        elif "widowx" in robot_name:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["sensor_data"][camera_name]["rgb"]

def save_obs_images(env, obs, step):
    # Get camera image
    print("\n📸 Camera observations:")
    overhead_rgb = get_image_from_maniskill3_obs_dict(env, obs)
    print(f"  overhead_camera: {overhead_rgb.shape} - {overhead_rgb.dtype}")

    # Convert to numpy array if needed
    if len(overhead_rgb.shape) == 4:
        overhead_rgb = overhead_rgb[0].cpu().numpy()
    elif hasattr(overhead_rgb, 'cpu'):
        overhead_rgb = overhead_rgb.cpu().numpy()

    # Ensure uint8 format for image saving
    if overhead_rgb.dtype != np.uint8:
        overhead_rgb = (overhead_rgb * 255).astype(np.uint8)

    # Save directly as image
    imageio.imwrite(f"test_results/camera_views_step{step}.png", overhead_rgb)
    print(f"Saved camera view to: test_results/camera_views_step{step}.png")
    
# Create environment
# Environments used in the experiments: "PickCarrotScene-v0", "PickEggplantScene-v0", "PickSpoonScene-v0", "GraspSingleOpenedCokeCanInScene-v0"
env =  gym.make(
    "PickSpoonScene-v0",
    num_envs=2,
    # obs_mode="rgb",
    obs_mode="rgb+segmentation",
    render_mode="rgb_array"
)
if test_cpu:
    env = CPUGymWrapper(env)

# Wrap with RecordEpisode to record trajectories and/or videos
env = RecordEpisode(
    env,
    output_dir="./test_results",           # Directory to save recordings
    save_trajectory=True,                 # Save trajectory data (.h5 + .json)
    save_video=True,                      # Save videos (.mp4)
    max_steps_per_video=80,             # Required for GPU parallel envs
    video_fps=3,                         # Video frame rate
    trajectory_name="episode",            # Name prefix for saved files
)

print(f"Environment created: {env.spec.id}")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")

# Reset environment
obs, info = env.reset()
print(f"Environment reset successfully")
print(f"  Robot: {env.unwrapped.agent.robot.name}")


i = 0
save_image = True
# Save initial observation images
if save_image:
    save_obs_images(env,obs, i)
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    success = info['success'][0].item() if hasattr(info['success'], 'item') else info['success']
    print(f"  Step {i+1}: success={success}, reward={reward}")
    i+=1

    # Save images every 20 steps
    if save_image and i % 20 == 0:
        save_obs_images(env,obs, i)

    if test_cpu:
        if truncated:
            break
    else:
        if truncated.any():
            break

print(f"\n✓ Test completed successfully!")
env.close()
