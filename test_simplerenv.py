import gymnasium as gym
import matplotlib.pyplot as plt
# from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *
from mani_skill.utils.wrappers.record import RecordEpisode

def get_image_from_maniskill3_obs_dict(env, obs, camera_name=None):
    import torch
    # obtain image from observation dictionary returned by ManiSkill environment
    if camera_name is None:
        if "google_robot" in env.unwrapped.robot_uids.uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.unwrapped.robot_uids.uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    img = obs["sensor_data"][camera_name]["rgb"]
    return img.to(torch.uint8)

env = gym.make(
  "PutSpoonOnTableClothInSceneReward-v0",
  # obs_mode="rgb+segmentation",
  obs_mode="rgb",
  num_envs=1, # if num_envs > 1, GPU simulation backend is used.
  render_mode="rgb_array",
  reward_mode="normalized_dense",
)

# Wrap with RecordEpisode to record trajectories and/or videos
env = RecordEpisode(
    env,
    output_dir="./",           # Directory to save recordings
    save_trajectory=True,                 # Save trajectory data (.h5 + .json)
    save_video=True,                      # Save videos (.mp4)
    max_steps_per_video=125,             # Required for GPU parallel envs
    video_fps=30,                         # Video frame rate
    trajectory_name="episode",            # Name prefix for saved files
)

obs, _ = env.reset()
# returns language instruction for each parallel env
# instruction = env.unwrapped.get_language_instruction()
# print("instruction:", instruction[0])

# Get the first image and display it
# image = get_image_from_maniskill3_obs_dict(env, obs) # this is the image observation for policy inference

i = 0

while True:
    # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
    # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
    # image = get_image_from_maniskill3_obs_dict(env, obs) # this is the image observation for policy inference

    action = env.action_space.sample() # replace this with your policy inference
    for j in range(len(action)): action[j]=0.0  # test the limits
    action[-1] = -1.0
    obs, reward, terminated, truncated, info = env.step(action)
    print("Reward:", reward)
    if truncated.any():
        break
print("Episode Info", info)
env.close() 