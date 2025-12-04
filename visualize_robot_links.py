#!/usr/bin/env python3
"""
Script to visualize a robot and list all its links in ManiSkill
Usage: python visualize_robot_links.py --robot-uid widowx250s
"""

import argparse
import gymnasium as gym
import numpy as np

def list_robot_links(robot_uid="widowx250s", render=False):
    """List all links in a robot and optionally render it"""

    print(f"\n{'='*60}")
    print(f"Robot: {robot_uid}")
    print(f"{'='*60}\n")

    # Create an environment with the robot (using MyTestEnv as example)
    render_mode = "human" if render else None
    import mani_skill.envs
    env = gym.make(
        "MyTestEnv-v0",
        render_mode=render_mode,
        num_envs=1
    )

    env.reset()

    # Get robot instance
    robot = env.unwrapped.agent.robot

    # Get base_link for reference
    base_link = robot.links_map["base_link"]
    base_pose = base_link.pose
    base_pos = base_pose.p[0].cpu().numpy()

    # List all links
    print("Available Links (in WORLD frame):")
    print("-" * 60)
    for idx, (link_name, link_obj) in enumerate(robot.links_map.items()):
        # Get link pose in world frame
        world_pose = link_obj.pose
        world_pos = world_pose.p[0].cpu().numpy()  # Position in world frame

        # Calculate relative to base_link
        rel_pos = world_pos - base_pos

        print(f"{idx+1:2d}. {link_name:30s} | world: [{world_pos[0]:7.4f}, {world_pos[1]:7.4f}, {world_pos[2]:7.4f}] | rel_base: [{rel_pos[0]:7.4f}, {rel_pos[1]:7.4f}, {rel_pos[2]:7.4f}]")

    print("\n" + "="*60)
    print(f"Total links: {len(robot.links_map)}")
    print(f"Base link position (world frame): [{base_pos[0]:7.4f}, {base_pos[1]:7.4f}, {base_pos[2]:7.4f}]")
    print("="*60 + "\n")

    # List all joints
    print("\nJoint Information:")
    print("-" * 60)
    for idx, joint in enumerate(robot.get_joints()):
        print(f"{idx+1:2d}. {joint.name:30s} | type: {joint.type}")

    print("\n" + "="*60)
    print(f"Total joints: {len(robot.get_joints())}")
    print("="*60 + "\n")

    # Controller information
    if hasattr(env.unwrapped.agent, 'controller'):
        controller = env.unwrapped.agent.controller
        if hasattr(controller, 'controllers'):
            arm_controller = controller.controllers.get('arm')
            if arm_controller and hasattr(arm_controller, 'ee_link'):
                print(f"\nConfigured End Effector Link: {arm_controller.ee_link.name}")
                ee_pos = arm_controller.ee_link.pose.p[0].cpu().numpy()
                print(f"EE Position: [{ee_pos[0]:7.4f}, {ee_pos[1]:7.4f}, {ee_pos[2]:7.4f}]")

    if render:
        print("\n\nRendering robot. Press Ctrl+C to exit.")
        try:
            for _ in range(10000):
                env.step(np.zeros(env.action_space.shape))
                env.render()
        except KeyboardInterrupt:
            print("\nExiting...")

    env.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize robot and list all links")
    parser.add_argument(
        "--robot-uid",
        type=str,
        default="widowx250s",
        help="Robot UID to visualize (e.g., 'widowx250s', 'panda', 'fetch')"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the robot in GUI"
    )

    args = parser.parse_args()

    list_robot_links(args.robot_uid, args.render)

if __name__ == "__main__":
    main()
