#!/usr/bin/env python3
"""
Script to create a 3D plot of robot link coordinate frames
Usage: python plot_robot_frames.py
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import gymnasium as gym
import mani_skill.envs


def plot_coordinate_frame(ax, position, rotation_matrix, scale=0.02, label="", alpha=0.8):
    """Plot a coordinate frame (xyz axes) at a given position with given orientation"""
    # X-axis (red), Y-axis (green), Z-axis (blue)
    colors = ['r', 'g', 'b']
    axis_labels = ['x', 'y', 'z']

    for i, (color, axis_label) in enumerate(zip(colors, axis_labels)):
        direction = rotation_matrix[:, i] * scale
        ax.quiver(position[0], position[1], position[2],
                 direction[0], direction[1], direction[2],
                 color=color, alpha=alpha, arrow_length_ratio=0.3, linewidth=2)


def plot_robot_links_3d(save_path="robot_frames_3d.png"):
    """Create 3D plot of all robot link frames"""

    print("Loading environment and robot...")
    env = gym.make("MyTestEnv-v0", num_envs=1)
    env.reset()

    robot = env.unwrapped.agent.robot

    # Collect link data
    link_names = []
    link_positions = []
    link_rotations = []

    print("Extracting link poses...")
    for link_name, link_obj in robot.links_map.items():
        pose = link_obj.pose
        pos = pose.p[0].cpu().numpy()
        # Get rotation matrix from quaternion
        rot_mat = pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()

        link_names.append(link_name)
        link_positions.append(pos)
        link_rotations.append(rot_mat)

    link_positions = np.array(link_positions)

    # Create 3D plot
    print("Creating 3D visualization...")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot link positions as points
    ax.scatter(link_positions[:, 0], link_positions[:, 1], link_positions[:, 2],
              c='black', marker='o', s=50, alpha=0.6, label='Link origins')

    # Plot coordinate frames for key links
    key_links = ['base_link', 'wrist_link', 'gripper_link', 'gripper_bar_link',
                 'fingers_link', 'ee_gripper_link', 'left_finger_link', 'right_finger_link']

    for i, (name, pos, rot) in enumerate(zip(link_names, link_positions, link_rotations)):
        if name in key_links:
            # Larger frames for key links
            plot_coordinate_frame(ax, pos, rot, scale=0.03, label=name, alpha=0.9)
            # Add text label
            ax.text(pos[0], pos[1], pos[2] + 0.02, name, fontsize=8, ha='center')
        else:
            # Smaller frames for other links
            plot_coordinate_frame(ax, pos, rot, scale=0.015, alpha=0.4)

    # Plot line connecting links (kinematic chain)
    ax.plot(link_positions[:, 0], link_positions[:, 1], link_positions[:, 2],
           'k-', alpha=0.3, linewidth=1, label='Kinematic chain')

    # Highlight TCP (ee_gripper_link)
    tcp_idx = link_names.index('ee_gripper_link')
    tcp_pos = link_positions[tcp_idx]
    ax.scatter([tcp_pos[0]], [tcp_pos[1]], [tcp_pos[2]],
              c='red', marker='*', s=300, label='TCP (ee_gripper_link)', zorder=10)

    # Highlight fingers_link (alternative TCP)
    fingers_idx = link_names.index('fingers_link')
    fingers_pos = link_positions[fingers_idx]
    ax.scatter([fingers_pos[0]], [fingers_pos[1]], [fingers_pos[2]],
              c='orange', marker='s', s=200, label='fingers_link', zorder=10)

    # Set labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title('WidowX250s Robot Link Coordinate Frames\n(RGB = XYZ axes)', fontsize=14, fontweight='bold')

    # Set equal aspect ratio
    max_range = np.array([
        link_positions[:, 0].max() - link_positions[:, 0].min(),
        link_positions[:, 1].max() - link_positions[:, 1].min(),
        link_positions[:, 2].max() - link_positions[:, 2].min()
    ]).max() / 2.0

    mid_x = (link_positions[:, 0].max() + link_positions[:, 0].min()) * 0.5
    mid_y = (link_positions[:, 1].max() + link_positions[:, 1].min()) * 0.5
    mid_z = (link_positions[:, 2].max() + link_positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    # Add text box with coordinate frame info
    textstr = 'Coordinate Frame Colors:\n'
    textstr += 'Red = X-axis\n'
    textstr += 'Green = Y-axis\n'
    textstr += 'Blue = Z-axis\n\n'
    textstr += 'All positions in WORLD frame'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    
    plt.show()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 3D plot saved to: {save_path}")

    # Create a second plot showing just the end-effector region
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Focus on gripper links
    gripper_link_names = ['wrist_link', 'gripper_link', 'ee_arm_link', 'gripper_bar_link',
                          'fingers_link', 'left_finger_link', 'right_finger_link', 'ee_gripper_link']

    gripper_positions = []
    for name in gripper_link_names:
        idx = link_names.index(name)
        pos = link_positions[idx]
        rot = link_rotations[idx]
        gripper_positions.append(pos)

        # Plot frame
        plot_coordinate_frame(ax2, pos, rot, scale=0.025, alpha=0.9)
        # Add label
        ax2.text(pos[0], pos[1], pos[2] + 0.015, name, fontsize=9, ha='center', fontweight='bold')

    gripper_positions = np.array(gripper_positions)

    # Plot points
    ax2.scatter(gripper_positions[:, 0], gripper_positions[:, 1], gripper_positions[:, 2],
               c='black', marker='o', s=80, alpha=0.7)

    # Connect with lines
    ax2.plot(gripper_positions[:, 0], gripper_positions[:, 1], gripper_positions[:, 2],
            'k-', alpha=0.4, linewidth=2)

    # Highlight TCP and fingers_link
    tcp_idx_local = gripper_link_names.index('ee_gripper_link')
    tcp_pos = gripper_positions[tcp_idx_local]
    ax2.scatter([tcp_pos[0]], [tcp_pos[1]], [tcp_pos[2]],
               c='red', marker='*', s=400, label='TCP (ee_gripper_link)', zorder=10)

    fingers_idx_local = gripper_link_names.index('fingers_link')
    fingers_pos = gripper_positions[fingers_idx_local]
    ax2.scatter([fingers_pos[0]], [fingers_pos[1]], [fingers_pos[2]],
               c='orange', marker='s', s=300, label='fingers_link (grasp center)', zorder=10)

    ax2.set_xlabel('X (meters)', fontsize=12)
    ax2.set_ylabel('Y (meters)', fontsize=12)
    ax2.set_zlabel('Z (meters)', fontsize=12)
    ax2.set_title('WidowX250s End-Effector Coordinate Frames (Detailed View)',
                  fontsize=14, fontweight='bold')

    # Set tight bounds around gripper
    max_range = 0.08
    mid_x = gripper_positions[:, 0].mean()
    mid_y = gripper_positions[:, 1].mean()
    mid_z = gripper_positions[:, 2].mean()

    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.view_init(elev=15, azim=60)

    plt.tight_layout()
    plt.show()
    plt.savefig('robot_gripper_frames_3d.png', dpi=150, bbox_inches='tight')
    print(f"✓ Gripper detail plot saved to: robot_gripper_frames_3d.png")

    env.close()
    print("\n✓ Done! Check the generated PNG files.")


if __name__ == "__main__":
    plot_robot_links_3d()