"""
Test script to verify greenscreen overlay is working correctly.
This will create visualizations showing the greenscreen effect.

RGB Overlay Modes:
- "background": Normal greenscreen (background replaced, robot/can visible)
- "debug": 50/50 blend of simulation and overlay for visualization
- "none": No greenscreen (pure simulation render)
"""
import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Get mode from command line argument or use default
mode = sys.argv[1] if len(sys.argv) > 1 else "background"
if mode not in ["background", "debug", "none"]:
    print(f"Invalid mode '{mode}'. Use: background, debug, or none")
    sys.exit(1)

print(f"Testing greenscreen with mode: {mode}")

# Create environment with segmentation enabled (required for greenscreen)
env = gym.make(
    "GraspSingleOpenedCokeCanInScene-v0",
    obs_mode="rgb+segmentation",  # REQUIRED for greenscreen
    render_mode="rgb_array",
    num_envs=1,
)

# Override the rgb_overlay_mode (normally set in __init__)
env.unwrapped.rgb_overlay_mode = mode

# Reset and get initial observation
obs, _ = env.reset()

# Check if we have the overhead camera observation
if "overhead_camera" in obs:
    camera_obs = obs["overhead_camera"]

    # Display RGB and segmentation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RGB with greenscreen overlay
    rgb = camera_obs["rgb"][0].cpu().numpy()
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (with Greenscreen Overlay)")
    axes[0].axis("off")

    # Segmentation
    seg = camera_obs["segmentation"][0, :, :, 0].cpu().numpy()
    axes[1].imshow(seg, cmap="tab20")
    axes[1].set_title(f"Segmentation (unique IDs: {len(np.unique(seg))})")
    axes[1].axis("off")

    # Segmentation overlay on RGB
    axes[2].imshow(rgb)
    axes[2].imshow(seg, alpha=0.3, cmap="tab20")
    axes[2].set_title("RGB + Segmentation Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    output_file = f"greenscreen_test_{mode}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Greenscreen test saved to {output_file}")
    print(f"✓ Mode: {mode}")
    print(f"✓ Image shape: {rgb.shape}")
    print(f"✓ Unique segmentation IDs: {np.unique(seg)}")

    if mode == "background":
        print(f"✓ Robot and can should be visible (rendered from simulation)")
        print(f"✓ Background should show the real-world overlay image")
    elif mode == "debug":
        print(f"✓ 50/50 blend: simulation and overlay both at 50% opacity")
        print(f"✓ Useful for checking alignment between sim and real images")
    else:  # none
        print(f"✓ Pure simulation render (no greenscreen overlay)")
else:
    print("✗ Error: overhead_camera not found in observations")
    print(f"Available cameras: {list(obs.keys())}")

env.close()
