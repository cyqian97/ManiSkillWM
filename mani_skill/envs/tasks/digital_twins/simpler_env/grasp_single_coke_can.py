"""
GraspSingleOpenedCokeCanInScene environment from SimplerEnv, ported to ManiSkill3.
Simplified and concise implementation.
"""
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

# from mani_skill import ASSET_DIR
from pathlib import Path
ASSET_DIR = Path("mani_skill/assets")
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.structs.pose import Pose

@register_env("GraspSingleOpenedCokeCanInScene-v0", max_episode_steps=100)
class GraspSingleOpenedCokeCanInScene(BaseDigitalTwinEnv):
    """Pick up an opened coke can from a table. SimplerEnv digital twin environment.

    Note: Use obs_mode="rgb+segmentation" to enable greenscreen overlay.
    The greenscreen replaces the background with a real image while keeping the robot and can visible.
    """

    SUPPORTED_OBS_MODES = ("rgb", "rgb+segmentation",)

    def __init__(self, **kwargs):
        # Greenscreen setup
        self.rgb_overlay_paths = {
            "overhead_camera": str(ASSET_DIR / "scene_datasets/simpler_env/google_coke_can_real_eval_1.png")
        }
        # RGB overlay mode: "background" (default), "debug" (50/50 blend), or "none" (disabled)
        self.rgb_overlay_mode = "background"  # Change to "debug" for 50/50 visualization

        # Object properties
        self.obj_bbox = np.array([0.066, 0.123, 0.066])  # Coke can bbox
        self.obj_density = 50  # Empty opened can

        # Episode state
        self.consecutive_grasp = 0
        self.lifted_obj = False
        self.obj_height_after_settle = None

        super().__init__(robot_uids="googlerobot_static", **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=20, spacing=20)

    @property
    def _default_sensor_configs(self):
        # No additional cameras - overhead_camera is part of the robot agent
        return []

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            pose=sapien.Pose([0.0, 0.0, 2.0], euler2quat(0, np.pi/2, np.pi)),
            width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        # Load table scene
        builder = self.scene.create_actor_builder()
        scene_offset = np.array([-1.6616, -3.0337, 0.0])
        # Rotation for Habitat scene: y-axis up to z-axis up
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])

        scene_file = str(ASSET_DIR / "scene_datasets/simpler_env/google_pick_coke_can_1_v4.glb")
        builder.add_nonconvex_collision_from_file(scene_file, pose=scene_pose)
        builder.add_visual_from_file(scene_file, pose=scene_pose)
        # Only translation applied to arena pose (rotation already in scene_pose)
        builder.initial_pose = sapien.Pose(p=-scene_offset)
        self.arena = builder.build_static(name="arena")

        # Load coke can
        model_dir = ASSET_DIR / "scene_datasets/simpler_env/opened_coke_can"
        builder = self.scene.create_actor_builder()
        material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=0.5, dynamic_friction=0.5, restitution=0.0
        )
        builder.add_multiple_convex_collisions_from_file(
            filename=str(model_dir / "collision.obj"),
            scale=[1.0]*3, material=material, density=self.obj_density
        )
        builder.add_visual_from_file(
            filename=str(model_dir / "textured.dae"), scale=[1.0]*3
        )
        self.obj = builder.build(name="opened_coke_can")

        # Exclude robot and can from greenscreen - they will be rendered from simulation
        # while the background (arena/table) will be replaced with the real-world overlay image
        self.remove_object_from_greenscreen(self.agent.robot)
        self.remove_object_from_greenscreen(self.obj)

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [0, 0, -1], [2.2, 2.2, 2.2], shadow=False, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _after_reconfigure(self, options: dict):
        super()._after_reconfigure(options)
        # Initialize episode state tensors for all environments
        self.obj_height_after_settle = torch.zeros(self.num_envs, device=self.device)
        self.consecutive_grasp = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.lifted_obj = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _load_agent(self, options: dict):
        # Robot initial pose - positioned near table height (table is at ~0.87m)
        # The Google Robot base should be at ground level, assuming the table scene includes the floor
        super()._load_agent(options, sapien.Pose(p=[0.35, 0.20, 0.0], q=[0,0,0,1]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Initialize robot pose - keep it at ground level
            # The robot base should stay where it was loaded
            # Create pose for the environments being reset
            robot_pos = torch.tensor([0.35, 0.20, 0.0], device=self.device).repeat(b, 1)
            robot_quat = torch.tensor([0., 0., 0., 1.], device=self.device).repeat(b, 1)
            self.agent.robot.set_pose(Pose.create_from_pq(robot_pos, robot_quat))

            # Drop object from above table (drawer unit)
            obj_init_xy = torch.rand((b, 2), device=self.device) * torch.tensor(
                [0.23, 0.44], device=self.device
            ) + torch.tensor([-0.35, -0.02], device=self.device)
            obj_init_z = 0.87 + 0.2  # table height + drop height

            # Random orientation
            ori_z = torch.rand(b, device=self.device) * 2 * np.pi
            quat = torch.from_numpy(
                np.array([euler2quat(0, 0, ori) for ori in ori_z.cpu().numpy()])
            ).to(self.device).float()

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, :2] = obj_init_xy
            xyz[:, 2] = obj_init_z
            self.obj.set_pose(Pose.create_from_pq(xyz, quat))

            # Settle physics
            self._settle(0.5)
            # Wake up the object to prevent sleeping by re-setting its current pose
            current_pose = self.obj.pose
            self.obj.set_pose(Pose.create_from_pq(current_pose.p[env_idx], current_pose.q[env_idx]))
            self._settle(6.0)

            # Record settled height
            self.obj_height_after_settle[env_idx] = self.obj.pose.p[env_idx, 2]

            # Reset episode state
            self.consecutive_grasp[env_idx] = 0
            self.lifted_obj[env_idx] = False

    def _settle(self, t: float):
        """Run simulation for t seconds to settle objects."""
        sim_steps = int(self.sim_freq * t / self.control_freq)
        for _ in range(sim_steps):
            self.scene.step()

    def _check_grasp(self):
        """Check if object is grasped by checking contact between fingers and object."""
        # Get finger links
        finger_links = [link for link in self.agent.robot.get_links()
                       if "finger" in link.name and "tip" in link.name]

        # Check contact forces with each finger
        has_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for finger_link in finger_links:
            contact_forces = self.scene.get_pairwise_contact_forces(finger_link, self.obj)
            net_forces = torch.linalg.norm(contact_forces, dim=1)
            has_contact = has_contact | (net_forces > 1e-6)

        return has_contact

    def evaluate(self):
        # Check grasp
        is_grasped = self._check_grasp()
        self.consecutive_grasp = torch.where(
            is_grasped, self.consecutive_grasp + 1, torch.zeros_like(self.consecutive_grasp)
        )
        self.lifted_obj = torch.where(
            ~is_grasped, torch.zeros_like(self.lifted_obj), self.lifted_obj
        )

        # Check contact with non-robot objects (lifted if only touching robot)
        contact_forces = self.scene.get_pairwise_contact_forces(self.obj, self.arena)
        net_forces = torch.linalg.norm(contact_forces, dim=1)
        no_table_contact = net_forces <= 1e-6

        # Check height
        diff_height = self.obj.pose.p[:, 2] - self.obj_height_after_settle
        lifted_significantly = no_table_contact & (diff_height > 0.01)
        self.lifted_obj = self.lifted_obj | lifted_significantly

        success = self.lifted_obj

        return {
            "is_grasped": is_grasped,
            "consecutive_grasp": self.consecutive_grasp >= 5,
            "lifted_object": self.lifted_obj,
            "success": success
        }

    def compute_dense_reward(self, obs, action, info):
        """Multi-stage dense reward following base_env.py structure."""
        # Get ee (TCP) position and orientation
        tcp_pose = self.agent.robot.links_map["link_gripper_tcp"].pose
        tcp_pos = tcp_pose.p
        tcp_quat = tcp_pose.q  # quaternion [w, x, y, z]
        w, x, y, z = tcp_quat[:, 0], tcp_quat[:, 1], tcp_quat[:, 2], tcp_quat[:, 3]

        # Compute gripper x-axis direction (for orientation reward)
        gripper_x_axis = torch.stack([
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
        ], dim=1)

        # Get object position
        pos_obj = self.obj.pose.p

        # Compute reward-related values
        tcp_to_obj_dist = torch.linalg.norm(pos_obj - tcp_pos, dim=1)

        # # Check if object has contact with table
        # contact_forces = self.scene.get_pairwise_contact_forces(self.obj, self.arena)
        # net_forces = torch.linalg.norm(contact_forces, dim=1)
        # no_table_contact = (net_forces <= 1e-6).float()  # True when no contact with table

        # Stage 1: Reaching reward - encourage TCP to reach the object
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        # Stage 1.5: Gripper orientation reward - encourage top-down grasping pose
        # For top-down grasp, the gripper's x-axis should point downward (negative z in world frame)
        target_orientation = torch.tensor([0.0, 0.0, -1.0], device=tcp_quat.device)
        orientation_alignment = (gripper_x_axis * target_orientation).sum(dim=1)
        orientation_reward = (orientation_alignment + 1) / 2
        is_not_grasped = 1.0 - info["is_grasped"].float()
        reward += orientation_reward * is_not_grasped * 0.5  # Only apply when not grasped yet

        # Stage 2: Grasping reward - encourage grasping the object
        is_grasped = info["is_grasped"]
        reward += is_grasped

        # Stage 3: Consecutive grasping reward - encourage maintaining the grasp
        is_consecutive_grasped = info["consecutive_grasp"]
        reward += is_consecutive_grasped

        # # Stage 5: Lifting reward - encourage lifting the object above the table
        # lift_threshold = 0.02  # Target lift height above settled position
        # current_lift = torch.clamp(pos_obj[:, 2] - self.obj_height_after_settle, min=0.0)
        # lifting_reward = torch.clamp(current_lift / lift_threshold, max=1.0)
        # reward += lifting_reward * is_consecutive_grasped

        # Stage 4: Success bonus - give maximum reward when task is successful
        reward[info["success"]] = 4.0

        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        """Normalize by the maximum possible reward (4.0)."""
        # Maximum reward is 4.0 from success bonus
        max_reward = 4.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def get_language_instruction(self, **kwargs):
        return ["pick opened coke can"] * self.num_envs
