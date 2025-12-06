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

@register_env("GraspSingleOpenedCokeCanInScene-v0", max_episode_steps=80)
class GraspSingleOpenedCokeCanInScene(BaseDigitalTwinEnv):
    """Pick up an opened coke can from a table. SimplerEnv digital twin environment.

    Note: Use obs_mode="rgb+segmentation" to enable greenscreen overlay.
    The greenscreen replaces the background with a real image while keeping the robot and can visible.
    """

    SUPPORTED_OBS_MODES = ("rgb+segmentation",)

    def __init__(self, **kwargs):
        # Greenscreen setup
        self.rgb_overlay_paths = {
            "overhead_camera": str(ASSET_DIR / "scene_datasets/simpler_env/google_coke_can_real_eval_1.png")
        }
        # RGB overlay mode: "background" (default), "debug" (50/50 blend), or "none" (disabled)
        self.rgb_overlay_mode = "debug"  # Change to "debug" for 50/50 visualization

        # Object properties
        self.obj_bbox = np.array([0.066, 0.123, 0.066])  # Coke can bbox
        self.obj_density = 50  # Empty opened can

        # Episode state
        self.consecutive_grasp = 0
        self.lifted_obj = False
        self.obj_height_after_settle = None

        super().__init__(robot_uids="googlerobot", **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=513, control_freq=3, spacing=20)

    @property
    def _default_sensor_configs(self):
        # No additional cameras - overhead_camera is part of the robot agent
        return []

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            pose=sapien.Pose([5.0, 1.0, 15.0], euler2quat(0, np.pi/2, 0)),
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

    def _load_agent(self, options: dict):
        # Robot initial pose
        super()._load_agent(options, sapien.Pose(p=[0.35, 0.20, 0.06205 + 0.017], q=[0,0,0,1]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Initialize robot
            # qpos = torch.tensor([
            #     -0.264, 0.083, 0.502, 1.157, 0.029, 1.593, -1.081, 0, 0, -0.003, 0.785
            # ], device=self.device)
            # self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([0.35, 0.20, 0.07905], [0,0,0,1]))

            # Drop object from above table (drawer unit)
            obj_init_xy = torch.rand((b, 2), device=self.device) * torch.tensor(
                [0.23, 0.44], device=self.device
            ) + torch.tensor([-0.35, -0.02], device=self.device)
            obj_init_z = 0.87 + 0.5  # table height + drop height

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
            self.obj.set_pose(self.obj.pose)  # Prevent sleeping
            self._settle(0.5)

            # Check if more settling needed
            lin_vel = torch.linalg.norm(self.obj.linear_velocity, dim=1)
            ang_vel = torch.linalg.norm(self.obj.angular_velocity, dim=1)
            if (lin_vel > 1e-3).any() or (ang_vel > 1e-2).any():
                self._settle(1.5)

            # Record settled height
            self.obj_height_after_settle = self.obj.pose.p[:, 2].clone()

            # Reset episode state
            self.consecutive_grasp = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.lifted_obj = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
        """Simple sparse reward: 1 if success, 0 otherwise."""
        return info["success"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        """Normalized version of dense reward."""
        return self.compute_dense_reward(obs, action, info)

    def get_language_instruction(self, **kwargs):
        return ["pick opened coke can"] * self.num_envs
