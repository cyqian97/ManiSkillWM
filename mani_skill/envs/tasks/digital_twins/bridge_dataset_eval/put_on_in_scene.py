import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    BaseBridgeEnv,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig



@register_env(
    "MyTestEnv-v0",
    max_episode_steps=100,
    asset_download_ids=["bridge_v2_real2sim"],
)
class MyTestEnv(BaseBridgeEnv):
    SUPPORTED_OBS_MODES = ("rgb", "rgb+segmentation")
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    scene_setting = "flat_table"
    objects_excluded_from_greenscreening = [
        "bridge_carrot_generated_modified",
        "bridge_plate_objaverse_larger",
    ]

    def __init__(self, **kwargs):
        self.rgb_overlay_mode = "none"
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xyz_configs.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.887529),
                                np.append(grid_pos_2, 0.869532),
                            ]
                        )
                    )
        xyz_configs = torch.tensor(np.stack(xyz_configs))
        quat_configs = torch.tensor(
            np.stack(
                [
                    np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),
                    np.array([euler2quat(0, 0, -np.pi / 2), [1, 0, 0, 0]]),
                ]
            )
        )
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "bridge_plate_objaverse_larger"
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def evaluate(self):
        info = super()._evaluate(
            success_require_src_completely_on_target=True,
        )
        return info

    def compute_dense_reward(self, obs, action, info):
        # Get ee, source and target positions
        tcp_pos = self.agent.robot.links_map["ee_gripper_link"].pose.p
        source_obj = self.objs[self.source_obj_name]  # carrot
        target_obj = self.objs[self.target_obj_name]  # plate
        pos_src = source_obj.pose.p
        pos_tgt = target_obj.pose.p

        # Stage 1: Reaching reward - encourage TCP to reach the source object
        # Get TCP position from the ee_gripper_link
        tcp_to_obj_dist = torch.linalg.norm(
            pos_src - tcp_pos, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        # Stage 2: Grasping reward - encourage grasping the source object
        is_grasped = info["is_src_obj_grasped"]
        reward += is_grasped
        
        # Stage 3: Consecutive grasping reward - encourage maintaining the grasp
        is_consecutive_grasped = info["consecutive_grasp"]
        # reward += is_consecutive_grasped
        
        # Stage 4: Placing reward - encourage moving source object to target
        offset = pos_src - pos_tgt
        obj_to_target_dist = torch.linalg.norm(offset[:, :2], axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_target_dist)
        reward += place_reward * is_consecutive_grasped

        # Stage 5: Success bonus - give maximum reward when task is successful
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        # Normalize by the maximum possible reward (5)
        # Check the compute_dense_reward method for the value
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def get_language_instruction(self, **kwargs):
        return ["put carrot on plate"] * self.num_envs
    
    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=20, spacing=5)


@register_env(
    "PutCarrotOnPlateInScene-v1",
    max_episode_steps=60,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutCarrotOnPlateInScene(BaseBridgeEnv):
    scene_setting = "flat_table"
    objects_excluded_from_greenscreening = [
        "bridge_carrot_generated_modified",
        "bridge_plate_objaverse_larger",
    ]

    def __init__(self, **kwargs):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xyz_configs.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.887529),
                                np.append(grid_pos_2, 0.869532),
                            ]
                        )
                    )
        xyz_configs = torch.tensor(np.stack(xyz_configs))
        quat_configs = torch.tensor(
            np.stack(
                [
                    np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),
                    np.array([euler2quat(0, 0, -np.pi / 2), [1, 0, 0, 0]]),
                ]
            )
        )
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "bridge_plate_objaverse_larger"
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def evaluate(self):
        info = super()._evaluate(
            success_require_src_completely_on_target=True,
        )
        return info

    def get_language_instruction(self, **kwargs):
        return ["put carrot on plate"] * self.num_envs


@register_env(
    "PutEggplantInBasketScene-v1",
    max_episode_steps=120,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutEggplantInBasketScene(BaseBridgeEnv):
    scene_setting = "sink"
    objects_excluded_from_greenscreening = ["eggplant"]

    def __init__(self, **kwargs):
        source_obj_name = "eggplant"
        target_obj_name = "dummy_sink_target_plane"  # invisible

        target_xy = np.array([-0.125, 0.025, 1])
        xy_center = [-0.105, 0.206]

        half_span_x = 0.01
        half_span_y = 0.015
        num_x = 2
        num_y = 4

        grid_pos = []
        for x in np.linspace(-half_span_x, half_span_x, num_x):
            for y in np.linspace(-half_span_y, half_span_y, num_y):
                grid_pos.append(np.array([x + xy_center[0], y + xy_center[1], 0.933]))
        xyz_configs = [np.stack([pos, target_xy], axis=0) for pos in grid_pos]
        xyz_configs = torch.tensor(np.stack(xyz_configs))
        quat_configs = torch.tensor(
            np.stack(
                [
                    np.array([euler2quat(0, 0, 0, "sxyz"), [1, 0, 0, 0]]),
                    np.array([euler2quat(0, 0, 1 * np.pi / 4, "sxyz"), [1, 0, 0, 0]]),
                    np.array([euler2quat(0, 0, -1 * np.pi / 4, "sxyz"), [1, 0, 0, 0]]),
                ]
            )
        )
        quat_configs = torch.tensor(
            [
                [[0.543729, 0.82549, 0.0746101, 0.131747], [1, 0, 0, 0]],  # -45 deg
                [[0.559342, 0.817133, -0.138906, -0.0116353], [1, 0, 0, 0]],  # 0 deg
                [[0.543029, 0.789388, -0.267736, -0.101503], [1, 0, 0, 0]],  # 45 deg
                [[-0.40751, -0.532897, 0.652644, 0.352154], [1, 0, 0, 0]],  # 90 deg
                [[-0.157154, 0.235481, 0.75148, -0.595927], [1, 0, 0, 0]],  # 135 deg
                [[-0.020137, 0.0213848, 0.953791, -0.299033], [1, 0, 0, 0]],  # 180 deg
                [[-0.108854, 0.337692, 0.897378, -0.262348], [1, 0, 0, 0]],  # 225 deg
                [[0.366114, 0.618434, 0.571457, 0.396152], [1, 0, 0, 0]],  # 270 deg
            ]
        )
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def evaluate(self, *args, **kwargs):
        return super()._evaluate(
            success_require_src_completely_on_target=False,
            z_flag_required_offset=0.06,
            *args,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return ["put eggplant into yellow basket"] * self.num_envs

    def _load_lighting(self, options):
        self.enable_shadow

        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [0, 0, -1],
            [0.3, 0.3, 0.3],
            position=[0, 0, 1],
            shadow=False,
            shadow_scale=5,
            shadow_map_size=2048,
        )


@register_env(
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    max_episode_steps=60,
    asset_download_ids=["bridge_v2_real2sim"],
)
class StackGreenCubeOnYellowCubeBakedTexInScene(BaseBridgeEnv):
    MODEL_JSON = "info_bridge_custom_baked_tex_v0.json"
    objects_excluded_from_greenscreening = [
        "baked_green_cube_3cm",
        "baked_yellow_cube_3cm",
    ]

    def __init__(
        self,
        **kwargs,
    ):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_xs = [0.05, 0.1]
        half_edge_length_ys = [0.05, 0.1]
        xyz_configs = []

        for (half_edge_length_x, half_edge_length_y) in zip(
            half_edge_length_xs, half_edge_length_ys
        ):
            grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
            grid_pos = (
                grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
                + xy_center[None]
            )

            for i, grid_pos_1 in enumerate(grid_pos):
                for j, grid_pos_2 in enumerate(grid_pos):
                    if i != j:
                        xyz_configs.append(
                            np.array(
                                [
                                    np.append(grid_pos_1, 0.887529),
                                    np.append(grid_pos_2, 0.887529),
                                ]
                            )
                        )

        quat_configs = [np.array([[1, 0, 0, 0], [1, 0, 0, 0]])]
        quat_configs = torch.tensor(quat_configs)
        xyz_configs = torch.tensor(np.stack(xyz_configs))
        source_obj_name = "baked_green_cube_3cm"
        target_obj_name = "baked_yellow_cube_3cm"
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def evaluate(self):
        info = super()._evaluate(
            success_require_src_completely_on_target=True,
        )
        return info

    def get_language_instruction(self, **kwargs):
        return ["stack the green block on the yellow block"] * self.num_envs


@register_env(
    "PutSpoonOnTableClothInScene-v1",
    max_episode_steps=60,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutSpoonOnTableClothInScene(BaseBridgeEnv):
    objects_excluded_from_greenscreening = [
        "table_cloth_generated_shorter",
        "bridge_spoon_generated_modified",
    ]

    def __init__(
        self,
        **kwargs,
    ):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xyz_configs.append(
                        np.array(
                            [np.append(grid_pos_1, 0.88), np.append(grid_pos_2, 0.875)]
                        )
                    )

        quat_configs = [
            np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
        ]
        quat_configs = torch.tensor(np.stack(quat_configs))
        xyz_configs = torch.tensor(np.stack(xyz_configs))
        source_obj_name = "bridge_spoon_generated_modified"
        target_obj_name = "table_cloth_generated_shorter"
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def evaluate(self):
        # this environment allows spoons to be partially on the table cloth to be considered successful
        return super()._evaluate(success_require_src_completely_on_target=False)

    def get_language_instruction(self, **kwargs):
        return ["put the spoon on the towel"] * self.num_envs
    
    
@register_env(
    "PutSpoonOnTableClothInSceneReward-v0",
    max_episode_steps=125,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutSpoonOnTableClothInSceneReward(PutSpoonOnTableClothInScene):
    SUPPORTED_OBS_MODES = ("rgb", "rgb+segmentation")
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    def compute_dense_reward(self, obs, action, info):
        # Get ee, source and target positions
        tcp_pose = self.agent.robot.links_map["ee_gripper_link"].pose
        tcp_pos = tcp_pose.p
        tcp_quat = tcp_pose.q  # quaternion [w, x, y, z]

        source_obj = self.objs[self.source_obj_name]  # spoon
        target_obj = self.objs[self.target_obj_name]  # tablecloth
        pos_src = source_obj.pose.p
        pos_tgt = target_obj.pose.p

        # Stage 1: Reaching reward - encourage TCP to reach the source object
        # Get TCP position from the ee_gripper_link
        tcp_to_obj_dist = torch.linalg.norm(
            pos_src - tcp_pos, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        # Stage 1.5: Gripper orientation reward - encourage top-down grasping pose
        # For top-down grasp, the gripper's z-axis should point downward (negative z in world frame)
        # Extract the z-axis of the gripper from the rotation matrix
        # The z-axis is the third column of the rotation matrix
        # For quaternion q = [w, x, y, z], the z-axis of the rotated frame is:
        # z_axis = [2(xz + wy), 2(yz - wx), 1 - 2(x^2 + y^2)]
        w, x, y, z = tcp_quat[:, 0], tcp_quat[:, 1], tcp_quat[:, 2], tcp_quat[:, 3]
        gripper_x_axis = torch.stack([ 1 - 2 * ( y * y + z*z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
           
        ], dim=1)

        # For top-down grasp, we want the gripper's x-axis to align with world's negative z-axis [0, 0, -1]
        target_orientation = torch.tensor([0.0, 0.0, -1.0], device=tcp_quat.device)
        # Compute dot product: 1 means perfectly aligned, -1 means opposite
        orientation_alignment = (gripper_x_axis * target_orientation).sum(dim=1)
        # Map from [-1, 1] to [0, 1], where 1 is perfect alignment with downward
        orientation_reward = (orientation_alignment + 1) / 2

        # Only apply this reward when not grasped yet (to encourage approach from top)
        is_not_grasped = 1.0 - info["is_src_obj_grasped"].float()
        reward += orientation_reward * is_not_grasped * 0.5  # Scale down to 0.5 max

        # Stage 2: Grasping reward - encourage grasping the source object
        is_grasped = info["is_src_obj_grasped"]
        reward += is_grasped
        
        # Stage 3: Consecutive grasping reward - encourage maintaining the grasp
        is_consecutive_grasped = info["consecutive_grasp"]
        reward += is_consecutive_grasped

        # Stage 3.5: Lifting reward - encourage lifting the object above the table
        # Get the initial z position of the table surface (assumed around 0.88 based on xyz_configs)
        table_height = 0.88
        lift_threshold = 0.20  # Target lift height above table
        current_lift = torch.clamp(pos_src[:, 2] - table_height, min=0.0)
        lifting_reward = torch.clamp(current_lift / lift_threshold, max=1.0)
        reward += lifting_reward * is_consecutive_grasped

        # Stage 4: Placing reward - encourage moving source object to target
        # Only apply this reward when the source object has no contact with the table
        contact_forces = self.scene.get_pairwise_contact_forces(source_obj, self.arena)
        net_forces = torch.linalg.norm(contact_forces, dim=1)
        no_table_contact = (net_forces <= 0.001).float()  # True when no contact with table
        
        offset = pos_src - pos_tgt
        obj_to_target_dist = torch.linalg.norm(offset[:, :2], axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_target_dist)
        reward += place_reward * is_consecutive_grasped * no_table_contact

        # Stage 5: Success bonus - give maximum reward when task is successful
        reward[info["success"]] = 7.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        # Normalize by the maximum possible reward (7)
        # Check the compute_dense_reward method for the value
        max_reward = 7.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=20, spacing=5)