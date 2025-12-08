from copy import deepcopy
from typing import List

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.link import Link


# TODO (stao) (xuanlin): Add mobile base, model it properly based on real2sim
@register_agent(asset_download_ids=["googlerobot"])
class GoogleRobot(BaseAgent):
    """
    Google Robot with mobile base (wheels can move).

    This robot has 11 controllable joints (excluding the fixed finger rotation joints):
    - 2 wheel joints (if mobile)
    - 7 arm joints
    - 2 gripper (finger) joints
    - Note: joint_finger_{left/right} are set as fixed joints in the URDF
    """
    uid = "googlerobot"
    urdf_path = (
        f"{ASSET_DIR}/robots/googlerobot/google_robot_meta_sim_fix_fingertip.urdf"
    )

    # Friction settings for fingers (based on SimplerEnv defaults)
    finger_friction = 2.0
    finger_min_patch_radius = 0.1
    finger_nail_min_patch_radius = 0.01

    urdf_config = dict(
        _materials=dict(
            finger_mat=dict(
                static_friction=finger_friction,
                dynamic_friction=finger_friction,
                restitution=0.0,
            ),
            finger_tip_mat=dict(
                static_friction=finger_friction,
                dynamic_friction=finger_friction,
                restitution=0.0,
            ),
            finger_nail_mat=dict(
                static_friction=0.1, dynamic_friction=0.1, restitution=0.0
            ),
            base_mat=dict(static_friction=0.1, dynamic_friction=0.0, restitution=0.0),
            wheel_mat=dict(static_friction=1.0, dynamic_friction=0.0, restitution=0.0),
        ),
        link=dict(
            link_base=dict(material="base_mat", patch_radius=0.1, min_patch_radius=0.1),
            link_wheel_left=dict(
                material="wheel_mat", patch_radius=0.1, min_patch_radius=0.1
            ),
            link_wheel_right=dict(
                material="wheel_mat", patch_radius=0.1, min_patch_radius=0.1
            ),
            link_finger_left=dict(
                material="finger_mat",
                patch_radius=finger_min_patch_radius,
                min_patch_radius=finger_min_patch_radius,
            ),
            link_finger_right=dict(
                material="finger_mat",
                patch_radius=finger_min_patch_radius,
                min_patch_radius=finger_min_patch_radius,
            ),
            link_finger_tip_left=dict(
                material="finger_tip_mat",
                patch_radius=finger_min_patch_radius,
                min_patch_radius=finger_min_patch_radius,
            ),
            link_finger_tip_right=dict(
                material="finger_tip_mat",
                patch_radius=finger_min_patch_radius,
                min_patch_radius=finger_min_patch_radius,
            ),
            link_finger_nail_left=dict(
                material="finger_nail_mat",
                patch_radius=finger_nail_min_patch_radius,
                min_patch_radius=finger_nail_min_patch_radius,
            ),
            link_finger_nail_right=dict(
                material="finger_nail_mat",
                patch_radius=finger_nail_min_patch_radius,
                min_patch_radius=finger_nail_min_patch_radius,
            ),
        ),
    )

    arm_joint_names = [
        "joint_torso",
        "joint_shoulder",
        "joint_bicep",
        "joint_elbow",
        "joint_forearm",
        "joint_wrist",
        "joint_gripper",
        "joint_head_pan",
        "joint_head_tilt",
    ]
    gripper_joint_names = ["joint_finger_right", "joint_finger_left"]
    ee_link_name = "link_gripper_tcp"

    # PD parameters from SimplerEnv (force drive mode)
    arm_stiffness = [
        1700.0,
        1737.0471680861954,
        979.975871856535,
        930.0,
        1212.154500274304,
        432.96500923932535,
        468.0013365498738,
        2000,
        2000,
    ]
    arm_damping = [
        1059.9791902443303,
        1010.4720585373592,
        767.2803161582076,
        680.0,
        674.9946964336588,
        274.613381336198,
        340.532560578637,
        900,
        900,
    ]
    arm_force_limit = [300, 300, 100, 100, 100, 100, 100, 100, 100]
    arm_friction = 0.0

    gripper_stiffness = 200
    gripper_damping = 8
    gripper_force_limit = 60

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-1.0,
            pos_upper=1.0,
            rot_lower=-np.pi / 2,
            rot_upper=np.pi / 2,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
            drive_mode="force",
        )
        arm_pd_ee_delta_pose_align = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-1.0,
            pos_upper=1.0,
            rot_lower=-np.pi / 2,
            rot_upper=np.pi / 2,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            frame="ee_align",
            normalize_action=False,
            drive_mode="force",
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # Mimic joint configuration for Google Robot gripper
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=-1.3 - 0.01,  # trick to have force when grasping
            upper=1.3 + 0.01,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )

        controller_configs = dict(
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose_align=dict(
                arm=arm_pd_ee_delta_pose_align, gripper=gripper_pd_joint_pos
            ),
        )

        # Make deepcopy
        return deepcopy(controller_configs)

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="overhead_camera",
                pose=sapien.Pose([0, 0, 0], [ 0.65328148,  0.27059805, -0.27059805,  0.65328148]), #[0.5, 0.5, -0.5, 0.5]
                width=640,
                height=512,
                entity_uid="link_camera",
                intrinsic=np.array([[425.0, 0, 305.0], [0, 413.1, 233.0], [0, 0, 1]]),
            )
        ]

    def _after_loading_articulation(self):
        """Called after the articulation is loaded."""
        super()._after_loading_articulation()

        # Cache finger links for grasp checking
        self.finger_right_link: Link = self.robot.links_map["link_finger_right"]
        self.finger_left_link: Link = self.robot.links_map["link_finger_left"]
        self.finger_right_tip_link: Link = self.robot.links_map["link_finger_tip_right"]
        self.finger_left_tip_link: Link = self.robot.links_map["link_finger_tip_left"]


@register_agent(asset_download_ids=["googlerobot"])
class GoogleRobotStatic(GoogleRobot):
    """
    Google Robot with static base (wheels are fixed).

    This variant uses a URDF where the wheels are fixed, making it suitable for
    tabletop manipulation tasks. The robot has 9 controllable joints:
    - 7 arm joints (torso, shoulder, bicep, elbow, forearm, wrist, gripper)
    - 2 gripper (finger) joints

    Note: head_pan and head_tilt are excluded from the arm controller because they are not
    in the kinematic chain from base to the end-effector.

    Based on SimplerEnv's GoogleRobotStaticBase implementation.
    """
    uid = "googlerobot_static"
    urdf_path = f"{ASSET_DIR}/robots/googlerobot/google_robot_static_fixed.urdf"

    # Override arm_joint_names to exclude head joints (not in kinematic chain to gripper)
    arm_joint_names = [
        "joint_torso",
        "joint_shoulder",
        "joint_bicep",
        "joint_elbow",
        "joint_forearm",
        "joint_wrist",
        "joint_gripper",
    ]

    # Override PD parameters to match the 7 arm joints (excluding head joints)
    arm_stiffness = [
        1700.0,
        1737.0471680861954,
        979.975871856535,
        930.0,
        1212.154500274304,
        432.96500923932535,
        468.0013365498738,
    ]
    arm_damping = [
        1059.9791902443303,
        1010.4720585373592,
        767.2803161582076,
        680.0,
        674.9946964336588,
        274.613381336198,
        340.532560578637,
    ]
    arm_force_limit = [300, 300, 100, 100, 100, 100, 100]

