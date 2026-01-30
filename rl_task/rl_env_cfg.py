'''
observations
actions
rewards
terminations
resets
'''

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import *
from isaaclab.utils.math import combine_frame_transforms
import torch


def object_position_in_robot_root_frame(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    from isaaclab.utils.math import subtract_frame_transforms
    
    robot = env.scene[robot_cfg.name]
    object_asset = env.scene[object_cfg.name]
    
    object_pos_w = object_asset.data.root_pos_w[:, :3]
    
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], 
        robot.data.root_state_w[:, 3:7],  
        object_pos_w
    )
    
    return object_pos_b

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=joint_pos_rel)
        joint_vel = ObsTerm(func=joint_vel_rel)
        object_position = ObsTerm(
            func=object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            }
        )
        target_object_position = ObsTerm(
            func=generated_commands, 
            params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class CommandsCfg:
    """Command terms for the MDP - provides target positions for curriculum learning."""
    object_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_link",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
        scale=0.5,
        use_default_offset=True, 
    )
    
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": 0.5},
        close_command_expr={"gripper": -0.17},
    )


def object_is_lifted(env, minimal_height: float):
    obj_z = env.scene["object"].data.root_pos_w[:, 2]
    return (obj_z > minimal_height).float()


def object_ee_distance(env, std: float):
    ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
    obj_pos = env.scene["object"].data.root_pos_w
    distance = torch.norm(obj_pos - ee_pos, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def gripper_closed_when_near(
    env,
    distance_threshold: float,
    closed_threshold: float,
):
    robot = env.scene["robot"]
    ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
    obj_pos = env.scene["object"].data.root_pos_w
    distance = torch.norm(obj_pos - ee_pos, dim=-1)
    gripper_joint_idx = robot.joint_names.index("gripper")
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]
    is_near = distance < distance_threshold
    is_closed = gripper_pos < closed_threshold
    return (is_near & is_closed).float()


def gripper_closed_when_far(
    env,
    distance_threshold: float,
    closed_threshold: float,
):
    robot = env.scene["robot"]
    ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
    obj_pos = env.scene["object"].data.root_pos_w
    distance = torch.norm(obj_pos - ee_pos, dim=-1)
    gripper_joint_idx = robot.joint_names.index("gripper")
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]
    is_far = distance > distance_threshold
    is_closed = gripper_pos < closed_threshold
    return (is_far & is_closed).float()


def object_goal_distance(
    env,
    std: float,
    minimal_height: float,
    command_name: str,
):
    robot = env.scene["robot"]
    obj_pos = env.scene["object"].data.root_pos_w
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - obj_pos, dim=-1)
    lifted = obj_pos[:, 2] > minimal_height
    return lifted.float() * (1.0 - torch.tanh(distance / std))

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(
        func=object_ee_distance,
        weight=1.0,
        params={
            "std": 0.1,
        },
    )

    lifting_object = RewTerm(
        func=object_is_lifted,
        weight=15.0,
        params={
            "minimal_height": 0.06,
        },
    )

    object_goal_tracking = RewTerm(
        func=object_goal_distance,
        weight=16.0,
        params={
            "std": 0.3,
            "minimal_height": 0.06,
            "command_name": "object_pose",
        },
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=object_goal_distance,
        weight=5.0,
        params={
            "std": 0.05,
            "minimal_height": 0.06,
            "command_name": "object_pose",
        },
    )

    gripper_close_near = RewTerm(
        func=gripper_closed_when_near,
        weight=2.0,
        params={
            "distance_threshold": 0.05,
            "closed_threshold": 0.1,
        },
    )

    gripper_close_far = RewTerm(
        func=gripper_closed_when_far,
        weight=-0.2,
        params={
            "distance_threshold": 0.08,
            "closed_threshold": 0.1,
        },
    )

    action_rate = RewTerm(func=action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    time_out = DoneTerm(func=time_out, time_out=True)
    
    object_dropping = DoneTerm(
        func=root_height_below_minimum, 
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

@configclass
class EventsCfg:
    """Configuration for events."""
    
    reset_all = EventTerm(func=reset_scene_to_default, mode="reset")
    
    reset_object_position = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.03, 0.03)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )
