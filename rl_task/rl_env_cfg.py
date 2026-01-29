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
            pos_x=(-0.1, 0.1),
            pos_y=(-0.3, -0.1),
            pos_z=(0.2, 0.35),
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
    
    gripper_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=0.5,
        use_default_offset=True, 
    )


def gripper_close_when_near(
    env,
    distance_threshold: float,
    closed_threshold: float,
):
    """Reward for closing gripper when near object. Uses gradual reward for better learning."""
    robot = env.scene["robot"]
    ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :]  # Index first frame: [num_envs, 3]
    obj_pos = env.scene["object"].data.root_pos_w

    dist = torch.norm(ee_pos - obj_pos, dim=-1)
    gripper_joint_idx = robot.joint_names.index("gripper")
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]

    # Check if near object (3D distance)
    is_near = dist < distance_threshold
    
    # Check if gripper is above or at same height as object (for better grasping)
    # Don't reward closing if gripper is below the object
    height_diff = ee_pos[:, 2] - obj_pos[:, 2]
    is_above = height_diff >= -0.01  # Allow slight tolerance (1cm below is okay)
    
    # Check horizontal distance (x, y only) - want to be close horizontally
    horizontal_dist = torch.norm((ee_pos - obj_pos)[:, :2], dim=-1)
    is_close_horizontally = horizontal_dist < 0.03  # 3cm horizontal distance
    
    # Only reward closing when: near AND (above or at height) AND close horizontally
    good_position = is_near & is_above & is_close_horizontally
    
    # Gradual reward: reward closing when in good position, with higher reward for more closed
    # Normalize gripper position: 0.5 (open) -> 0.0, -0.17 (closed) -> 1.0
    closing_progress = torch.clamp((0.5 - gripper_pos) / (0.5 - (-0.17)), 0.0, 1.0)

    if env.common_step_counter % 200 == 0:
        # Get the last gripper action (last dimension of action vector)
        gripper_action = None
        if hasattr(env, 'action_manager') and hasattr(env.action_manager, 'action'):
            # Action shape is [num_envs, action_dim], gripper is last dimension
            if env.action_manager.action is not None:
                gripper_action = env.action_manager.action[0, -1].item()  # Last dim is gripper
        
        print(
            "[DEBUG]",
            "gripper_pos:", gripper_pos[0].item(),
            "gripper_action:", gripper_action if gripper_action is not None else "N/A",
            "dist:", dist[0].item(),
            "obj_z:", obj_pos[0, 2].item(),
            "good_pos:", good_position[0].item(),
            "closing_progress:", closing_progress[0].item(),
        )
    
    # Only give reward when in good position
    return good_position.float() * closing_progress

def gripper_close_when_far(
    env,
    distance_threshold: float,
    closed_threshold: float,
):
    robot = env.scene["robot"]
    ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :]  # Index first frame: [num_envs, 3]
    obj_pos = env.scene["object"].data.root_pos_w

    dist = torch.norm(ee_pos - obj_pos, dim=-1)
    gripper_joint_idx = robot.joint_names.index("gripper")
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]

    return ((dist > distance_threshold) & (gripper_pos < closed_threshold)).float()


def touches_object(env):
    """Check if gripper is touching the object using proximity-based detection."""
    robot = env.scene["robot"]
    obj = env.scene["object"]
    ee_frame = env.scene["ee_frame"]
    
    # Get positions
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]  # shape [num_envs, 3]
    obj_pos = obj.data.root_pos_w  # shape [num_envs, 3]
    
    # Check distance between gripper and object
    distance = torch.norm(ee_pos - obj_pos, dim=-1)
    # Consider touching if very close (within 2cm)
    is_touching = distance < 0.02
    
    return is_touching.float()


def object_is_lifted(env, minimal_height: float):
    obj_z = env.scene["object"].data.root_pos_w[:, 2]
    return (obj_z > minimal_height).float()


def object_lifted_and_held(
    env,
    steps: int,
    minimal_height: float,
):
    lifted = object_is_lifted(env, minimal_height)

    if not hasattr(env, "_lift_counter"):
        env._lift_counter = torch.zeros_like(lifted)

    env._lift_counter = torch.where(
        lifted > 0,
        env._lift_counter + 1,
        torch.zeros_like(env._lift_counter),
    )

    return (env._lift_counter >= steps).float()


def reach_object(env, distance_scale: float):
    ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :]  # Index first frame: [num_envs, 3]
    obj_pos = env.scene["object"].data.root_pos_w
    dist = torch.norm(ee_pos - obj_pos, dim=-1)
    return torch.exp(-dist / distance_scale)

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reach_object = RewTerm(
        func=reach_object,
        weight=1.0,
        params={
            "distance_scale": 0.1,
        },
    )

    gripper_close_when_near = RewTerm(
        func=gripper_close_when_near,
        weight=1.5,  # Increased from 0.4 to provide stronger signal for closing
        params={
            "distance_threshold": 0.04,
            "closed_threshold": -0.05,
        },
    )

    gripper_close_when_far = RewTerm(
        func=gripper_close_when_far,
        weight=-0.3,
        params={
            "distance_threshold": 0.06,
            "closed_threshold": -0.05,
        },
    )

    contact_with_object = RewTerm(
        func=touches_object,
        weight=0.6,
    )

    lift_object = RewTerm(
        func=object_is_lifted,
        weight=6.0,
        params={
            "minimal_height": 0.06,
        },
    )

    lift_and_hold = RewTerm(
        func=object_lifted_and_held,
        weight=2.0,
        params={
            "steps": 10,
            "minimal_height": 0.06,
        },
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
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )