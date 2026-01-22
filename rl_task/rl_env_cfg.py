'''
observations
actions
rewards
terminations
resets
'''

from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionToLimitsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import mdp
import torch


def ee_pos(env, asset_cfg: SceneEntityCfg):
    # Get the frame transformer from the scene
    frame_transformer = env.scene[asset_cfg.name]
    # Get the target frame position (world frame)
    ee_pos_w = frame_transformer.data.target_pos_w[:, 0, :]
    return ee_pos_w


def object_pose(env, asset_cfg: SceneEntityCfg):
    # Gets rigid body
    rigid_object = env.scene[asset_cfg.name]
    # Get the root position in world frame
    object_pos_w = rigid_object.data.root_pos_w
    return object_pos_w

def ee_to_object_relative(env, ee_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    ee_frame = env.scene[ee_cfg.name]
    object_asset = env.scene[object_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = object_asset.data.root_pos_w
    
    relative_pos = ee_pos - obj_pos
    return relative_pos

@configclass
class ObservationsCfg:
    class PolicyCfg(ObsGroup):
        joint_positions = ObsTerm(
            func = mdp.joint_pos_rel,
            params = {
                "asset_cfg" : SceneEntityCfg(
                    name = "robot",
                    joint_names = ".*"
                )
            }
        )

        joint_velocities = ObsTerm(
            func = mdp.joint_vel_rel,
            params = {
                "asset_cfg" : SceneEntityCfg(
                    name = "robot",
                    joint_names = ".*"
                )
            }
        )

        gripper = ObsTerm(
            func = mdp.joint_pos_rel,
            params = {
                "asset_cfg" : SceneEntityCfg(
                    name = "robot",
                    joint_names = "gripper"
                )
            }
        )

        ee_pos = ObsTerm(
            func= ee_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="ee_frame"
                )
            }
        )

        object_position = ObsTerm(
            func = object_pose,
            params = {
                "asset_cfg": SceneEntityCfg(
                    name = "object"
                )
            }
        )

        ee_to_object = ObsTerm(  # NEW
            func=ee_to_object_relative,
            params={
                "ee_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object"),
            }
        )
    policy = PolicyCfg()

@configclass
class ActionsCfg:
    arm_joints = JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan",
            "shoulder_lift", 
            "elbow_flex",
            "wrist_flex",
            "wrist_roll"
        ],
    )

    gripper = JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
    )


def is_grasped(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, 
               ee_cfg: SceneEntityCfg, distance_threshold: float = 0.05,
               gripper_closed_threshold: float = 0.8):
    robot = env.scene[robot_cfg.name]
    object_asset = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_cfg.name]
    
    # Get positions
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = object_asset.data.root_pos_w  
    
    # Compute distance between EE and object
    distance = torch.norm(ee_pos - obj_pos, dim=-1)  
    
    # Get gripper joint position by finding index from joint names
    gripper_joint_idx = robot.joint_names.index("gripper")
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx] 
    
    # Check conditions
    close_enough = distance < distance_threshold
    gripper_closed = gripper_pos > gripper_closed_threshold
    
    # Object is grasped if both conditions are met
    is_grasped = close_enough & gripper_closed
    
    return is_grasped.float()

def height(env, object_cfg: SceneEntityCfg):
    object = env.scene[object_cfg.name]
    obj_pos = object.data.root_pos_w

    height = obj_pos[:, 2]

    return height

def dropped(env, object_cfg: SceneEntityCfg, ee_cfg: SceneEntityCfg,
            height_margin: float = 0.02):
    object_asset = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]  
    ee_height = ee_pos[:, 2] 
    
    obj_pos = object_asset.data.root_pos_w  
    obj_height = obj_pos[:, 2] 
    
    dropped = obj_height < (ee_height - height_margin)
    
    return dropped.float()

def distance_to_object(env, ee_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, scale: float = 0.1):
    ee_frame = env.scene[ee_cfg.name]
    object_asset = env.scene[object_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = object_asset.data.root_pos_w
    
    distance = torch.norm(ee_pos - obj_pos, dim=-1)
    return torch.exp(-distance / scale)

def joint_movement_penalty(env, robot_cfg: SceneEntityCfg, base_joint_weight: float = 2.0):
    robot = env.scene[robot_cfg.name]
    joint_vel = robot.data.joint_vel
    
    base_joint_idx = robot.joint_names.index("shoulder_pan")
    base_vel = torch.abs(joint_vel[:, base_joint_idx])
    
    other_joint_vel = torch.abs(joint_vel[:, [i for i in range(robot.num_joints) if i != base_joint_idx]])
    other_vel_norm = torch.norm(other_joint_vel, dim=-1)
    
    penalty = base_vel * base_joint_weight + other_vel_norm
    return -penalty



def self_collision_penalty(env, robot_cfg: SceneEntityCfg, ee_cfg: SceneEntityCfg, min_distance: float = 0.15):
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_cfg.name]
    
    base_pos = robot.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    
    distance_to_base = torch.norm(ee_pos - base_pos, dim=-1)
    too_close = distance_to_base < min_distance
    
    return -too_close.float()

def touches_object(env, ee_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, 
                   touch_threshold: float = 0.03):
    ee_frame = env.scene[ee_cfg.name]
    object_asset = env.scene[object_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = object_asset.data.root_pos_w
    
    distance = torch.norm(ee_pos - obj_pos, dim=-1)
    is_touching = distance < touch_threshold
    
    return is_touching.float()

def object_out_of_range(env, object_cfg: SceneEntityCfg, 
                        max_distance: float = 0.5):
    object_asset = env.scene[object_cfg.name]
    obj_pos = object_asset.data.root_pos_w
    
    distance = torch.norm(obj_pos, dim=-1)
    out_of_range = distance > max_distance
    
    return out_of_range.bool()

#how does weights affect? change them?
@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.5)  # Reduced from 1.0
    terminated = RewTerm(func=mdp.is_terminated, weight=-2.0)
    
    approach_object = RewTerm(
        func=distance_to_object,  # Use exponential version
        weight=5.0,  # Increased from 2.0
        params={
            "ee_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("object"),
            "scale": 0.1,
        }
    )
    
    joint_movement = RewTerm(
        func=joint_movement_penalty,
        weight=-0.1,  # Small penalty for movement
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "base_joint_weight": 2.0,  # Penalize base more
        }
    )
    
    self_collision = RewTerm(
        func=self_collision_penalty,
        weight=-2.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_cfg": SceneEntityCfg("ee_frame"),
            "min_distance": 0.15,
        }
    )
    
    touches_object = RewTerm(
        func=touches_object,
        weight=5.0,  # Increased from 3.0
        params={
            "ee_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("object"),
            "touch_threshold": 0.03,
        }
    )
    
    grasped = RewTerm(
        func=is_grasped, 
        weight=15.0,  # Increased from 10.0
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"]), 
            "object_cfg": SceneEntityCfg("object"), 
            "ee_cfg": SceneEntityCfg("ee_frame"),
            "distance_threshold": 0.05,
            "gripper_closed_threshold": 0.8
        }
    )
    
    object_height = RewTerm(
        func=height, 
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
        }
    )
    
    object_dropped = RewTerm(
        func=dropped, 
        weight=-5.0,
        params={
            "object_cfg": SceneEntityCfg("object"), 
            "ee_cfg": SceneEntityCfg("ee_frame")
        }
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    object_out_of_range = DoneTerm(
        func=object_out_of_range,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "max_distance": 0.5,
        }
    )

@configclass
class EventsCfg:
    # Reset robot joints
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"]),
            "position_range" : (0.0,0.0),
            "velocity_range" : (0.0,0.0)
        }
    )
    
    # Reset object with randomized position
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="object"),
            "pose_range": {
                "x": (0.15, 0.30),   # Forward reach (conservative, 80% of max ~0.39m)
                "y": (-0.15, 0.15),  # Side-to-side
                "z": (0.02, 0.05)    # Height above table
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0)
            }
        }
    )