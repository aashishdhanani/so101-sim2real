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
        ),

        joint_velocities = ObsTerm(
            func = mdp.joint_vel_rel,
            params = {
                "asset_cfg" : SceneEntityCfg(
                    name = "robot",
                    joint_names = ".*"
                )
            }
        ),

        gripper = ObsTerm(
            func = mdp.joint_pos_rel,
            params = {
                "asset_cfg" : SceneEntityCfg(
                    name = "robot",
                    joint_names = "gripper"
                ),
                "lower" : -0.174533,
                "upper" : 1.74533,
            }
        ),

        ee_pos = ObsTerm(
            func= ee_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="ee_frame"
                )
            }
        ),

        object_position = ObsTerm(
            func = object_pose,
            params = {
                "asset_cfg": SceneEntityCfg(
                    name = "object"
                )
            }
        ),
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
    
    # Get heights (z-coordinates)
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]  
    ee_height = ee_pos[:, 2] 
    
    obj_pos = object_asset.data.root_pos_w  
    obj_height = obj_pos[:, 2] 
    
    # Object is dropped if it's below the EE (with some margin)
    dropped = obj_height < (ee_height - height_margin)
    
    return dropped.float()

#how does weights affect? change them?
@configclass
class RewardsCfg:
    #1 Alive
    alive = RewTerm(func=mdp.is_alive, weight = 1.0)
    terminated = RewTerm(func=mdp.is_terminated, weight = -2.0)
    #is_grasped
    grasped = RewTerm(
        func=is_grasped, 
        weight=5.0,
        params = {
            "robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"]), 
            "object_cfg" : SceneEntityCfg("object"), 
            "ee_cfg" : SceneEntityCfg("ee_frame"),
            "distance_threshold": 0.05,
            "gripper_closed_threshold": 0.8
        }    
        
    )
    #object_height
    object_height = RewTerm(
        func=height, 
        weight=5.0,
        params= {
            "object_cfg" : SceneEntityCfg("object"),
        }    
    )
    #object_dropped
    object_dropped = RewTerm(
        func=dropped, 
        weight=-5.0,
        params = {
            "object_cfg" : SceneEntityCfg("object"), 
            "ee_cfg" : SceneEntityCfg("ee_frame")
        } 
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_limit,
        params = {"asset_cfg" : SceneEntityCfg("robot", joint_names=[".*"])}
    )

@configclass
class EventsCfg:
    # Reset robot joints
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])
        }
    )
    
    # Reset object with randomized position
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="object"),
            "pose_range": {
                "x": (-0.2, 0.2), 
                "y": (-0.2, 0.2),  
                "z": (-0.005, 0.045)  
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0)
            }
        }
    )