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

def ee_to_object_relative(env, ee_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    ee_frame = env.scene[ee_cfg.name]
    object_asset = env.scene[object_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = object_asset.data.root_pos_w
    
    relative_pos = ee_pos - obj_pos
    return relative_pos

def ee_to_object_relative_normalized(env, ee_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, 
                                     scale: float = 0.3):
    ee_frame = env.scene[ee_cfg.name]
    object_asset = env.scene[object_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = object_asset.data.root_pos_w
    
    relative_pos = ee_pos - obj_pos
    normalized = relative_pos / scale
    return torch.clamp(normalized, -1.0, 1.0)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(
            func=object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            }
        )
        target_object_position = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class CommandsCfg:
    """Command terms for the MDP - provides target positions for curriculum learning."""
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_link",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
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
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
        scale=0.5,
        use_default_offset=True,  # KEY: Centers actions around default joint positions
    )
    
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": 0.5},
        close_command_expr={"gripper": 0.0},
    )


def is_grasped(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, 
               ee_cfg: SceneEntityCfg, distance_threshold: float = 0.03,
               gripper_closed_threshold: float = 0.8,
               height_threshold: float = 0.04,
               velocity_match_threshold: float = 0.1):
    robot = env.scene[robot_cfg.name]
    object_asset = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = object_asset.data.root_pos_w
    obj_vel = object_asset.data.root_lin_vel_w
    
    gripper_link_idx = robot.body_names.index("gripper_link")
    ee_vel = robot.data.body_lin_vel_w[:, gripper_link_idx, :]
    
    distance = torch.norm(ee_pos - obj_pos, dim=-1)
    
    gripper_joint_idx = robot.joint_names.index("gripper")
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]
    
    obj_height = obj_pos[:, 2]
    obj_vel_norm = torch.norm(obj_vel, dim=-1)
    ee_vel_norm = torch.norm(ee_vel, dim=-1)
    vel_diff = torch.abs(obj_vel_norm - ee_vel_norm)
    
    close_enough = distance < distance_threshold
    gripper_closed = gripper_pos > gripper_closed_threshold
    is_lifted = obj_height > height_threshold
    velocity_matches = vel_diff < velocity_match_threshold
    
    is_grasped = close_enough & gripper_closed & is_lifted & velocity_matches
    
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

def relative_distance_reward(env, ee_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    ee_frame = env.scene[ee_cfg.name]
    object_asset = env.scene[object_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = object_asset.data.root_pos_w
    
    relative_pos = ee_pos - obj_pos
    distance = torch.norm(relative_pos, dim=-1)
    
    return -distance

def joint_movement_penalty(env, robot_cfg: SceneEntityCfg, base_joint_weight: float = 2.0):
    robot = env.scene[robot_cfg.name]
    joint_vel = robot.data.joint_vel
    
    base_joint_idx = robot.joint_names.index("shoulder_pan")
    base_vel = torch.abs(joint_vel[:, base_joint_idx])
    
    other_indices = [i for i in range(robot.num_joints) if i != base_joint_idx]
    other_joint_vel = torch.abs(joint_vel[:, other_indices])
    other_vel_norm = torch.norm(other_joint_vel, dim=-1)
    
    penalty = base_vel * base_joint_weight + other_vel_norm
    return -penalty

def action_rate_penalty(env):
    return mdp.action_rate_l2(env)

def base_joint_position_penalty(env, robot_cfg: SceneEntityCfg, max_deviation: float = 1.0):
    robot = env.scene[robot_cfg.name]
    base_joint_idx = robot.joint_names.index("shoulder_pan")
    base_joint_pos = robot.data.joint_pos[:, base_joint_idx]
    
    deviation = torch.abs(base_joint_pos)
    normalized_deviation = torch.clamp(deviation / max_deviation, 0.0, 1.0)
    return -normalized_deviation ** 2



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

def gripper_close_when_near(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, 
                             ee_cfg: SceneEntityCfg, distance_threshold: float = 0.08,
                             gripper_closed_threshold: float = 0.5):
    robot = env.scene[robot_cfg.name]
    object_asset = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = object_asset.data.root_pos_w
    distance = torch.norm(ee_pos - obj_pos, dim=-1)
    
    gripper_joint_idx = robot.joint_names.index("gripper")
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]
    
    close_to_object = distance < distance_threshold
    gripper_closing = gripper_pos > gripper_closed_threshold
    
    reward = (close_to_object & gripper_closing).float()
    return reward

def object_out_of_range(env, object_cfg: SceneEntityCfg, 
                        max_distance: float = 0.5):
    object_asset = env.scene[object_cfg.name]
    obj_pos = object_asset.data.root_pos_w
    
    distance = torch.norm(obj_pos, dim=-1)
    out_of_range = distance > max_distance
    
    return out_of_range.bool()

def vertical_approach_reward(env, ee_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, 
                             height_above: float = 0.05):
    ee_frame = env.scene[ee_cfg.name]
    object_asset = env.scene[object_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = object_asset.data.root_pos_w
    
    height_diff = ee_pos[:, 2] - obj_pos[:, 2]
    horizontal_dist = torch.norm((ee_pos - obj_pos)[:, :2], dim=-1)
    
    is_above = (height_diff > 0) & (height_diff < height_above) & (horizontal_dist < 0.05)
    return is_above.float()

def horizontal_swipe_penalty(env, ee_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    ee_frame = env.scene[ee_cfg.name]
    object_asset = env.scene[object_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = object_asset.data.root_pos_w
    
    height_diff = ee_pos[:, 2] - obj_pos[:, 2]
    horizontal_dist = torch.norm((ee_pos - obj_pos)[:, :2], dim=-1)
    
    too_low = height_diff < -0.02
    too_far_horizontal = horizontal_dist > 0.08
    
    bad_approach = too_low | too_far_horizontal
    return -bad_approach.float()


def object_ee_distance(env, std: float, object_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg):
    """Reward for reaching object using tanh-kernel (smoother gradients)."""
    from isaaclab.assets import RigidObject
    from isaaclab.sensors import FrameTransformer
    
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    
    return 1 - torch.tanh(object_ee_distance / std)

def object_is_lifted(env, minimal_height: float, object_cfg: SceneEntityCfg):
    """Simple binary reward for lifting object above threshold."""
    from isaaclab.assets import RigidObject
    
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_goal_distance(env, std: float, minimal_height: float, command_name: str,
                        robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    """Reward for tracking goal pose using tanh-kernel."""
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import combine_frame_transforms
    
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], 
        robot.data.root_state_w[:, 3:7], 
        des_pos_b
    )
    
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    reaching_object = RewTerm(
        func=object_ee_distance, 
        params={"std": 0.05, "object_cfg": SceneEntityCfg("object"), "ee_frame_cfg": SceneEntityCfg("ee_frame")}, 
        weight=1.0
    )
    
    lifting_object = RewTerm(
        func=object_is_lifted, 
        params={"minimal_height": 0.025, "object_cfg": SceneEntityCfg("object")}, 
        weight=25.0
    )
    
    object_goal_tracking = RewTerm(
        func=object_goal_distance,
        params={
            "std": 0.3, 
            "minimal_height": 0.025, 
            "command_name": "object_pose",
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object")
        },
        weight=30.0,
    )
    
    object_goal_tracking_fine_grained = RewTerm(
        func=object_goal_distance,
        params={
            "std": 0.05, 
            "minimal_height": 0.025, 
            "command_name": "object_pose",
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object")
        },
        weight=5.0,
    )
    
    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

@configclass
class EventsCfg:
    """Configuration for events."""
    
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )