'''
observations
actions
rewards
terminations
resets
'''

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionToLimitsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import mdp


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

class ActionCfg:
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

class Rewards:
    pass

class Termination:
    pass