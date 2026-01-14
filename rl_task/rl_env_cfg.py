'''
observations
actions
rewards
terminations
resets
'''
from isaaclab.assets import Articulation
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg



class ObservationsCfg:
    policy = ObsGroup(
        joint_positions = ObsTerm(
            func = joint_pos,
            params = {
                "config" : SceneEntityCfg(
                    name = "robot",
                    joint_names = ".*"
                )
            }
        ),

        joint_velocities = ObsTerm(
            func = joint_vel,
            params = {
                "config" : SceneEntityCfg(
                    name = "robot",
                    joint_names = ".*"
                )
            }
        ),

        gripper = ObsTerm(
            func = gripper_move,
            params = {
                "config" : SceneEntityCfg(
                    name = "robot",
                    joint_names = "gripper"
                ),
                "lower" = -0.174533,
                "upper" = 1.74533,
            }
        ),

        ee_pos=ObservationTermCfg(
            func=ee_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="ee_frame"
                )
            }
        ),
    )

class Actions:
    def __init__(self, scene, action_type):
        self.scene = scene
        self.action_type = action_type
        self.robot = scene["robot"]

class Rewards:
    pass

class Termination:
    pass