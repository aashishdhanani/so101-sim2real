'''
observations
actions
rewards
terminations
resets
'''
from isaaclab.assets import Articulation


class Observations:
    '''
    minimal observation needed:
    1. robot joint positions + velocities
    2. EE position + object position 
    3. gripper open/close state
    '''
    def __init__(self, scene: dict[str, Articulation]):
        self.scene = scene
        self.robot = scene["robot"]
        self.ee_frame = scene["ee_frame"]
        self.object = scene["object"]

    def get_joint_data(self, robot):
        joint_positions = robot.data.joint_pos
        joint_velocities = robot.data.joint_vel
        base_position = robot.data.root_pos_w
        base_orientation = robot.data.root_quat_w

        return joint_positions, joint_velocities, base_position, base_orientation
    
    def get_gripper_open_close(self, robot):
        lower_limit = -0.174533  
        upper_limit = 1.74533
        joint_names = robot.data.joint_names
        gripper_idx = joint_names.index("gripper")

        gripper_pos = robot.data.joint_pos[gripper_idx]
        gripper_normalized = (gripper_pos - lower_limit) / (upper_limit - lower_limit)

        return gripper_normalized


    def get_ee_pos_object_pos(self, ee_frame):
        ee_pos = ee_frame.data.target_pos_w
        ee_orientation = ee_frame.data.target_quat_w   
        ee_lin_velocity = ee_frame.data.target_lin_vel_w 
        ee_ang_velocity = ee_frame.data.target_ang_vel_w

        return ee_pos, ee_orientation, ee_lin_velocity, ee_ang_velocity


class Actions:
    pass

class Rewards:
    pass

class Termination:
    pass