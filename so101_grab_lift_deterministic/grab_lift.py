import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import os
import time


import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.devices import Se3KeyboardCfg, Se3Keyboard, Se3GamepadCfg, Se3Gamepad

# Convert URDF to USD
urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "robot", "so101_new_calib.urdf"))
cfg = UrdfConverterCfg(
    asset_path=urdf_path,
    fix_base=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=1000.0, 
            damping=100.0      
        )
    )
)

converter = UrdfConverter(cfg=cfg)
usd_path = converter.usd_path

# Define the robot configuration for Isaac Lab
#mess here to figure out how to place the robot in a certain postion, relative to block?
SO101_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0
        ),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"], 
            stiffness=None,  
            damping=None,  
        )
    },
)

class GrabLiftSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    #add assets here for like a block/object to be picked up - lets just do cube
    cuboid = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cuboid",  # Use ENV_REGEX_NS placeholder
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.3, 0.0, 0.05),  # Position relative to each environment origin
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # robot - set prim_path directly with the ENV_REGEX_NS placeholder
    so101 = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/So101",  # This gets replaced with /World/envs/env_0/So101, etc.
        spawn=SO101_CONFIG.spawn,
        actuators=SO101_CONFIG.actuators,
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    keyboard = Se3Keyboard(Se3KeyboardCfg("cpu"))
    joint_targets = scene["so101"].data.default_joint_pos.clone()
    joint_step = 0.01

    # Track key state: True = key is being held, False = released
    key_states = {}
    key_timeout = 0.15  # If no press event for this long, consider key released
    
    # Key mappings
    key_mappings = [
        ("LEFT", 0, -1), ("RIGHT", 0, 1),
        ("DOWN", 1, -1), ("UP", 1, 1),
        ("NUMPAD_2", 2, -1), ("NUMPAD_5", 2, 1),
        ("NUMPAD_4", 3, -1), ("NUMPAD_6", 3, 1),
        ("NUMPAD_7", 4, -1), ("NUMPAD_9", 4, 1),
        ("NUMPAD_8", 5, -1), ("NUMPAD_ENTER", 5, 1),
    ]
    
    # Set up callbacks to track key presses
    for key_name, joint_idx, direction in key_mappings:
        key_id = f"joint_{joint_idx}_{'inc' if direction > 0 else 'dec'}"
        keyboard.add_callback(key_name, lambda k=key_id: key_states.update({k: time.time()}))
        key_states[key_id] = 0  # Initialize to 0 (not pressed)
    
    # Get joint limits from the robot
    joint_limits = scene["so101"].data.joint_limits  # Shape: [num_envs, num_joints, 2] where [:, :, 0] is lower, [:, :, 1] is upper
    
    print("[INFO]: Hold keys for continuous movement. Movement stops at joint limits.")

    while simulation_app.is_running():
        keyboard.advance()
        
        current_time = time.time()
        
        # Apply continuous movement for keys that are being held
        for joint_idx in range(6):
            inc_key = f"joint_{joint_idx}_inc"
            dec_key = f"joint_{joint_idx}_dec"
            
            # Check if key is being held (pressed recently)
            inc_held = (current_time - key_states.get(inc_key, 0)) < key_timeout
            dec_held = (current_time - key_states.get(dec_key, 0)) < key_timeout
            
            # Get current joint limits for this joint
            lower_limit = joint_limits[0, joint_idx, 0].item()
            upper_limit = joint_limits[0, joint_idx, 1].item()
            current_pos = joint_targets[0, joint_idx].item()
            
            # Apply movement if key is held and within limits
            if inc_held and current_pos < upper_limit:
                new_pos = current_pos + joint_step
                if new_pos <= upper_limit:  # Check before applying
                    joint_targets[:, joint_idx] += joint_step
                    
            if dec_held and current_pos > lower_limit:
                new_pos = current_pos - joint_step
                if new_pos >= lower_limit:  # Check before applying
                    joint_targets[:, joint_idx] -= joint_step
        
        scene["so101"].set_joint_position_target(joint_targets)
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((3.5, 0.0, 3.2), (0.0, 0.0, 0.5))
    # Design scene
    scene_cfg = GrabLiftSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()

    num_joints = scene["so101"].data.default_joint_pos.shape[1]
    print(f"[INFO]: Number of joints: {num_joints}")

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)

if __name__ == "__main__":

    main()

    simulation_app.close()


