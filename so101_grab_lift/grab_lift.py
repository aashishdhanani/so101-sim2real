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


import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg

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

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    #add assets here for like a block/object to be picked up - lets just do cube
    cuboid = AssetBaseCfg(
        prim_path="/World/Objects/Cuboid", spawn=sim_utils.CuboidCfg(size=(0.1,0.1,0.1))
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

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_so101_state = scene["so101"].data.default_root_state.clone()
            root_so101_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the jetbot's orientation and velocity
            scene["so101"].write_root_pose_to_sim(root_so101_state[:, :7])
            scene["so101"].write_root_velocity_to_sim(root_so101_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["so101"].data.default_joint_pos.clone(),
                scene["so101"].data.default_joint_vel.clone(),
            )
            scene["so101"].write_joint_state_to_sim(joint_pos, joint_vel)

            #grab and lift sequence then reset environment



            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting so101 state...")

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
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)

if __name__ == "__main__":

    main()

    simulation_app.close()


