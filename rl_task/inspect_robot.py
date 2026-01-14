"""Script to inspect the structure of scene["robot"]"""

import os
import sys
from isaaclab.app import AppLauncher

# Launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from scene_cfg import Scene

def inspect_robot():
    """Inspect and print the structure of scene["robot"]"""
    
    # Create simulation
    sim_cfg = sim_utils.SimulationCfg(device="cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Create scene with 1 environment
    scene_cfg = Scene(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()
    
    # Access the robot
    robot = scene["robot"]
    print(robot.data.joint_pos)
    
    # print("=" * 80)
    # print("ROBOT OBJECT TYPE:")
    # print("=" * 80)
    # print(f"Type: {type(robot)}")
    # print(f"Type name: {type(robot).__name__}")
    # print()
    
    # print("=" * 80)
    # print("ROBOT OBJECT ATTRIBUTES:")
    # print("=" * 80)
    # # Get all attributes (excluding private ones)
    # attrs = [attr for attr in dir(robot) if not attr.startswith('_')]
    # for attr in sorted(attrs):
    #     try:
    #         value = getattr(robot, attr)
    #         if not callable(value):
    #             print(f"  {attr}: {type(value).__name__}")
    #     except:
    #         pass
    # print()
    
    # print("=" * 80)
    # print("ROBOT.DATA ATTRIBUTES (Main Data Container):")
    # print("=" * 80)
    # if hasattr(robot, 'data'):
    #     data_attrs = [attr for attr in dir(robot.data) if not attr.startswith('_')]
    #     for attr in sorted(data_attrs):
    #         try:
    #             value = getattr(robot.data, attr)
    #             if not callable(value):
    #                 if hasattr(value, 'shape'):
    #                     print(f"  {attr}: {type(value).__name__} shape={value.shape}")
    #                 elif hasattr(value, '__len__'):
    #                     try:
    #                         print(f"  {attr}: {type(value).__name__} len={len(value)}")
    #                     except:
    #                         print(f"  {attr}: {type(value).__name__}")
    #                 else:
    #                     print(f"  {attr}: {type(value).__name__}")
    #         except Exception as e:
    #             print(f"  {attr}: <error accessing: {e}>")
    # print()
    
    # print("=" * 80)
    # print("SAMPLE DATA VALUES:")
    # print("=" * 80)
    # if hasattr(robot, 'data'):
    #     # Joint positions
    #     if hasattr(robot.data, 'joint_pos'):
    #         print(f"joint_pos shape: {robot.data.joint_pos.shape}")
    #         print(f"joint_pos sample (first env, all joints): {robot.data.joint_pos[0]}")
    #         print()
        
    #     # Joint velocities
    #     if hasattr(robot.data, 'joint_vel'):
    #         print(f"joint_vel shape: {robot.data.joint_vel.shape}")
    #         print(f"joint_vel sample (first env, all joints): {robot.data.joint_vel[0]}")
    #         print()
        
    #     # Root position
    #     if hasattr(robot.data, 'root_pos_w'):
    #         print(f"root_pos_w shape: {robot.data.root_pos_w.shape}")
    #         print(f"root_pos_w sample (first env): {robot.data.root_pos_w[0]}")
    #         print()
        
    #     # Root orientation
    #     if hasattr(robot.data, 'root_quat_w'):
    #         print(f"root_quat_w shape: {robot.data.root_quat_w.shape}")
    #         print(f"root_quat_w sample (first env): {robot.data.root_quat_w[0]}")
    #         print()
    
    # print("=" * 80)
    # print("ROBOT AS DICT (if applicable):")
    # print("=" * 80)
    # try:
    #     robot_dict = dict(robot)
    #     print(f"Can be converted to dict: Yes")
    #     print(f"Dict keys: {list(robot_dict.keys())}")
    # except:
    #     print("Cannot be converted to dict (not a dict-like object)")
    #     print("Access via: scene['robot'] -> Articulation object")
    #     print("Then access data via: scene['robot'].data.joint_pos, etc.")
    # print()
    
    # print("=" * 80)
    # print("SUMMARY:")
    # print("=" * 80)
    # print("scene['robot'] is an Articulation object")
    # print("Access data via: robot.data.<attribute_name>")
    # print("Key attributes you'll likely use:")
    # print("  - robot.data.joint_pos      (joint positions)")
    # print("  - robot.data.joint_vel      (joint velocities)")
    # print("  - robot.data.root_pos_w     (base position in world)")
    # print("  - robot.data.root_quat_w    (base orientation in world)")
    # print("=" * 80)

if __name__ == "__main__":
    inspect_robot()
    simulation_app.close()

