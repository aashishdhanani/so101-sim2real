"""Script to inspect the structure of scene["robot"] and object"""

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
    """Inspect and print the structure of scene["robot"] and object"""
    
    # Create simulation
    sim_cfg = sim_utils.SimulationCfg(device="cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Create scene with 1 environment
    scene_cfg = Scene(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()
    scene.update(0.0)  # Update scene to populate data
    
    # Access the robot
    robot = scene["robot"]
    print("=" * 80)
    print("ROBOT JOINT POSITIONS:")
    print("=" * 80)
    print(f"Shape: {robot.data.joint_pos.shape}")
    print(f"Values: {robot.data.joint_pos}")
    print()
    
    # Access the object
    print("=" * 80)
    print("OBJECT ASSET INSPECTION:")
    print("=" * 80)
    object_asset = scene["object"]
    print(f"Object type: {type(object_asset)}")
    print(f"Object type name: {type(object_asset).__name__}")
    print()
    
    # Check if object has data attribute
    if hasattr(object_asset, 'data'):
        print("=" * 80)
        print("OBJECT.DATA ATTRIBUTES:")
        print("=" * 80)
        data_attrs = [attr for attr in dir(object_asset.data) if not attr.startswith('_')]
        for attr in sorted(data_attrs):
            try:
                value = getattr(object_asset.data, attr)
                if not callable(value):
                    if hasattr(value, 'shape'):
                        print(f"  {attr}: shape={value.shape}, dtype={value.dtype}")
                    elif hasattr(value, '__len__'):
                        try:
                            print(f"  {attr}: len={len(value)}")
                        except:
                            print(f"  {attr}: {type(value).__name__}")
                    else:
                        print(f"  {attr}: {type(value).__name__}")
            except Exception as e:
                print(f"  {attr}: <error accessing: {e}>")
        print()
    
    # Inspect root position (world position)
    print("=" * 80)
    print("OBJECT POSITION (root_pos_w):")
    print("=" * 80)
    if hasattr(object_asset, 'data') and hasattr(object_asset.data, 'root_pos_w'):
        root_pos = object_asset.data.root_pos_w
        print(f"Shape: {root_pos.shape}")
        print(f"Dtype: {root_pos.dtype}")
        print(f"Full tensor: {root_pos}")
        print()
        
        # For single environment
        if root_pos.shape[0] == 1:
            pos = root_pos[0]  # First (and only) environment
            print(f"Position for env 0: {pos}")
            print(f"  Index 0 (X): {pos[0].item():.6f} meters")
            print(f"  Index 1 (Y): {pos[1].item():.6f} meters")
            print(f"  Index 2 (Z/HEIGHT): {pos[2].item():.6f} meters")
            print()
            print("=" * 80)
            print("HEIGHT INFORMATION:")
            print("=" * 80)
            print(f"Height (Z-coordinate) is at index 2")
            print(f"Current height: {pos[2].item():.6f} meters")
            print(f"Expected initial height from scene_cfg: 0.055 meters")
            print(f"Difference: {pos[2].item() - 0.055:.6f} meters")
        else:
            # Multiple environments
            print(f"Multiple environments ({root_pos.shape[0]}):")
            for env_idx in range(root_pos.shape[0]):
                pos = root_pos[env_idx]
                print(f"  Env {env_idx}: X={pos[0].item():.6f}, Y={pos[1].item():.6f}, Z={pos[2].item():.6f}")
    else:
        print("ERROR: root_pos_w not found in object.data")
        print("Available attributes:", dir(object_asset.data) if hasattr(object_asset, 'data') else "No data attribute")
    
    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print("object.data.root_pos_w shape: (num_envs, 3)")
    print("  - Index 0: X coordinate (meters)")
    print("  - Index 1: Y coordinate (meters)")
    print("  - Index 2: Z coordinate / HEIGHT (meters)")
    print("=" * 80)

if __name__ == "__main__":
    inspect_robot()
    simulation_app.close()