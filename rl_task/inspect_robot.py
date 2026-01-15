# sample_reach.py
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from scene_cfg import Scene
import torch
import numpy as np

def sample_reachable_positions(num_samples=1000):
    """Sample random joint configurations and record EE positions."""
    sim_cfg = sim_utils.SimulationCfg(device="cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    scene_cfg = Scene(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(0.0)  # Update scene to populate data
    
    robot = scene["robot"]
    ee_frame = scene["ee_frame"]
    
    # Get joint limits - need to access correctly
    # Joint limits shape: (num_envs, num_joints, 2) where last dim is [lower, upper]
    num_joints = robot.num_joints
    print(f"Robot has {num_joints} joints")
    
    # Get default joint positions to understand the range
    default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy()
    print(f"Default joint positions: {default_joint_pos}")
    
    # Try to get joint limits - may need to access differently
    try:
        if hasattr(robot.data, 'joint_limits'):
            joint_limits = robot.data.joint_limits[0].cpu().numpy()  # (num_joints, 2)
            print(f"Joint limits shape: {joint_limits.shape}")
            print(f"Joint limits:\n{joint_limits}")
        else:
            # Fallback: use URDF limits or hardcode based on your URDF
            # From your URDF: shoulder_pan: [-1.91986, 1.91986], etc.
            joint_limits = np.array([
                [-1.91986, 1.91986],  # shoulder_pan
                [-1.74533, 1.74533],  # shoulder_lift
                [-1.69, 1.69],        # elbow_flex
                [-1.65806, 1.65806],  # wrist_flex
                [-2.74385, 2.84121],  # wrist_roll
                [-0.174533, 1.74533]  # gripper
            ])
            print("Using hardcoded joint limits from URDF")
    except Exception as e:
        print(f"Error accessing joint limits: {e}")
        # Use hardcoded limits as fallback
        joint_limits = np.array([
            [-1.91986, 1.91986],
            [-1.74533, 1.74533],
            [-1.69, 1.69],
            [-1.65806, 1.65806],
            [-2.74385, 2.84121],
            [-0.174533, 1.74533]
        ])
    
    reachable_positions = []
    
    print(f"\nSampling {num_samples} random configurations...")
    
    for i in range(num_samples):
        # Sample random joint positions within limits
        joint_pos = torch.zeros(1, num_joints, device=sim.device)
        
        for j in range(num_joints):
            lower = joint_limits[j, 0]
            upper = joint_limits[j, 1]
            # Sample uniform random value in range
            rand_val = torch.rand(1, device=sim.device).item()
            joint_pos[0, j] = rand_val * (upper - lower) + lower
        
        # Set joint positions using write_joint_state_to_sim (more direct)
        joint_vel = torch.zeros(1, num_joints, device=sim.device)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
        # Step simulation multiple times to let it settle
        for _ in range(200):  # More steps for better settling
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        
        # Get EE position
        scene.update(sim.get_physics_dt())  # Update before reading
        ee_pos = ee_frame.data.target_pos_w[0, 0, :].cpu().numpy()
        reachable_positions.append(ee_pos)
        
        if (i + 1) % 100 == 0:
            print(f"  Sampled {i+1}/{num_samples} - Current EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
    
    # Analyze results
    reachable_positions = np.array(reachable_positions)
    
    print("\n" + "=" * 80)
    print("REACHABLE WORKSPACE ANALYSIS:")
    print("=" * 80)
    print(f"X range: [{reachable_positions[:, 0].min():.3f}, {reachable_positions[:, 0].max():.3f}]")
    print(f"Y range: [{reachable_positions[:, 1].min():.3f}, {reachable_positions[:, 1].max():.3f}]")
    print(f"Z range: [{reachable_positions[:, 2].min():.3f}, {reachable_positions[:, 2].max():.3f}]")
    print("=" * 80)
    print("\nSuggested workspace bounds for object randomization:")
    print(f'  "x_range": ({reachable_positions[:, 0].min():.2f}, {reachable_positions[:, 0].max():.2f}),')
    print(f'  "y_range": ({reachable_positions[:, 1].min():.2f}, {reachable_positions[:, 1].max():.2f}),')
    print(f'  "z_range": ({reachable_positions[:, 2].min():.2f}, {reachable_positions[:, 2].max():.2f}),')
    
    simulation_app.close()
    
    return reachable_positions

if __name__ == "__main__":
    positions = sample_reachable_positions(1000)