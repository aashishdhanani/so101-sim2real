# Adding the SO101 Robot to Isaac Lab

This project successfully demonstrates how to add a custom robot (the SO101 robotic arm) to an Isaac Lab simulation environment. The robot is loaded from a URDF file, converted to USD format, and then simulated with a simple wave motion demonstration.

## Overview

The script `add_so101.py` accomplishes the following:
1. **Converts URDF to USD**: Transforms the robot description from URDF format (used in ROS) to USD format (used by Isaac Sim/Omniverse)
2. **Configures the robot**: Sets up physics properties, actuators, and joint drives
3. **Creates a simulation scene**: Builds an environment with ground, lighting, and the robot
4. **Runs the simulation**: Executes a physics simulation loop that makes the robot perform a wave motion

## Key Concepts

### URDF (Unified Robot Description Format)
- **What it is**: An XML-based file format used in ROS to describe a robot's physical structure
- **Contains**: Links (rigid bodies), joints (connections between links), visual/ collision meshes (STL files), inertial properties, and joint limits
- **In this project**: The `robot/so101_new_calib.urdf` file describes the SO101 robotic arm with all its links, joints, and references to STL mesh files in `robot/assets/`

### USD (Universal Scene Description)
- **What it is**: A file format developed by Pixar and used by NVIDIA Omniverse/Isaac Sim to represent 3D scenes
- **Why convert**: Isaac Lab/Isaac Sim works natively with USD files, not URDF files directly
- **In this project**: The `UrdfConverter` automatically converts the URDF to USD format, handling mesh paths, joint configurations, and physics properties

### Isaac Lab
- **What it is**: NVIDIA's framework for robotics simulation and reinforcement learning
- **Built on**: Isaac Sim (which uses Omniverse and USD)
- **Provides**: High-level APIs for creating scenes, configuring robots, running simulations, and training RL policies

### Articulation
- **What it is**: In robotics simulation, an articulation is a connected chain of rigid bodies (links) connected by joints
- **In this project**: The SO101 robot is an articulation with multiple joints that can be controlled

## Imports Explained

```python
from isaaclab.app import AppLauncher
```
- Launches the Omniverse/Isaac Sim application and handles command-line arguments

```python
import isaaclab.sim as sim_utils
```
- Provides simulation utilities: `SimulationContext`, `SimulationCfg`, `UsdFileCfg`, `GroundPlaneCfg`, etc.

```python
from isaaclab.assets.articulation import ArticulationCfg
```
- Configuration class for defining articulated robots (robots with joints)

```python
from isaaclab.actuators import ImplicitActuatorCfg
```
- Configuration for actuators that control joints (implicit actuators use the joint drive settings from the URDF)

```python
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
```
- Converts URDF files to USD format for use in Isaac Sim

```python
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
```
- Creates and manages simulation scenes containing multiple assets (robots, ground, lights, etc.)

```python
from isaaclab.assets import AssetBaseCfg
```
- Base configuration class for any asset in the scene (ground, lights, etc.)

## Code Structure

### 1. URDF to USD Conversion (Lines 33-49)

```python
urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "robot", "so101_new_calib.urdf"))
```
- Constructs the absolute path to the URDF file relative to the script location
- This ensures the path works regardless of where the script is run from

```python
cfg = UrdfConverterCfg(
    asset_path=urdf_path,
    fix_base=True,  # Robot base is fixed (not floating)
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",      # Joints are controlled by applying forces
        target_type="position",  # Target is a position (not velocity)
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=1000.0,  # How strongly the joint resists deviation from target
            damping=100.0      # How much the joint resists velocity (damping)
        )
    )
)
```
- Configures how the URDF should be converted
- `fix_base=True`: The robot's base is fixed to the world (not floating)
- Joint drive settings: Defines how joints respond to control commands (PD controller gains)

```python
converter = UrdfConverter(cfg=cfg)
usd_path = converter.usd_path
```
- Performs the conversion and gets the path to the generated USD file

### 2. Robot Configuration (Lines 51-72)

```python
SO101_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,              # Robot is affected by gravity
            max_depenetration_velocity=5.0,    # Max speed for resolving collisions
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,      # Robot links can collide with each other
            solver_position_iteration_count=8, # Physics solver iterations
            solver_velocity_iteration_count=0
        ),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # Regex: matches all joints
            stiffness=None,  # Use values from URDF/USD
            damping=None,    # Use values from URDF/USD
        )
    },
)
```
- Defines how the robot should be spawned in the simulation
- **Rigid body properties**: Physics properties for the robot's links
- **Articulation properties**: Settings for the entire robot (collisions, solver settings)
- **Actuators**: Control all joints using the drive settings from the URDF conversion

### 3. Scene Configuration (Lines 74-90)

```python
class NewRobotsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(...)      # Ground plane
    dome_light = AssetBaseCfg(...)  # Lighting
    so101 = ArticulationCfg(...)    # The robot
```
- Defines what assets exist in the simulation scene
- `{ENV_REGEX_NS}` is a placeholder that gets replaced with environment-specific paths (e.g., `/World/envs/env_0/So101`)

### 4. Simulation Loop (Lines 92-129)

The `run_simulator` function contains the main simulation loop:

1. **Reset Logic** (every 500 steps):
   - Resets the robot to its default position and joint states
   - Adjusts position based on environment origins (for multi-environment setups)

2. **Control Logic**:
   - Creates a sine wave motion for the first 4 joints: `0.25 * sin(2π * 0.5 * time)`
   - This creates a smooth, oscillating motion with amplitude 0.25 radians and frequency 0.5 Hz

3. **Physics Step**:
   - Writes control commands to the simulator
   - Steps physics forward by one timestep
   - Updates the scene state

### 5. Main Function (Lines 131-143)

```python
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((3.5, 0.0, 3.2), (0.0, 0.0, 0.5))
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)
```
- Creates the simulation context
- Sets up the camera view (eye position and target position)
- Creates the scene with the specified number of environments
- Resets the simulation and starts the main loop

## File Structure

```
so101-sim2real/
├── robot/
│   ├── so101_new_calib.urdf      # Robot description file
│   └── assets/                    # STL mesh files referenced by URDF
│       ├── base_motor_holder_so101_v1.stl
│       ├── base_so101_v2.stl
│       └── ... (other STL files)
└── add_so101/
    ├── add_so101.py               # Main script
    └── README.md                  # This file
```

## Running the Simulation

```bash
python add_so101/add_so101.py --num_envs 1
```

The `--num_envs` argument controls how many parallel environments to spawn (useful for training RL policies, but for demonstration, 1 is sufficient).

## What Happens When You Run It

1. The Omniverse/Isaac Sim application launches
2. The URDF file is converted to USD format (happens automatically)
3. A scene is created with ground, lighting, and the SO101 robot
4. The simulation starts, and the robot's first 4 joints perform a smooth wave motion
5. Every 500 steps, the robot resets to its initial position
6. The simulation continues until you close the application

## Key Takeaways

- **URDF → USD conversion** is necessary because Isaac Lab works with USD files
- **Configuration-based approach**: Everything is defined through config classes, making it easy to modify robot properties
- **Scene-based architecture**: The scene contains all assets (robot, ground, lights) and manages their interactions
- **Physics simulation**: The simulator steps forward in time, applying physics to all objects in the scene
- **Joint control**: Joints are controlled by setting position targets, and the joint drives (with PD gains) work to achieve those targets
