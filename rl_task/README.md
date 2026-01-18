# SO101 Robot RL Task

Reinforcement learning task for training a SO101 robot arm to perform reach-grasp-lift manipulation.

## Overview

This task implements a PPO-based RL environment where the SO101 robot learns to:
1. **Reach** - Move end-effector toward the object
2. **Grasp** - Close gripper at the right time and position
3. **Lift** - Raise the object above a threshold height

## Task Structure

### Environment
- **Type**: Manager-based RL environment (Isaac Lab)
- **Episode**: One attempt to pick up and lift the cube
- **Reset**: Robot joints and object position reset each episode
- **Termination**: Success (object lifted), timeout, or failure (object dropped, out of bounds)

### Action Space
- **Arm joints** (5 DOF): Joint position targets with automatic limit enforcement
- **Gripper**: Open/close control
- **Action type**: Joint position targets (absolute positions)

### Observation Space
Minimal observation set for task solvability:
1. **Joint positions and velocities** - Robot state
2. **End-effector position** - Gripper location in world frame
3. **Object position** - Target object location
4. **Gripper state** - Open/close status

### Reward Design
- **Alive**: Small positive reward for staying active
- **Terminated**: Penalty for early termination
- **Grasped**: Reward when object is close to EE and gripper is closed
- **Object height**: Reward for lifting object higher
- **Object dropped**: Penalty if object falls below end-effector

### Termination Conditions
- **Timeout**: Episode length exceeded (20 seconds)
- **Out of bounds**: Robot joints exceed limits

## File Structure

```
rl_task/
├── __init__.py              # Environment registration
├── scene_cfg.py             # Scene configuration (robot, object, table, etc.)
├── rl_env_cfg.py            # RL environment config (observations, actions, rewards, terminations, events)
├── manager_rl_env.py        # Main environment configuration class
├── train.py                 # Training script
├── agents/
│   ├── rsl_rl_ppo_cfg.py   # PPO agent configuration
│   └── README.md           # Detailed PPO explanation
└── TRAINING_EXPLANATION.md # Training structure and expectations
```

## Configuration Components

### Scene Configuration (`scene_cfg.py`)
Defines the simulation scene:
- **Robot**: SO101 arm with URDF-based configuration
- **End-effector frame**: Frame transformer sensor tracking gripper pose
- **Object**: Rigid object (cube) to be manipulated
- **Table**: Work surface for object placement
- **Ground plane**: Floor of the scene
- **Lighting**: Scene illumination

### RL Environment Configuration (`rl_env_cfg.py`)
- **Observations**: Joint states, EE position, object position
- **Actions**: Joint position control with limits
- **Rewards**: Grasping, lifting, alive bonuses
- **Terminations**: Timeout, out of bounds
- **Events**: Randomization of object position on reset

### Manager RL Environment (`manager_rl_env.py`)
Aggregates all configurations into the main environment class:
- Combines scene, observations, actions, rewards, terminations, events
- Sets decimation (4 physics steps per environment step)
- Sets episode length (20 seconds)

## Training

### Quick Start

```bash
# Train with visualization (1 environment)
python rl_task/train.py --num_envs 1

# Train headless (faster)
python rl_task/train.py --num_envs 1 --headless

# With custom WandB project
python rl_task/train.py --num_envs 1 --wandb-project "my-experiment"
```

### Training Configuration
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Max iterations**: 1500
- **Steps per env per iteration**: 24
- **Learning epochs per iteration**: 5
- **Mini-batches per epoch**: 4
- **Episode length**: 20 seconds

See `TRAINING_EXPLANATION.md` for detailed training structure.

## Environment Step Flow

### Physics Step (every simulation timestep):
1. Robot joints move
2. Objects fall/respond to physics
3. Collisions detected
4. Forces applied

### Environment Step (every 4 physics steps, decimation=4):
1. Get observations (joint pos, EE pos, object pos)
2. Policy computes action
3. Action applied to robot
4. Action held constant for next 4 physics steps
5. Compute rewards
6. Check terminations

## Key Implementation Details

### Observation Manager
- Uses `ObservationTermCfg` for each observation component
- Groups observations into "policy" group for actor/critic
- Automatically handles batching across parallel environments

### Action Manager
- Uses `JointPositionToLimitsActionCfg` for automatic joint limit enforcement
- Separate control for arm joints and gripper

### Reward Manager
- Custom reward functions for task-specific behaviors
- Weighted reward terms for balancing different objectives

### Event Manager
- Randomizes object position on reset for better generalization
- Resets robot joints to default positions

## Monitoring

Training metrics are logged to:
- **WandB**: Real-time dashboard (rewards, losses, episode lengths)
- **TensorBoard**: Synced with WandB via RSL-RL
- **Console**: Progress updates and checkpoints

Checkpoints are saved every 50 iterations to `./logs/so101_lift/`

## References

- **PPO Algorithm**: See `agents/README.md` for detailed explanation
- **Training Structure**: See `TRAINING_EXPLANATION.md` for iteration/epoch breakdown
- **Isaac Lab Docs**: [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
