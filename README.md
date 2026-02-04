# SO101 Sim-to-Real Manipulation Pipeline

This repository contains a reinforcement learning pipeline for training a fixed-base robotic arm (SO101) to perform manipulation tasks in simulation, with the goal of transferring learned policies to real hardware.

## Project Overview

This project implements a PPO-based RL environment where the SO101 robot learns to:
1. **Reach** - Move end-effector toward an object
2. **Grasp** - Close gripper at the right time and position
3. **Lift** - Raise the object above a threshold height
4. **Transport** - Move the lifted object to a target position

The current implementation focuses on a single-object lift task as a foundation for more complex manipulation scenarios.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support 
- **RAM**: Minimum 16GB, 32GB+ recommended
- **Storage**: At least 50GB free space for Isaac Sim installation

### Software Requirements
- **OS**: Linux (Ubuntu 20.04/22.04 recommended) or Windows
- **Python**: 3.11 (required by Isaac Sim 5.1.0)
- **CUDA**: 11.8 or 12.1 (compatible with Isaac Sim 5.1.0)
- **Isaac Sim**: 5.1.0 (will be installed via pip)
- **Isaac Lab**: 2.3.1+ (will be installed via pip)

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd so101-sim2real
```

### Step 2: Install Isaac Sim

Isaac Sim 5.1.0 is installed via pip as part of the project dependencies. However, you may need to set up the Isaac Sim environment first.

**For Linux:**
```bash
# Download and install Isaac Sim 5.1.0 from NVIDIA
# Follow instructions at: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html

# Set environment variables (add to ~/.bashrc or ~/.zshrc)
export ISAAC_SIM_PATH=/path/to/isaac-sim  # Update with your Isaac Sim installation path
```

**For Windows:**
- Download Isaac Sim 5.1.0 from NVIDIA Omniverse Launcher
- Install and note the installation path
- Set environment variables in System Properties

### Step 3: Set Up Python Environment

This project uses `uv` for dependency management. Install `uv` if you don't have it:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### Step 4: Install Project Dependencies

```bash
# Create virtual environment and install dependencies
uv sync

# This will:
# - Create a virtual environment in .venv
# - Install Isaac Lab (2.3.1+)
# - Install Isaac Sim (5.1.0)
# - Install WandB for logging
# - Install all other dependencies
```

**Note**: The first installation may take 30-60 minutes as Isaac Sim and its dependencies are large.

### Step 5: Verify Installation

```bash
# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Verify Isaac Lab installation
python -c "import isaaclab; print(isaaclab.__version__)"

# Verify Isaac Sim installation
python -c "import isaacsim; print(isaacsim.__version__)"
```

### Step 6: Set Up WandB (Optional but Recommended)

For experiment tracking and logging:

```bash
# Install WandB CLI (if not already installed)
pip install wandb

# Login to WandB
wandb login

# Follow the prompts to authenticate
```

## Project Structure

```
so101-sim2real/
├── robot/                          # Robot URDF and assets
│   ├── so101_new_calib.urdf        # Main robot URDF file
│   └── assets/                     # STL and part files for robot components
├── rl_task/                        # Main RL training code
│   ├── __init__.py                 # Environment registration
│   ├── train.py                    # Training script entry point
│   ├── manager_rl_env.py           # Main environment configuration
│   ├── scene_cfg.py                # Scene setup (robot, object, table)
│   ├── rl_env_cfg.py               # MDP configuration (obs, actions, rewards)
│   ├── cli_args.py                 # Command-line argument parsing
│   ├── agents/                     # RL algorithm configurations
│   │   └── rsl_rl_ppo_cfg.py       # PPO hyperparameters
│   ├── logs/                       # Training logs and checkpoints
│   └── wandb/                      # WandB run data
├── add_so101/                      # Utility scripts for adding robot to scenes
├── so101_grab_lift_deterministic/  # Deterministic control scripts
├── pyproject.toml                  # Project dependencies
├── uv.lock                         # Locked dependency versions
└── README.md                       # This file
```

## Running Training

### Basic Training Command

```bash
# Navigate to the rl_task directory
cd rl_task

# Activate virtual environment (if not already active)
source ../.venv/bin/activate  # Linux/Mac
# or
..\.venv\Scripts\activate  # Windows

# Run training with default settings
python train.py
```

### Training with Custom Parameters

```bash
# Specify number of environments (default: 4096)
python train.py --num_envs 2048

# Specify number of training iterations (default: 1500)
python train.py --max_iterations 2000

# Set random seed for reproducibility
python train.py --seed 42

# Specify GPU device
python train.py --device cuda:0

# Enable video recording during training
python train.py --video --video_length 200 --video_interval 2000

# Resume from checkpoint
python train.py --resume

# Resume from specific run
python train.py --resume --load_run "2025-01-29_120559-ubrx6jl9"
```

### WandB Logging

```bash
# Use WandB for logging (default)
python train.py --logger wandb --log_project_name "so101-lift"

# Use TensorBoard instead
python train.py --logger tensorboard

# Disable logging
python train.py --logger none
```

## Configuration Files

### Environment Configuration (`rl_task/manager_rl_env.py`)

Main environment settings:
- Number of parallel environments: `num_envs=4096`
- Episode length: `episode_length_s=5.0`
- Physics timestep: `dt=0.01` (100Hz)
- Decimation: `decimation=2` (policy updates every 2 physics steps)

### Scene Configuration (`rl_task/scene_cfg.py`)

Defines the simulation scene:
- Robot: SO101 arm loaded from URDF
- Object: DexCube (scaled to 0.5x)
- Table: Seattle Lab Table
- Initial joint positions and actuator settings

### RL Configuration (`rl_task/rl_env_cfg.py`)

MDP components:
- **Observations**: Joint positions/velocities, object position (robot-relative), target position, previous action
- **Actions**: Continuous joint positions for arm, binary open/close for gripper
- **Rewards**: Reaching, lifting, goal tracking (coarse + fine), smoothness penalties
- **Terminations**: Time limit, object dropping
- **Curriculum**: Gradually increases smoothness penalties over 10k steps

### PPO Configuration (`rl_task/agents/rsl_rl_ppo_cfg.py`)

PPO hyperparameters:
- Network architecture: [256, 128, 64] hidden layers
- Learning rate: 1e-4 (adaptive)
- PPO clip: 0.2
- Entropy coefficient: 0.006
- GAE: gamma=0.98, lambda=0.95
- Training: 5 epochs, 4 mini-batches per rollout

## Monitoring Training

### WandB Dashboard

If using WandB, training metrics are automatically logged:
- Episode rewards (mean, std, min, max)
- Individual reward terms
- Policy loss, value loss, entropy
- Episode lengths
- Success rates

Access your dashboard at: https://wandb.ai

### Log Files

Training logs and checkpoints are saved to:
```
rl_task/logs/rsl_rl/so101_lift/<timestamp>_<run_name>/
├── model_<iteration>.pt          # Policy checkpoints
├── progress.csv                  # Training progress
└── videos/                       # Recorded videos (if --video enabled)
```

### Checkpoint Structure

Checkpoints are saved every 50 iterations (configurable in `rsl_rl_ppo_cfg.py`):
- `model_<iteration>.pt`: Full model state (actor + critic)
- Can be loaded for evaluation or continued training

## Evaluation

### Running Evaluation

Use the `predict.py` script to evaluate a trained policy:

```bash
# Basic evaluation with default settings (10 episodes)
python predict.py --checkpoint logs/rsl_rl/so101_lift/<run_name>/model_<iteration>.pt

# Evaluate with more episodes
python predict.py --checkpoint logs/rsl_rl/so101_lift/<run_name>/model_<iteration>.pt --num_episodes 50

# Evaluate with video recording
python predict.py --checkpoint logs/rsl_rl/so101_lift/<run_name>/model_<iteration>.pt --video

# Specify number of parallel environments (default: 64 for evaluation)
python predict.py --checkpoint logs/rsl_rl/so101_lift/<run_name>/model_<iteration>.pt --num_envs 128

# Run in headless mode (no GUI)
python predict.py --checkpoint logs/rsl_rl/so101_lift/<run_name>/model_<iteration>.pt --headless

# Specify device
python predict.py --checkpoint logs/rsl_rl/so101_lift/<run_name>/model_<iteration>.pt --device cuda:0
```

### Finding Checkpoints

Checkpoints are saved in:
```
logs/rsl_rl/so101_lift/<timestamp>_<run_name>/model_<iteration>.pt
```

You can use:
- **Full path**: `--checkpoint logs/rsl_rl/so101_lift/2025-01-29_120559-ubrx6jl9/model_1500.pt`
- **Relative path**: `--checkpoint model_1500.pt` (if in the run directory)
- **Just filename**: The script will search for the checkpoint automatically

### Evaluation Output

The script prints:
- Success rate (percentage of episodes where object was lifted)
- Episode reward statistics (mean, std, min, max)
- Episode length statistics (mean, std, min, max)

Videos (if `--video` enabled) are saved to:
```
logs/eval/<timestamp>/videos/
```

## Next Steps

Future work includes:
- Vision-based observations
- Domain randomization
- More diverse grasp scenarios
- Sim-to-real transfer experiments
- Multi-object manipulation

## References

- **Isaac Lab Documentation**: https://isaac-sim.github.io/IsaacLab/
- **RSL-RL**: https://github.com/leggedrobotics/rsl_rl
- **PPO Paper**: https://arxiv.org/abs/1707.06347

## Contact

For questions or issues, please open an issue on GitHub or contact [aashishd2004@gmail.com].
