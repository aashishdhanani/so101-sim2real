# SO101 Grab and Lift Simulation

This project implements a simulation environment for the SO101 robotic arm to practice deterministic grab and lift operations using Isaac Lab.

## Overview

The simulation spawns multiple parallel environments (configurable via `--num_envs`) containing:
- **SO101 robotic arm**: A 6-DOF manipulator with joints for shoulder pan/lift, elbow flex, wrist flex/roll, and gripper
- **Cuboid objects**: Small cubes positioned in front of each robot for grasping practice

## Features

- **Multi-environment support**: Spawn multiple parallel environments for training or testing
- **Keyboard control**: Manual control of all 6 joints using keyboard input
  - Joint 0 (shoulder_pan): Left/Right arrow keys
  - Joint 1 (shoulder_lift): Down/Up arrow keys
  - Joint 2 (elbow_flex): Numpad 2/5
  - Joint 3 (wrist_flex): Numpad 4/6
  - Joint 4 (wrist_roll): Numpad 7/9
  - Joint 5 (gripper): Numpad 8/Enter
- **Joint limit enforcement**: Movement automatically stops at joint limits
- **Continuous movement**: Hold keys for continuous joint motion

## Usage

```bash
python so101_grab_lift/grab_lift.py --num_envs 10
```

Make sure the simulation viewport is focused to use keyboard controls (click anywhere in the screen before using the keybinds)
