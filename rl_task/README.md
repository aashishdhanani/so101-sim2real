## Designing a RL task

1. Enviornment type
Each episode = one attempt to pick up the cube
Reset the robot + cube each episode
Terminate on success, timeout, failure

2. Stages
- Reach cube, grasp cube, and lift cube

3. Action space
Joint position deltas vs end-efector deltas?

4. Observation Space
I want the smallest set of information that makes the task solvabe
- Joint positions
- Joint velocities
- End-effector pose?
- Cube post relative to end effector
- Gripper state

5. Reward Design
- Reach
- Grasp + bonus if the gripper closes and the cube is between the fingers
- Lift

6. Termination conditions
- Cube lifted above height threshold
- robot outside its joint boundaries
- Timeout
- cube falls out

7. Algorithm choice
- Been reading and seeing that PPO is a base algorithm that I can learn and use

### What is PPO (Proximal Policy Optimization)?
You have an agent that
- looks at a state s
- chooses an action a
- recieves a reward r
- repeats

Imagine teaching someone to bowl: if you change their style a tiny bit, results usually improve, but if you change a lot in one day, they might get worse 

So PPO says to try the current style, figure out which moves were better than expected, update the style to do more of those moves, but put it on a leash so it is not too drastic of changes. This leash is called clipping

Core RL math:
1. Return (How good was this situation?)
2. Value function (expected return from a state) - meaning if I'm in a state s and follow policy, what score do i expect on average
3. Q function (expected return if I take action a first)
4. Advantage (was this action better than normal?)
5. PPO is a policy gradient method which says to increase expected "goodness", push parameters in the direction that increases policy for good actions
6. PPO collects data using an old policy, but wants to update a new policy, so ration is new/old. If ratio = 1, new policy gives same probability as old policy, >1 means new policy makes that action more likely, and <1 means that new policy makes action less likely
6.1 - PPO modifies the objective to prevent this ratio value from drifting too far away from 1 so here is where clipping comes in. You have some small epsilon value and have 2 cases
6.1.1 - Advantage > 0 (action was good) -> We want to increase its probability meaning ratio > 1, but PPO says increase it but not by too much, so if ratio goes above 1 + epsilon, clipping replaces with 1 + epsilon. 
6.1.2 - Advantage < 0 (action is bad) -> decrease but not by much. So if ratio goes < 1 - epsilon, replaced with 1-epsilon

PPO is actor critic so you train 2 networks
1. Actor (policy network) - outputs the policy
2. Critic (value network) - outputs the value 

Why both? Actor decides what to do and critic estimates how good the states are so we can compute advantage

How does PPO compute advantage?
1. TD error - reward + value of future discounted - value of now
2. Generalized advantage estimation - have a bias lambda value where when = 0, there low variance and more bias and close to 1 means less bias and more variance

Algorithm step by step:
1. Initialize policy params and value params
2. Repeat for iterations
    set old policy <-- policy 
    collect rollouts: run policy in env for T steps (parallel envs too)
    compute
        - Value
        - TD error
        - advantages
        - returns R
    for K epochs
        shuffle rollout data into minibatches
        for each minibatch
            compute ratio
            compute clipped surrogate?
            compute value loss L
            compute entropy bonus
            take gradient step
3. DONE

8. Analysis and Evaluation

Configuration for a manipulation scene with a robot, object, table, and environment elements.

## Components

### `robot: ArticulationCfg`
The robot arm in the scene.
- **Type**: Articulation (multi-joint robot)
- **Purpose**: The manipulator that performs the task
- **Example**: SO101 arm, Franka Panda, etc.
- **Must be set**: Yes (defined in agent-specific config)

### `ee_frame: FrameTransformerCfg`
Sensor that tracks the end-effector (gripper) position and orientation.
- **Type**: Frame Transformer Sensor
- **Purpose**: Tracks where the gripper is relative to the robot base
- **Used for**: Computing rewards (distance to object), observations
- **Key info**: 
  - Source frame: Robot base link (e.g., `base_link`)
  - Target frame: End-effector link (e.g., `gripper_link` or `gripper_frame_link`)
  - TCP offset: Position offset from link origin to tool center point
- **Must be set**: Yes (defined in agent-specific config)

### `object: RigidObjectCfg | DeformableObjectCfg`
The object being manipulated (e.g., cube, box).
- **Type**: Rigid or Deformable Object
- **Purpose**: The item the robot picks up/manipulates
- **Used for**: 
  - Rewards (reaching, lifting, goal tracking)
  - Observations (object position)
  - Events (resetting object position)
  - Terminations (object dropping)
- **Must be set**: Yes (defined in agent-specific config)

### `table: AssetBaseCfg`
The table/work surface in the scene.
- **Type**: Static Asset
- **Purpose**: Surface where objects are placed
- **Spawn**: USD file from Isaac Nucleus (`SeattleLabTable`)
- **Position**: `(0.5, 0, 0)` with rotation `(0.707, 0, 0, 0.707)`
- **Inherited**: Yes (already defined in base class)

### `plane: AssetBaseCfg`
The ground plane.
- **Type**: Static Asset
- **Purpose**: Floor/ground of the scene
- **Spawn**: Built-in ground plane
- **Position**: `(0, 0, -1.05)` (1.05m below origin)
- **Inherited**: Yes (already defined in base class)

### `light: AssetBaseCfg`
Scene lighting.
- **Type**: Light Asset
- **Purpose**: Illuminates the scene
- **Spawn**: Dome light with color `(0.75, 0.75, 0.75)` and intensity `3000.0`
- **Inherited**: Yes (already defined in base class)
