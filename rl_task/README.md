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
