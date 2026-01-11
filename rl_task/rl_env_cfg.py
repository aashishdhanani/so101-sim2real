'''
observations
actions
rewards
terminations
resets
'''


# observation = [
#   agent_state,        # pose, velocity, joints
#   agent_dynamics,     # velocities, angular rates
#   env_state,          # objects, obstacles
#   relative_state,     # object-to-agent, goal-to-object
#   goal_state,         # target, command
# ]
class Observations:
    scene["so101"]