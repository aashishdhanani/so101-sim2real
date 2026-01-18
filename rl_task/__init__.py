from gymnasium import register
from isaaclab.envs import ManagerBasedRLEnv
from manager_rl_env import So101EnvCfg

register(
    id="SO101-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg" : So101EnvCfg()}
)