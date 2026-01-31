import gymnasium as gym

from . import agents

gym.register(
    id="SO101-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.manager_rl_env:So101EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:So101PPORunnerCfg",
    },
    disable_env_checker=True,
)

