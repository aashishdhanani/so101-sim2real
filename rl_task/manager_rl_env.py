from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from scene_cfg import SceneCfg
from rl_env_cfg import ObservationsCfg, ActionsCfg, RewardsCfg, TerminationsCfg, EventsCfg

@configclass
class So101EnvCfg(ManagerBasedRLEnvCfg):
    #cofig for the so101 environment
    
    #scene
    scene: SceneCfg = SceneCfg(num_envs=64, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events:  EventsCfg = EventsCfg()

    decimation: int = 4
    episode_length_s: float = 20.0
