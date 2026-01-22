from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from scene_cfg import SceneCfg
from rl_env_cfg import ObservationsCfg, ActionsCfg, RewardsCfg, TerminationsCfg, EventsCfg, CommandsCfg

@configclass
class So101EnvCfg(ManagerBasedRLEnvCfg):
    #config for the so101 environment
    
    #scene
    scene: SceneCfg = SceneCfg(num_envs=64, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()  # ADD THIS LINE
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events:  EventsCfg = EventsCfg()

    decimation: int = 2
    episode_length_s: float = 5.0

    def __post_init__(self):
        # Physics settings for better contact/grasping
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625