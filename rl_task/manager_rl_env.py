from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from scene_cfg import SceneCfg
from rl_env_cfg import ObservationsCfg, ActionsCfg, RewardsCfg, TerminationsCfg, EventsCfg, CommandsCfg, CurriculumCfg
from isaaclab.markers.config import FRAME_MARKER_CFG


@configclass
class So101EnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg() 
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    events:  EventsCfg = EventsCfg()

    def __post_init__(self):
        """Post initialization."""
        marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore[attr-defined]
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame.visualizer_cfg = marker_cfg


        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
