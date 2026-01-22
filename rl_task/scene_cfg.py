import os

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "robot", "so101_new_calib.urdf"))
cfg = UrdfConverterCfg(
    asset_path=urdf_path,
    fix_base=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=1000.0, 
            damping=100.0      
        )
    )
)

converter = UrdfConverter(cfg=cfg)
usd_path = converter.usd_path

@configclass
class SceneCfg(InteractiveSceneCfg):

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/So101",
        init_state=ArticulationCfg.InitialStateCfg(
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": -0.0,
                "wrist_flex": 1.57, 
                "wrist_roll": -0.0,
                "gripper": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, 
                solver_position_iteration_count=8, 
                solver_velocity_iteration_count=0
            ),
        ),

        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
                effort_limit_sim=1.9,
                velocity_limit_sim=1.5,  
                stiffness={
                    "shoulder_pan": 200.0,  
                    "shoulder_lift": 170.0,
                    "elbow_flex": 120.0,    
                    "wrist_flex": 80.0,      
                    "wrist_roll": 50.0,  
                },
                damping={
                    "shoulder_pan": 80.0,
                    "shoulder_lift": 65.0,
                    "elbow_flex": 45.0,
                    "wrist_flex": 30.0,
                    "wrist_roll": 20.0,
                },
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                effort_limit_sim=2.5,  
                velocity_limit_sim=1.5,
                stiffness=60.0, 
                damping=20.0,  
            ),
        },
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/So101/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/So101/gripper_link",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.01, 0.0, -0.09),  
                ),
            ),
        ],
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0, 0.015), rot=(1, 0, 0, 0)),  
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.5, 0.5, 0.5),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0, 0, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),

    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )