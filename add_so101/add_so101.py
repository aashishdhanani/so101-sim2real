import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="This script demonstrates the SO101 robot scene with table and cube."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

urdf_path = Path(__file__).resolve().parent.parent / "robot" / "so101_new_calib.urdf"

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
        spawn=sim_utils.UrdfFileCfg(
            fix_base=True,
            replace_cylinders_with_capsules=True,
            asset_path=str(urdf_path),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
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
        soft_joint_pos_limit_factor=0.9,
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.015), rot=(1, 0, 0, 0)),
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

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((3.5, 0.0, 3.2), (0.0, 0.0, 0.5))
    
    scene_cfg = SceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    print("[INFO]: Setup complete...")
    print("[INFO]: Scene contains robot, table, and cube - all stationary")
    
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
