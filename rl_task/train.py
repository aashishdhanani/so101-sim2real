import argparse
from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv

parser = argparse.ArgumentParser(description="Train So101 robot with PPO")
parser.add_argument("--task", type=str, default="SO101-Lift-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--wandb-project", type=str, default="so101-lift", help="W&B project name")
parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from typing import cast
import gymnasium as gym
import wandb
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from agents.rsl_rl_ppo_cfg import So101PPORunnerCfg
from manager_rl_env import So101EnvCfg

def main():
    # Create environment config (override num_envs if specified)
    env_cfg = None
    if args_cli.num_envs is not None:
        env_cfg = So101EnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = cast(ManagerBasedRLEnv, env)
    
    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=True)
    
    # Create agent configuration
    agent_cfg = So101PPORunnerCfg()
    
    # Set up logging directory
    log_dir = "./logs/so101_lift"
    
    # Initialize wandb
    wandb.init(
        project=args_cli.wandb_project,
        name=args_cli.wandb_name or agent_cfg.experiment_name,
        config={
            "num_envs": env.num_envs,
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "max_iterations": agent_cfg.max_iterations,
            "num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
            "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            "learning_rate": agent_cfg.algorithm.learning_rate,
            "gamma": agent_cfg.algorithm.gamma,
            "lam": agent_cfg.algorithm.lam,
            "clip_param": agent_cfg.algorithm.clip_param,
            "entropy_coef": agent_cfg.algorithm.entropy_coef,
            "actor_hidden_dims": agent_cfg.policy.actor_hidden_dims,
            "critic_hidden_dims": agent_cfg.policy.critic_hidden_dims,
        },
        sync_tensorboard=True,  # Sync with tensorboard logs from RSL-RL
    )
    
    # Create PPO runner
    runner = OnPolicyRunner(
        env,
        vars(agent_cfg),
        log_dir=log_dir,
        device=str(env.device),
    )
    
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Task: {args_cli.task}")
    print(f"Number of environments: {env.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Log directory: {log_dir}")
    print(f"\nTraining Configuration:")
    print(f"  - Max iterations: {agent_cfg.max_iterations}")
    print(f"  - Steps per env per iteration: {agent_cfg.num_steps_per_env}")
    print(f"  - Learning epochs per iteration: {agent_cfg.algorithm.num_learning_epochs}")
    print(f"  - Mini-batches per epoch: {agent_cfg.algorithm.num_mini_batches}")
    print(f"  - Episode length: {env_cfg.episode_length_s if env_cfg else 20.0}s")
    print(f"\nTotal steps per iteration: {env.num_envs * agent_cfg.num_steps_per_env}")
    print(f"Total training steps: {env.num_envs * agent_cfg.num_steps_per_env * agent_cfg.max_iterations}")
    print("=" * 60)
    
    # Start training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {log_dir}")
    print("=" * 60)
    
    wandb.finish()
    
    # Close the simulator
    simulation_app.close()

if __name__ == "__main__":
    main()