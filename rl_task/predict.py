import argparse
import sys
import os
import glob
import torch
import numpy as np

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="Evaluate a trained RL policy.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--task", type=str, default="SO101-Lift-v0")
parser.add_argument("--video", action="store_true", default=False, help="Record videos")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if not args_cli.checkpoint:
    parser.error("--checkpoint is required for evaluation")

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from agents.rsl_rl_ppo_cfg import So101PPORunnerCfg
from manager_rl_env import So101EnvCfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def find_checkpoint(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    log_path = os.path.join("logs", "rsl_rl", "so101_lift", path)
    if os.path.exists(log_path):
        return os.path.abspath(log_path)
    matches = glob.glob(os.path.join("logs", "rsl_rl", "so101_lift", "**", path), recursive=True)
    if matches:
        return os.path.abspath(matches[0])
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def main():
    checkpoint_path = find_checkpoint(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    env_cfg = So101EnvCfg()
    agent_cfg = So101PPORunnerCfg()
    
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    eval_log_dir = os.path.join("logs", "eval")
    os.makedirs(eval_log_dir, exist_ok=True)
    env_cfg.log_dir = eval_log_dir

    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    except (gym.error.UnregisteredEnv, ValueError):
        from gymnasium import register
        register(
            id="SO101-Lift-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"cfg": env_cfg}
        )
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # type: ignore

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(eval_log_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore

    from dataclasses import MISSING
    
    config_dict = {}
    agent_vars = vars(agent_cfg)
    for key, value in agent_vars.items():
        if value is not MISSING:
            try:
                if not isinstance(value, type(MISSING)):
                    config_dict[key] = value
            except:
                if value != MISSING:
                    config_dict[key] = value
    
    config_dict["obs_groups"] = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    
    algorithm = getattr(agent_cfg, 'algorithm', None)
    if algorithm is not None:
        config_dict["algorithm"] = vars(algorithm)
    policy = getattr(agent_cfg, 'policy', None)
    if policy is not None:
        config_dict["policy"] = vars(policy)

    runner = OnPolicyRunner(env, config_dict, log_dir=eval_log_dir, device=agent_cfg.device)  # type: ignore
    runner.load(checkpoint_path)
    print(f"[INFO] Model loaded. Starting evaluation...")

    policy_fn = runner.get_inference_policy(device=env.unwrapped.device)  # type: ignore

    episode_rewards = []
    episode_lengths = []
    success_count = 0

    obs = env.get_observations()  # type: ignore
    episode_reward = np.zeros(env_cfg.scene.num_envs)
    episode_length = np.zeros(env_cfg.scene.num_envs, dtype=int)
    episode_count = 0
    target_episodes = env_cfg.scene.num_envs * 10

    while simulation_app.is_running() and episode_count < target_episodes:
        with torch.inference_mode():
            actions = policy_fn(obs)
            obs, reward, done, info = env.step(actions)  # type: ignore
        
        episode_reward += reward.cpu().numpy()  # type: ignore
        episode_length += 1

        done_indices = np.where(done.cpu().numpy())[0] # type: ignore
        if len(done_indices) > 0:
            for idx in done_indices:
                episode_rewards.append(episode_reward[idx])
                episode_lengths.append(episode_length[idx])
                if episode_reward[idx] > 10.0:
                    success_count += 1
                episode_count += 1
            
            episode_reward[done_indices] = 0
            episode_length[done_indices] = 0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Success rate: {success_count / len(episode_rewards) * 100:.1f}% ({success_count}/{len(episode_rewards)})")
    print(f"\nRewards - Mean: {np.mean(episode_rewards):.2f}, Std: {np.std(episode_rewards):.2f}, Min: {np.min(episode_rewards):.2f}, Max: {np.max(episode_rewards):.2f}")
    print(f"Lengths - Mean: {np.mean(episode_lengths):.1f}, Std: {np.std(episode_lengths):.1f}, Min: {np.min(episode_lengths)}, Max: {np.max(episode_lengths)}")
    print("=" * 60)

    if args_cli.video:
        print(f"\n[INFO] Videos saved to: {os.path.join(eval_log_dir, 'videos')}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
