# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import importlib.metadata as metadata
import os
import time
import torch
from packaging import version

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

# Optional in some IsaacLab versions.
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    get_published_pretrained_checkpoint = None

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import sys
from pathlib import Path
_PROJECT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_PATH))
import tasks


# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    raise RuntimeError(
        f"Unsupported rsl-rl-lib version '{installed_version}'. Expected >= {RSL_RL_VERSION}."
    )


def _migrate_legacy_rsl_rl_checkpoint(loaded_dict: dict) -> tuple[dict, bool]:
    """Convert legacy rsl-rl checkpoint schema to the current actor/critic schema."""
    if "actor_state_dict" in loaded_dict and "critic_state_dict" in loaded_dict:
        return loaded_dict, False

    if "model_state_dict" not in loaded_dict:
        return loaded_dict, False

    model_state = loaded_dict["model_state_dict"]
    if not isinstance(model_state, dict):
        return loaded_dict, False

    actor_state = {}
    critic_state = {}

    for key, value in model_state.items():
        if key.startswith("actor."):
            actor_state[f"mlp.{key[len('actor.') :]}"] = value
        elif key.startswith("actor_obs_normalizer."):
            actor_state[f"obs_normalizer.{key[len('actor_obs_normalizer.') :]}"] = value
        elif key == "log_std":
            actor_state["distribution.std_param"] = value
        elif key.startswith("critic."):
            critic_state[f"mlp.{key[len('critic.') :]}"] = value
        elif key.startswith("critic_obs_normalizer."):
            critic_state[f"obs_normalizer.{key[len('critic_obs_normalizer.') :]}"] = value

    migrated = {
        "actor_state_dict": actor_state,
        "critic_state_dict": critic_state,
        "optimizer_state_dict": loaded_dict.get("optimizer_state_dict"),
        "iter": loaded_dict.get("iter", loaded_dict.get("iteration", 0)),
        "infos": loaded_dict.get("infos"),
    }
    return migrated, True


def _filter_incompatible_state_dict(source_state: dict, target_state: dict, state_name: str) -> dict:
    """Drop missing or shape-incompatible tensors before loading a state dict."""
    filtered_state = {}
    dropped = []
    for key, value in source_state.items():
        if key not in target_state:
            dropped.append((key, "missing_key"))
            continue
        if hasattr(value, "shape") and hasattr(target_state[key], "shape") and value.shape != target_state[key].shape:
            dropped.append((key, f"shape_mismatch {tuple(value.shape)} != {tuple(target_state[key].shape)}"))
            continue
        filtered_state[key] = value

    if dropped:
        print(f"[WARNING]: Dropped {len(dropped)} incompatible {state_name} tensor(s) during checkpoint load.")
    return filtered_state


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Adapt deprecated config fields to the installed rsl-rl API version.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            print(
                "[INFO] Pretrained checkpoint lookup is not available in this IsaacLab version. "
                "Install a newer IsaacLab build or pass --checkpoint <path>."
            )
            return
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    loaded_dict = torch.load(resume_path, weights_only=False, map_location=agent_cfg.device)
    loaded_dict, migrated = _migrate_legacy_rsl_rl_checkpoint(loaded_dict)

    if migrated:
        print("[WARNING]: Migrated legacy checkpoint format to current actor/critic schema.")

    if "actor_state_dict" in loaded_dict and "critic_state_dict" in loaded_dict:
        loaded_dict["actor_state_dict"] = _filter_incompatible_state_dict(
            loaded_dict["actor_state_dict"], runner.alg.actor.state_dict(), "actor"
        )
        loaded_dict["critic_state_dict"] = _filter_incompatible_state_dict(
            loaded_dict["critic_state_dict"], runner.alg.critic.state_dict(), "critic"
        )

    load_cfg = None if not migrated else {"actor": True, "critic": True, "optimizer": False, "iteration": False}
    runner.alg.load(loaded_dict, load_cfg=load_cfg, strict=False)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if hasattr(runner, "export_policy_to_jit") and hasattr(runner, "export_policy_to_onnx"):
        # rsl-rl >= 5.x export path
        runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")
    else:
        # legacy fallback for older policy wrappers
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
