import gymnasium as gym
from . import agents
from agents import FrogFlatPPORunnerCfg, FrogTerrainPPORunnerCfg

gym.register(
    id="frog-flat-v0",
    ntry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.frog_flat_terrain:FrogFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.frog_rsl_rl_ppo:FrogFlatPPORunnerCfg",
    },
)