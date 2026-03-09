"""Custom reward functions for the wheeled-leg robot."""
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def chassis_pitch_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the chassis pitch (forward/backward tilt) in L2 norm.

    Uses the Y-component of projected gravity in the robot body frame.
    When the robot is level, projected_gravity ≈ [0, 0, -g], so a non-zero
    Y component indicates a pitch deviation.
    """
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])
