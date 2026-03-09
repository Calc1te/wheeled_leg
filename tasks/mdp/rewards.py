"""Custom reward functions for the wheeled-leg robot."""
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_symmetry_l2(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize left-right joint asymmetry in L2 norm.

    For a symmetric wheeled-leg robot, mirrored joints should have equal and
    opposite positions (left + right ≈ 0). Penalizes deviation from this.
    The left and right joint lists must have the same length and be ordered
    so that paired joints correspond element-wise.
    """
    asset = env.scene[left_cfg.name]
    left_pos = asset.data.joint_pos[:, left_cfg.joint_ids]
    right_pos = asset.data.joint_pos[:, right_cfg.joint_ids]
    return torch.sum(torch.square(left_pos + right_pos), dim=1)


def chassis_pitch_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the chassis pitch (forward/backward tilt) in L2 norm.

    Uses the Y-component of projected gravity in the robot body frame.
    When the robot is level, projected_gravity ≈ [0, 0, -g], so a non-zero
    Y component indicates a pitch deviation.
    """
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])
