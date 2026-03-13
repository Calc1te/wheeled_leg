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


def terrain_level_forward_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_forward_speed: float = 0.2,
    target_forward_speed: float = 1.0,
) -> torch.Tensor:
    """Reward harder terrain traversal, but only when the robot actually moves forward.

    The terrain level is normalized to [0, 1] using ``max_terrain_level`` and multiplied by
    a smooth forward-speed gate. This prevents a policy from exploiting the reward by simply
    idling on harder terrain tiles.
    """
    terrain = env.scene.terrain
    asset = env.scene[asset_cfg.name]

    if not hasattr(terrain, "terrain_levels"):
        return torch.zeros(env.num_envs, device=env.device)

    terrain_level = terrain.terrain_levels.float()
    max_level = max(int(getattr(terrain, "max_terrain_level", 1)) - 1, 1)
    level_norm = terrain_level / float(max_level)

    forward_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0)
    # 0 below threshold, then smoothly ramps to 1 around target speed.
    speed_gate = torch.clamp((forward_speed - min_forward_speed) / max(target_forward_speed - min_forward_speed, 1e-6), 0.0, 1.0)

    return level_norm * speed_gate
