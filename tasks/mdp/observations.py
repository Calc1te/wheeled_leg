from __future__ import annotations

import functools
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


# ---------------------------------------------------------------------------
# Generic NaN/Inf sanitiser for observation functions
# ---------------------------------------------------------------------------

def nan_safe(func):
    """Decorator: replaces NaN with 0 and clips ±Inf in the returned tensor.

    When the physics solver becomes unstable (robot flips, penetrates ground,
    etc.), *any* observation derived from physics state can contain NaN or Inf.
    A single NaN entering the rsl_rl empirical normaliser corrupts its running
    mean/var **permanently**, which propagates to the actor's log_std and
    crashes with ``normal expects all elements of std >= 0.0``.

    Wrapping every observation function with this decorator breaks the chain at
    the source: the rollout buffer will never contain NaN, so the normaliser
    stays healthy and training can continue even if some envs briefly explode.
    """
    import functools
    import torch

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    return _wrapper


def normalize_obs(func, scale: float, clip: float = 5.0):
    """Decorator: sanitize, normalize, and clip observation tensors.

    Args:
        func: Observation function to wrap.
        scale: Divisor applied to observation values.
        clip: Symmetric clamp bound applied after normalization.
    """
    import functools
    import torch

    if scale <= 0.0:
        raise ValueError("scale must be > 0 for observation normalization")

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        result = result / scale
        return torch.clamp(result, -clip, clip)

    return _wrapper


# ---------------------------------------------------------------------------
# Custom observation functions
# ---------------------------------------------------------------------------

def height_scan_safe(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan that sanitises NaN / Inf before returning.

    ``torch.clamp`` does **not** remove NaN – ``clamp(NaN, -1, 1) == NaN``.
    When the physics solver becomes unstable (e.g. the robot flips) the ray-
    caster can produce NaN hit positions, which then poison the observation
    buffer, the empirical normaliser and ultimately the policy weights
    (manifesting as ``normal expects all elements of std >= 0.0``).

    This wrapper replaces every NaN with 0 and every ±Inf with ±1 *before*
    the value leaves the observation function, so the downstream ``clip``
    in ``ObsTerm`` only has to deal with finite numbers.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    # nan_to_num: NaN → 0.0, +Inf → posinf (default ~1e38), -Inf → neginf
    heights = torch.nan_to_num(heights, nan=0.0, posinf=1.0, neginf=-1.0)
    return heights


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if any of the sensors in the list are currently in contact."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = sensor.data.net_forces_w_history[:, 0, :]
    contact = torch.norm(net_contact_forces, dim=-1) > threshold
    return contact.float()

@nan_safe
def base_velocity_safe(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the base linear and angular velocity."""
    return torch.cat([env.scene.articulations["robot"].data.root_lin_vel_b, env.scene.articulations["robot"].data.root_ang_vel_b], dim=-1)

@nan_safe
def joint_pos_safe(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the joint positions."""
    return env.scene.articulations["robot"].data.joint_pos

@nan_safe
def joint_vel_safe(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the joint velocities."""
    return env.scene.articulations["robot"].data.joint_vel