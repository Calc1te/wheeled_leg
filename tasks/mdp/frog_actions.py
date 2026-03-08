from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTermCfg
from dataclasses import MISSING

class JointPosWheelVelAction(ActionTerm):
    """
    Action term that applies joint position commands to leg joints and velocity commands to wheel joints.
    """
    cfg: JointPosWheelVelActionCfg
    
    def __init__(self, cfg: JointPosWheelVelActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # Resolve joint indices
        self._leg_joint_ids, self._leg_joint_names = self._asset.find_joints(cfg.leg_joint_names)
        self._wheel_joint_ids, self._wheel_joint_names = self._asset.find_joints(cfg.wheel_joint_names)
        
        # Check if indices were found
        if len(self._leg_joint_ids) == 0:
            raise ValueError(f"Could not find leg joints matching: {cfg.leg_joint_names}")
        if len(self._wheel_joint_ids) == 0:
            raise ValueError(f"Could not find wheel joints matching: {cfg.wheel_joint_names}")
            
        # Total action dimension: [leg joints, wheel joints]
        self._num_actions = len(self._leg_joint_ids) + len(self._wheel_joint_ids)
        
        # Buffers for actions
        self._raw_actions = torch.zeros(self.num_envs, self._num_actions, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_actions, device=self.device)
        self._leg_actions = torch.zeros(self.num_envs, len(self._leg_joint_ids), device=self.device)
        self._wheel_actions = torch.zeros(self.num_envs, len(self._wheel_joint_ids), device=self.device)

    @property
    def action_dim(self) -> int:
        return self._num_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        # Store raw actions
        self._raw_actions[:] = actions
        
        num_legs = len(self._leg_joint_ids)
        
        # Splitting the action vector into legs (position) and wheels (velocity)
        raw_leg_actions = actions[:, :num_legs]
        raw_wheel_actions = actions[:, num_legs:]
        
        # Scale actions and apply default offset if configured
        if self.cfg.use_default_offset:
            # Retrieves default pos from asset configuration / data context
            leg_default_pos = self._asset.data.default_joint_pos[:, self._leg_joint_ids]
            self._leg_actions[:] = leg_default_pos + (raw_leg_actions * self.cfg.leg_scale)
        else:
            self._leg_actions[:] = raw_leg_actions * self.cfg.leg_scale
            
        self._wheel_actions[:] = raw_wheel_actions * self.cfg.wheel_scale
        
        # Store processed actions
        self._processed_actions[:, :num_legs] = self._leg_actions
        self._processed_actions[:, num_legs:] = self._wheel_actions
        
    def apply_actions(self):
        # Set leg joint position targets
        self._asset.set_joint_position_target(self._leg_actions, joint_ids=self._leg_joint_ids)
        
        # Set wheel joint velocity targets
        self._asset.set_joint_velocity_target(self._wheel_actions, joint_ids=self._wheel_joint_ids)

@configclass
class JointPosWheelVelActionCfg(ActionTermCfg):
    """Configuration for position and velocity action term."""
    class_type: type = JointPosWheelVelAction
    
    asset_name: str = "robot"
    """Name of the asset in the scene to apply actions to."""
    
    leg_joint_names: list[str] | str = MISSING
    """Joint names or regex for leg joints (position controlled)."""
    
    wheel_joint_names: list[str] | str = MISSING
    """Joint names or regex for wheel joints (velocity controlled)."""
    
    leg_scale: float = 1.0
    """Scale factor for leg actions."""
    
    wheel_scale: float = 1.0
    """Scale factor for wheel actions."""
    
    use_default_offset: bool = True
    """Whether to add default joint positions to the leg action commands. Defaults to True."""
    pass