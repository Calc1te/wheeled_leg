import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.commands.commands_cfg import CommandTermCfg
from isaaclab.managers.command_manager import CommandTerm
from isaaclab.utils import configclass

@configclass
class BaseHeightCommandCfg(CommandTermCfg):
    """Configuration for base height commands."""
    class_type: type = "base_height_command.BaseHeightCommand"
    
    asset_name: str = "robot"
    
    @configclass
    class Ranges:
        """Ranges for the commands."""
        base_height: tuple[float, float] = (0.3, 0.6)
        
    ranges: Ranges = Ranges()

class BaseHeightCommand(CommandTerm):
    """Base height command generator."""

    cfg: BaseHeightCommandCfg
    
    def __init__(self, cfg: BaseHeightCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)
        self.height_command_b = torch.zeros(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        return "BaseHeightCommand: target height commands"

    @property
    def command(self) -> torch.Tensor:
        """The desired base height."""
        return self.height_command_b

    def _update_metrics(self):
        """Update metrics for logging."""
        pass

    def _resample_command(self, env_ids: torch.Tensor):
        """Resample the desired base height."""
        self.height_command_b[env_ids, 0] = (
            torch.rand(len(env_ids), device=self.device) * (self.cfg.ranges.base_height[1] - self.cfg.ranges.base_height[0])
            + self.cfg.ranges.base_height[0]
        )

    def _resample(self, env_ids: torch.Tensor):
        self._resample_command(env_ids)

    def _update_command(self):
        """No modifications, command remains the same."""
        pass
