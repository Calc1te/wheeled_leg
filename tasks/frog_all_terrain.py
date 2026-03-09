import math
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .mdp.terminations import joint_pos_out_of_manual_limit
from .mdp.frog_actions import JointPosWheelVelActionCfg


import sys
from pathlib import Path
_PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_PATH))
from assets.frog_CFG import FROG_CONFIG as _ROBOT_CONFIG
from tasks.mdp import rewards as local_rewards
from tasks.mdp.terrain_cfg import PhantomX_ROUGH_TERRAINS_CFG

@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PhantomX_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = _ROBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/HopperTrex",
        spawn=_ROBOT_CONFIG.spawn.replace(activate_contact_sensors=True),
        init_state=_ROBOT_CONFIG.init_state.replace(
            pos=(0.0, 0.0, 1.0),
        ),
    )

    # Height scanner mounted at the robot base
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/HopperTrex/HopperTrex/chassis_base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(2.5, 2.5)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/HopperTrex/HopperTrex/.*",
        history_length=3,
        track_air_time=True,
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 20.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos_and_wheel_vel = JointPosWheelVelActionCfg(
        asset_name="robot",
        leg_joint_names=["thigh.*", "knee.*"],
        wheel_joint_names=["whee.*"],
        leg_scale=0.5,
        wheel_scale=5.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_orientation = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_orientation = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy : PolicyCfg = PolicyCfg()
    critic : CriticCfg = CriticCfg()

@configclass
class EventCfg:
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="chassis_base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.4, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.5),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Encourage wheels to maintain contact with ground
    wheel_contact = RewTerm(
        func=mdp.desired_contacts,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="whee.*"),
            "threshold": 1.0,
        },
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.60}
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    chassis_pitch_l2 = RewTerm(
        func=local_rewards.chassis_pitch_l2,
        weight=-3.0,
    )
    joint_symmetry_l2 = RewTerm(
        func=local_rewards.joint_symmetry_l2,
        weight=-0.8,
        params={
            "left_cfg": SceneEntityCfg("robot", joint_names=["thigh_left.*", "knee_left"]),
            "right_cfg": SceneEntityCfg("robot", joint_names=["thigh_right.*", "knee_right"]),
        },
    )

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.08)

    base_link_contact = RewTerm(
        func =  mdp.illegal_contact,
        params = {
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names= "chassis_base"),
            "threshold": 1.0
        },
        weight = -100
    )

    calf_link_contact = RewTerm(
        func =  mdp.illegal_contact,
        params = {
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names= "calf.*"),
            "threshold": 1.0
        },
        weight = -10
    )
    # -- optional penalties
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="thigh.*"), # find left / right_link
            "threshold": 1.0,
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

@configclass
class FrogTerrainEnvCfg(ManagerBasedRLEnvCfg):
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs = 4096, env_spacing=3.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # 120Hz / 4 = 30Hz RL control frequency
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0 / 120.0  # 120Hz physics simulation
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**16
        # Convex Decomposition 产生更多碰撞图元，需要更大的 collision stack
        self.sim.physx.gpu_collision_stack_size = 128 * 2**20  # 128 MB（默认约16MB不够）
        # sensor update periods
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt * 2  # 60 Hz


@configclass
class FrogTerrainEnvCfg_PLAY(FrogTerrainEnvCfg):
    """Play configuration: fewer envs, visualization enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 5
        self.scene.height_scanner.debug_vis = True
        self.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)
        self.curriculum.terrain_levels = None  # type: ignore[assignment]