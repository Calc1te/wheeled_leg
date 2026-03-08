import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from pathlib import Path
_PROJECT_PATH = Path(__file__).resolve().parents[1]

FROG_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_PROJECT_PATH / "assets" / "frog.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.4),
        joint_pos = {".*": 0.0},
        joint_vel = {".*": 0.0}
    ),
    actuators = {
        # joint_motor   effort 30nm
        #               saturation 97nm
        #               vel_limit 60rpm no load
        #               

        # Upper joint motors (*joint0)
        "upper_joint_motors": DCMotorCfg(
            joint_names_expr = [".*joint0"],
            effort_limit = 30.0,
            saturation_effort = 97.0,
            velocity_limit = 6.283,
            stiffness=80.0,              # Kp  [Nm/rad]
            damping=2.0,                 # Kd  [Nm·s/rad]
            friction=0.0,
        ),
        # Lower joint motors (*joint1)
        "lower_joint_motors": DCMotorCfg(
            joint_names_expr = [".*joint1"],
            effort_limit = 30.0,
            saturation_effort = 97.0,
            velocity_limit = 6.283,
            stiffness=80.0,              # Kp  [Nm/rad]
            damping=2.0,                 # Kd  [Nm·s/rad]
            friction=0.0,
        ),
        # Wheel motors (*_drive)
        "wheel_motors": DCMotorCfg(
            joint_names_expr = [".*_drive"],
            effort_limit = 2.42,
            saturation_effort = 4.5,
            velocity_limit = 71.35,
            stiffness=0.0,               # Kp=0 for velocity control
            damping=2.0,                 # Kd  [Nm·s/rad]
            friction=0.0,
        )
    }
)
# joint names: ljoint0, ljoint1, l_drive, rjoint0, rjoint1, r_drive
# body_names:   base_link, 
#               right_link1_1, right_link2_1, right_wheel_1, 
#               left_link1_1, left_link1_2, left_wheel_1 