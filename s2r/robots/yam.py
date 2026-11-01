"""I2RT YAM (6-DoF arm + parallel gripper), MJCF from MuJoCo Menagerie (`i2rt_yam`)."""

from pathlib import Path

from s2r.robots.config import RobotConfig, register_robot

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "i2rt_yam"

YAM = register_robot(
  RobotConfig(
    name="yam",
    model_path=ASSETS / "yam.xml",
    arm_joints=tuple(f"joint{i}" for i in range(1, 7)),
    arm_actuators=tuple(f"joint{i}" for i in range(1, 7)),
    # Menagerie's `tcp_site` is the flange and `grasp_site` is on the flange axis, but the
    # finger pads sit 4.4 cm off that axis -> define a pinch site at the pad center.
    tcp_site="pinch_site",
    extra_sites=(("pinch_site", "link_6", (0.0, -0.044, 0.125), (1.0, 0.0, 0.0, -1.0)),),
    home_qpos=(0.0, 1.047, 1.047, 0.0, 0.0, 0.0),
    # DM4340 (joints 1-3) and DM4310 (joints 4-6) force ranges from the MJCF.
    torque_limits=(28.0, 28.0, 28.0, 10.0, 10.0, 10.0),
    kp=(40.0, 40.0, 40.0, 20.0, 10.0, 10.0),
    kd=(2.5, 2.5, 2.5, 0.5, 1.0, 1.0),
    gripper_actuator="gripper",
    gripper_open=0.041,
    gripper_closed=0.0,
    contact_excludes=(("lf_rot", "rf_rot"), ("lf_down", "rf_down"), ("lf_rot", "rf_down"), ("lf_down", "rf_rot")),
  )
)
