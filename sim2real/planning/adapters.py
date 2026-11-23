"""sim2real -> planner glue. The ONLY module in `planning/` that imports from the rest of sim2real;
drop this file to use the package elsewhere (build a `PlannerConfig` from any URDF instead)."""

from __future__ import annotations

from pathlib import Path

import mujoco

from sim2real.robots.config import RobotConfig
from sim2real.sim.scene import robot_spec

from .config import PlannerConfig
from .mjcf_to_urdf import UrdfInfo, export_urdf


def export_robot_urdf(robot: RobotConfig, workdir: str | Path, velocity=2.0) -> UrdfInfo:
  """Export `robot`'s arm (root body -> `tcp_site`, with config `extra_sites` applied) to
  `<workdir>/<name>.urdf`. velocity: URDF joint velocity limit(s) [rad/s] used by the planner."""
  m = robot_spec(robot).compile()
  body = int(m.site_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, robot.tcp_site)])
  while int(m.body_parentid[body]) != 0:  # robot root = the world child above the TCP body
    body = int(m.body_parentid[body])
  return export_urdf(m, m.body(body).name, robot.tcp_site, robot.arm_joints, Path(workdir) / f"{robot.name}.urdf",
                     effort=robot.torque_limits, velocity=velocity, robot_name=robot.name)


def planner_config_from_robot(robot: RobotConfig, workdir: str | Path, velocity=2.0, **overrides) -> PlannerConfig:
  """PlannerConfig for `robot` (un-prefixed `RobotConfig`; poses in its root-body frame, which is the
  `RobotMount` frame in multi-arm scenes). `overrides` set any other PlannerConfig field."""
  info = export_robot_urdf(robot, workdir, velocity)
  return PlannerConfig.from_urdf_info(info, default_q=robot.home_qpos, **overrides)
