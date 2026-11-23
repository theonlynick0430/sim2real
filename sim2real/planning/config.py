"""Planner configuration: one URDF arm, collision checking disabled. Pure data, no cuRobo import."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from .mjcf_to_urdf import UrdfInfo


@dataclasses.dataclass(frozen=True)
class PlannerConfig:
  urdf_path: str  # kinematic + inertial URDF; joint velocity limits are read from its <limit> tags
  base_link: str  # poses are expressed in this link's frame
  tool_frame: str  # link whose pose IK / planning targets
  joint_names: tuple[str, ...]  # actuated joints, in order (all other URDF joints must be fixed)
  default_q: tuple[float, ...]  # regularization / warmup configuration
  max_vel: tuple[float, ...] = ()  # mirrors the URDF velocity limits (for checking outputs only)
  max_acc: float | tuple[float, ...] = 10.0  # rad/s^2
  max_jerk: float | tuple[float, ...] = 500.0  # rad/s^3
  velocity_scale: float = 1.0  # scales the URDF velocity limits
  num_ik_seeds: int = 32
  position_tolerance: float = 0.005  # m
  orientation_tolerance: float = 0.05  # rad
  interpolation_dt: float = 0.02  # s, output trajectory sample period
  max_batch_size: int = 1  # max problems per batched call (CUDA graphs pad up to this)
  device: str = "cuda:0"

  def __post_init__(self):
    if len(self.default_q) != len(self.joint_names):
      raise ValueError("len(default_q) != len(joint_names)")
    if self.max_vel and len(self.max_vel) != len(self.joint_names):
      raise ValueError("len(max_vel) != len(joint_names)")

  @property
  def dof(self) -> int:
    return len(self.joint_names)

  def replace(self, **kw) -> "PlannerConfig":
    return dataclasses.replace(self, **kw)

  @classmethod
  def from_urdf_info(cls, info: UrdfInfo, default_q, **kw) -> "PlannerConfig":
    return cls(urdf_path=str(info.path), base_link=info.base_link, tool_frame=info.tool_link,
               joint_names=tuple(info.joint_names), default_q=tuple(float(x) for x in default_q),
               max_vel=tuple(info.velocity), **kw)

  def curobo_robot_dict(self) -> dict[str, Any]:
    """cuRobo (v2, format 2.0) robot config with no collision spheres: no self/world collision."""
    n = self.dof

    def per_joint(v):
      return [float(x) for x in v] if isinstance(v, (tuple, list)) else [float(v)] * n

    return {
      "kinematics": {
        "format_version": 2.0,
        "urdf_path": str(Path(self.urdf_path).resolve()),
        "asset_root_path": str(Path(self.urdf_path).resolve().parent),
        "base_link": self.base_link,
        "tool_frames": [self.tool_frame],
        "cspace": {
          "joint_names": list(self.joint_names),
          "default_joint_position": [float(x) for x in self.default_q],
          "null_space_weight": [1.0] * n,
          "cspace_distance_weight": [1.0] * n,
          "max_acceleration": per_joint(self.max_acc),
          "max_jerk": per_joint(self.max_jerk),
          "velocity_scale": [float(self.velocity_scale)] * n,
        },
      },
    }
