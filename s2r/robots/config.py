"""Robot configuration + registry.

A robot is *data*, not a class hierarchy: everything a backend, controller or
real-robot driver needs to know is in `RobotConfig`. Adding a new manipulator
(Franka, UR7e, ...) means writing one config and calling `register_robot`.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path


class ActuatorMode(str, Enum):
  """How the arm joints are actuated in simulation.

  POSITION: MuJoCo `position` actuators (implicit PD inside the integrator; ctrl = q_des).
            Real-robot equivalent: motor-side PD (e.g. DM motors in MIT mode with kp/kd set).
  TORQUE:   MuJoCo `motor` actuators (ctrl = tau). Required by OSC / impedance controllers.
            Real-robot equivalent: MIT mode with kp = kd = 0 and tau_ff = tau.
  """

  POSITION = "position"
  TORQUE = "torque"


@dataclasses.dataclass(frozen=True)
class RobotConfig:
  name: str
  model_path: Path  # MJCF or URDF (both loaded via mujoco.MjSpec.from_file)
  arm_joints: tuple[str, ...]  # controlled arm joints, in order
  arm_actuators: tuple[str, ...]  # one actuator per arm joint, same order
  tcp_site: str  # site used as the end-effector / TCP frame
  home_qpos: tuple[float, ...]  # arm joint positions at home (len == ndof)
  torque_limits: tuple[float, ...]  # |tau| limits per arm joint [Nm]
  # Default gains used when converting to POSITION mode / for JointPD defaults.
  kp: tuple[float, ...] = ()
  kd: tuple[float, ...] = ()
  # Gripper (optional): a single position actuator commanding finger opening.
  gripper_actuator: str | None = None
  gripper_open: float = 0.0  # ctrl value for fully open
  gripper_closed: float = 0.0  # ctrl value for fully closed
  actuator_mode: ActuatorMode = ActuatorMode.TORQUE
  # Body pairs whose contacts are disabled (e.g. finger-finger), saves contact budget on GPU.
  contact_excludes: tuple[tuple[str, str], ...] = ()
  # Sites added at scene-build time: (name, body, pos, quat_wxyz). Lets a config define a
  # TCP frame the source URDF/MJCF lacks (e.g. finger-pad center) without editing assets.
  extra_sites: tuple[tuple[str, str, tuple[float, float, float], tuple[float, float, float, float]], ...] = ()

  def __post_init__(self):
    n = len(self.arm_joints)
    for field in ("arm_actuators", "home_qpos", "torque_limits"):
      if len(getattr(self, field)) != n:
        raise ValueError(f"{self.name}: len({field}) != len(arm_joints) = {n}")
    for field in ("kp", "kd"):
      if getattr(self, field) and len(getattr(self, field)) != n:
        raise ValueError(f"{self.name}: len({field}) != {n}")

  @property
  def ndof(self) -> int:
    return len(self.arm_joints)

  def replace(self, **kw) -> "RobotConfig":
    return dataclasses.replace(self, **kw)


_REGISTRY: dict[str, RobotConfig] = {}


def register_robot(cfg: RobotConfig) -> RobotConfig:
  if cfg.name in _REGISTRY:
    raise KeyError(f"robot '{cfg.name}' already registered")
  _REGISTRY[cfg.name] = cfg
  return cfg


def get_robot(name: str, **overrides) -> RobotConfig:
  if name not in _REGISTRY:
    raise KeyError(f"unknown robot '{name}', registered: {sorted(_REGISTRY)}")
  cfg = _REGISTRY[name]
  return cfg.replace(**overrides) if overrides else cfg


def registered_robots() -> list[str]:
  return sorted(_REGISTRY)
