"""Unified simulation backend interface.

Both backends expose the same minimal surface:

  reset()                       -> all worlds to the scene's home keyframe
  step1() / step2()             -> split physics step (MuJoCo mj_step1/mj_step2 semantics);
                                   controllers run in between on *current* M, J, bias
  robot_state(needs)            -> RobotState (warp arrays, backend device)
  set_arm_ctrl(cmd)             -> write (nworld, n) command into the arm actuators
  set_gripper(value)            -> gripper ctrl, scalar or (nworld,) array
  control_substeps(ctrl, n)     -> n x [step1, controller, set_arm_ctrl, step2]

plus a few host-side (numpy) accessors used by scripts and viewers. Switching
CPU <-> GPU is `make_backend("cpu" | "warp", scene, nworld=...)`.
"""

from __future__ import annotations

import abc

import numpy as np
import warp as wp

from s2r.sim.robot_view import RobotView
from s2r.sim.scene import Scene
from s2r.sim.state import ALL_NEEDS, RobotState


class Backend(abc.ABC):
  name: str
  nworld: int
  device: wp.context.Device
  scene: Scene
  view: RobotView
  state: RobotState

  # --- physics -------------------------------------------------------------
  @abc.abstractmethod
  def reset(self) -> None: ...

  @abc.abstractmethod
  def step1(self) -> None: ...

  @abc.abstractmethod
  def step2(self) -> None: ...

  # --- robot I/O -----------------------------------------------------------
  @abc.abstractmethod
  def robot_state(self, needs=ALL_NEEDS) -> RobotState: ...

  @abc.abstractmethod
  def set_arm_ctrl(self, cmd: wp.array) -> None: ...

  @abc.abstractmethod
  def set_gripper(self, value) -> None: ...

  def control_substeps(self, controller, n: int) -> None:
    """Run `n` physics steps with the low-level controller in the loop."""
    for _ in range(n):
      self.step1()
      cmd = controller.compute(self.robot_state(controller.requires))
      self.set_arm_ctrl(cmd)
      self.step2()

  # --- host-side accessors (sync points; not for the RL hot loop) ----------
  @abc.abstractmethod
  def body_pos(self, name: str) -> np.ndarray: ...  # (nworld, 3)

  @abc.abstractmethod
  def set_mocap_pos(self, name: str, pos: np.ndarray) -> None: ...  # (nworld, 3)

  @abc.abstractmethod
  def set_body_free_pose(self, joint: str, pos: np.ndarray, quat: np.ndarray | None = None) -> None: ...

  @abc.abstractmethod
  def to_mjdata(self, world: int = 0, out=None):
    """MjData of one world (for viewer / debugging); refreshes `out` in place if given."""


def make_backend(kind: str, scene: Scene, nworld: int = 1, **kw) -> Backend:
  if kind == "cpu":
    from s2r.sim.mujoco_backend import MujocoBackend

    if nworld != 1:
      raise ValueError("cpu backend is single-world")
    return MujocoBackend(scene)
  if kind in ("warp", "gpu"):
    from s2r.sim.mjwarp_backend import MjWarpBackend

    return MjWarpBackend(scene, nworld=nworld, **kw)
  raise ValueError(f"unknown backend '{kind}' (cpu | warp)")
