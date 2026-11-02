"""Unified simulation backend interface.

Both backends expose the same minimal surface:

  reset()                       -> all worlds to the scene's home keyframe
  step1() / step2()             -> split physics step (MuJoCo mj_step1/mj_step2 semantics);
                                   controllers run in between on *current* M, J, bias
  robot_state(needs, arm=0)     -> RobotState of one arm (warp arrays, backend device)
  set_arm_ctrl(cmd, arm=0)      -> write (nworld, n) command into that arm's actuators
  set_gripper(value, arm=0)     -> gripper ctrl, scalar or (nworld,) array
  control_substeps(ctrls, n)    -> n x [step1, per-arm controller + set_arm_ctrl, step2]

Multi-arm scenes (e.g. bimanual) have one RobotView / RobotState per arm
(`views`, `states`); `view` / `state` are arm 0. Arms on fixed bases have a
block-diagonal mass matrix, so independent per-arm controllers are exact.

plus a few host-side (numpy) accessors used by scripts and viewers. Switching
CPU <-> GPU is `make_backend("cpu" | "warp", scene, nworld=...)`.
"""

from __future__ import annotations

import abc

import numpy as np
import warp as wp

from sim2real.sim.robot_view import RobotView
from sim2real.sim.scene import Scene
from sim2real.sim.state import ALL_NEEDS, RobotState


class Backend(abc.ABC):
  name: str
  nworld: int
  device: wp.context.Device
  scene: Scene
  views: tuple[RobotView, ...]  # one per arm (scene.mounts order)
  states: tuple[RobotState, ...]

  @property
  def view(self) -> RobotView:
    return self.views[0]

  @property
  def state(self) -> RobotState:
    return self.states[0]

  @property
  def narm(self) -> int:
    return len(self.views)

  # --- physics -------------------------------------------------------------
  @abc.abstractmethod
  def reset(self) -> None: ...

  @abc.abstractmethod
  def step1(self) -> None: ...

  @abc.abstractmethod
  def step2(self) -> None: ...

  # --- robot I/O -----------------------------------------------------------
  @abc.abstractmethod
  def robot_state(self, needs=ALL_NEEDS, arm: int = 0) -> RobotState: ...

  @abc.abstractmethod
  def set_arm_ctrl(self, cmd: wp.array, arm: int = 0) -> None: ...

  @abc.abstractmethod
  def set_gripper(self, value, arm: int = 0) -> None: ...

  def control_substeps(self, controllers, n: int) -> None:
    """Run `n` physics steps with the low-level controller(s) in the loop.

    `controllers`: one controller (arm 0) or a sequence with one entry per arm
    (None = leave that arm's ctrl unchanged). All arms see the same physics step.
    """
    ctrls = as_controller_list(controllers, self.narm)
    for _ in range(n):
      self.step1()
      for arm, c in enumerate(ctrls):
        if c is not None:
          self.set_arm_ctrl(c.compute(self.robot_state(c.requires, arm)), arm)
      self.step2()

  # --- host-side accessors (sync points; not for the RL hot loop) ----------
  @abc.abstractmethod
  def body_pos(self, name: str) -> np.ndarray: ...  # (nworld, 3)

  @abc.abstractmethod
  def body_quat(self, name: str) -> np.ndarray: ...  # (nworld, 4) wxyz

  @abc.abstractmethod
  def set_mocap_pos(self, name: str, pos: np.ndarray) -> None: ...  # (nworld, 3)

  @abc.abstractmethod
  def set_body_free_pose(self, joint: str, pos: np.ndarray, quat: np.ndarray | None = None) -> None: ...

  @abc.abstractmethod
  def to_mjdata(self, world: int = 0, out=None):
    """MjData of one world (for viewer / debugging); refreshes `out` in place if given."""


def as_controller_list(controllers, narm: int) -> list:
  ctrls = list(controllers) if isinstance(controllers, (list, tuple)) else [controllers]
  if len(ctrls) > narm:
    raise ValueError(f"{len(ctrls)} controllers for {narm} arm(s)")
  return ctrls


def make_backend(kind: str, scene: Scene, nworld: int = 1, **kw) -> Backend:
  if kind == "cpu":
    from sim2real.sim.mujoco_backend import MujocoBackend

    if nworld != 1:
      raise ValueError("cpu backend is single-world")
    return MujocoBackend(scene)
  if kind in ("warp", "gpu"):
    from sim2real.sim.mjwarp_backend import MjWarpBackend

    return MjWarpBackend(scene, nworld=nworld, **kw)
  raise ValueError(f"unknown backend '{kind}' (cpu | warp)")
