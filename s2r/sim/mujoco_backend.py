"""CPU backend: native MuJoCo C, single world, float64 physics.

Dynamics quantities come straight from MuJoCo C (`mj_fullM`, `mj_jacSite`,
`qfrc_bias`) and are copied into float32 Warp CPU arrays so the *same* Warp
controller kernels run here, on the GPU, and on the real robot.
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp

from s2r.sim.backend import Backend
from s2r.sim.robot_view import RobotView
from s2r.sim.scene import Scene
from s2r.sim.state import ALL_NEEDS, NEED_BIAS, NEED_J, NEED_M, RobotState


def fill_state_from_mjdata(
  mjm: mujoco.MjModel, mjd: mujoco.MjData, view: RobotView, state: RobotState, needs=ALL_NEEDS, world: int = 0
) -> None:
  """Copy robot quantities from an MjData (after mj_step1 / mj_forward) into a CPU RobotState.

  Shared by the CPU sim backend and the real-robot dynamics model.
  """
  q, dq = state.q.numpy(), state.dq.numpy()  # zero-copy views on CPU
  q[world] = mjd.qpos[view.qpos_adr]
  dq[world] = mjd.qvel[view.dof_adr]
  state.tcp_pos.numpy()[world] = mjd.site_xpos[view.tcp_site_id]
  state.tcp_rot.numpy()[world] = mjd.site_xmat[view.tcp_site_id].reshape(3, 3)
  if NEED_M in needs:
    full = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, mjd, full)
    state.M.numpy()[world] = full[np.ix_(view.dof_adr, view.dof_adr)]
  if NEED_J in needs:
    jacp = np.zeros((3, mjm.nv))
    jacr = np.zeros((3, mjm.nv))
    mujoco.mj_jacSite(mjm, mjd, jacp, jacr, view.tcp_site_id)
    J = state.J.numpy()[world]
    J[:3] = jacp[:, view.dof_adr]
    J[3:] = jacr[:, view.dof_adr]
  if NEED_BIAS in needs:
    state.bias.numpy()[world] = mjd.qfrc_bias[view.dof_adr]


class MujocoBackend(Backend):
  name = "cpu"

  def __init__(self, scene: Scene):
    self.scene = scene
    self.mjm = scene.mjm
    self.mjd = mujoco.MjData(self.mjm)
    self.nworld = 1
    self.device = wp.get_device("cpu")
    self.views = tuple(RobotView.from_model(self.mjm, r) for r in scene.robots)
    self.states = tuple(RobotState.allocate(1, v.ndof, self.device) for v in self.views)
    self._home = self.mjm.key(scene.home_key).id
    self.reset()

  # physics
  def reset(self) -> None:
    mujoco.mj_resetDataKeyframe(self.mjm, self.mjd, self._home)
    mujoco.mj_forward(self.mjm, self.mjd)

  def step1(self) -> None:
    mujoco.mj_step1(self.mjm, self.mjd)

  def step2(self) -> None:
    mujoco.mj_step2(self.mjm, self.mjd)

  # robot I/O
  def robot_state(self, needs=ALL_NEEDS, arm: int = 0) -> RobotState:
    fill_state_from_mjdata(self.mjm, self.mjd, self.views[arm], self.states[arm], needs)
    return self.states[arm]

  def set_arm_ctrl(self, cmd: wp.array, arm: int = 0) -> None:
    self.mjd.ctrl[self.views[arm].act_ids] = cmd.numpy()[0]

  def set_gripper(self, value, arm: int = 0) -> None:
    if self.views[arm].gripper_act_id >= 0:
      self.mjd.ctrl[self.views[arm].gripper_act_id] = float(np.asarray(value).reshape(-1)[0])

  # host accessors
  def body_pos(self, name: str) -> np.ndarray:
    return self.mjd.body(name).xpos[None].copy()

  def body_quat(self, name: str) -> np.ndarray:
    return self.mjd.body(name).xquat[None].copy()

  def set_mocap_pos(self, name: str, pos: np.ndarray) -> None:
    self.mjd.mocap_pos[self.mjm.body(name).mocapid[0]] = np.asarray(pos).reshape(-1, 3)[0]

  def set_body_free_pose(self, joint: str, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
    adr = self.mjm.joint(joint).qposadr[0]
    self.mjd.qpos[adr : adr + 3] = np.asarray(pos).reshape(-1, 3)[0]
    if quat is not None:
      self.mjd.qpos[adr + 3 : adr + 7] = np.asarray(quat).reshape(-1, 4)[0]
    self.mjd.qvel[self.mjm.joint(joint).dofadr[0] : self.mjm.joint(joint).dofadr[0] + 6] = 0
    mujoco.mj_forward(self.mjm, self.mjd)

  def to_mjdata(self, world: int = 0, out=None):
    if out is not None and out is not self.mjd:
      mujoco.mj_copyData(out, self.mjm, self.mjd)
      return out
    return self.mjd
