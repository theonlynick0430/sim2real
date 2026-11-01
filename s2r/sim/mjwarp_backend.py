"""MJWarp backend: nworld parallel worlds on a Warp device (CUDA; Warp-CPU works for debugging).

Hot path (`control_substeps`) is pure kernel launches over preallocated
buffers, so on CUDA it is captured once into a CUDA graph and replayed.
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from s2r.dynamics import kernels as K
from s2r.sim.backend import Backend
from s2r.sim.robot_view import RobotView
from s2r.sim.scene import Scene
from s2r.sim.state import ALL_NEEDS, NEED_J, NEED_M, RobotState


class MjWarpBackend(Backend):
  name = "warp"

  def __init__(
    self,
    scene: Scene,
    nworld: int = 1,
    device: str | None = None,
    nconmax: int | None = None,
    njmax: int | None = None,
    use_graph: bool = True,
  ):
    self.scene = scene
    self.mjm = scene.mjm
    self.nworld = nworld
    self.device = wp.get_device(device)
    self.view = RobotView.from_model(self.mjm, scene.robot)
    self.use_graph = use_graph and self.device.is_cuda
    self._graphs: dict = {}
    n = self.view.ndof

    with wp.ScopedDevice(self.device):
      self.m = mjw.put_model(self.mjm)
      self.d = mjw.make_data(self.mjm, nworld=nworld, nconmax=nconmax or scene.cfg.nconmax,
                             njmax=njmax or scene.cfg.njmax)
      self.state = RobotState.allocate(nworld, n, self.device)
      # index maps
      self._qpos_adr = wp.array(self.view.qpos_adr, dtype=int)
      self._dof_adr = wp.array(self.view.dof_adr, dtype=int)
      self._act_ids = wp.array(self.view.act_ids, dtype=int)
      self._m_idx = wp.array(self.view.m_csr_index, dtype=int)
      # mjw.jac scratch
      self._jacp = wp.zeros((nworld, 3, self.mjm.nv), dtype=float)
      self._jacr = wp.zeros((nworld, 3, self.mjm.nv), dtype=float)
      self._jac_point = wp.zeros(nworld, dtype=wp.vec3)
      self._jac_body = wp.full(nworld, self.view.tcp_body_id, dtype=int)
      self._grip = wp.zeros(nworld, dtype=float)
      self._home = self.mjm.key(scene.home_key).id
    self.reset()

  # --- physics -----------------------------------------------------------------
  def reset(self, world_mask: wp.array | None = None) -> None:
    """Reset to home keyframe; `world_mask` (nworld,) int keyframe ids per world (-1 = keep)."""
    with wp.ScopedDevice(self.device):
      mjw.reset_data_keyframe(self.m, self.d, self._home if world_mask is None else world_mask)
      mjw.forward(self.m, self.d)

  def step1(self) -> None:
    mjw.step1(self.m, self.d)

  def step2(self) -> None:
    mjw.step2(self.m, self.d)

  # --- robot I/O -----------------------------------------------------------------
  def robot_state(self, needs=ALL_NEEDS) -> RobotState:
    s, d, n, nw = self.state, self.d, self.view.ndof, self.nworld
    wp.launch(
      K.gather_state_kernel,
      dim=(nw, n),
      inputs=[self._qpos_adr, self._dof_adr, self.view.tcp_site_id, d.qpos, d.qvel, d.qfrc_bias, d.site_xpos, d.site_xmat],
      outputs=[s.q, s.dq, s.bias, s.tcp_pos, s.tcp_rot],
      device=self.device,
    )
    if NEED_M in needs:
      wp.launch(K.gather_mass_kernel, dim=(nw, n, n), inputs=[self._m_idx, d.M], outputs=[s.M], device=self.device)
    if NEED_J in needs:
      wp.launch(K.tcp_point_kernel, dim=nw, inputs=[self.view.tcp_site_id, d.site_xpos], outputs=[self._jac_point],
                device=self.device)
      mjw.jac(self.m, d, self._jacp, self._jacr, self._jac_point, self._jac_body)
      wp.launch(K.gather_jac_kernel, dim=(nw, n), inputs=[self._dof_adr, self._jacp, self._jacr], outputs=[s.J],
                device=self.device)
    return s

  def set_arm_ctrl(self, cmd: wp.array) -> None:
    wp.launch(K.scatter_ctrl_kernel, dim=(self.nworld, self.view.ndof), inputs=[self._act_ids, cmd],
              outputs=[self.d.ctrl], device=self.device)

  def set_gripper(self, value) -> None:
    """`value`: python float, numpy (nworld,), or warp array (nworld,) on device."""
    if self.view.gripper_act_id < 0:
      return
    if isinstance(value, wp.array):
      arr = value
    else:
      self._grip.assign(np.broadcast_to(np.asarray(value, np.float32), (self.nworld,)))
      arr = self._grip
    wp.launch(K.set_ctrl_column_kernel, dim=self.nworld, inputs=[self.view.gripper_act_id, arr],
              outputs=[self.d.ctrl], device=self.device)

  def control_substeps(self, controller, n: int) -> None:
    if not self.use_graph:
      return super().control_substeps(controller, n)
    key = (id(controller), n)
    if key not in self._graphs:
      super().control_substeps(controller, 1)  # warm-up: compile kernels outside capture
      with wp.ScopedCapture(device=self.device) as cap:
        super().control_substeps(controller, n)
      self._graphs[key] = cap.graph
    wp.capture_launch(self._graphs[key])

  # --- diagnostics -----------------------------------------------------------------
  def overflow(self) -> dict:
    """Host sync: contact / constraint buffer usage. Overflowed contacts are silently dropped."""
    return {
      "nacon": int(self.d.nacon.numpy()[0]),
      "naconmax": int(self.d.naconmax),
      "nefc_max": int(self.d.nefc.numpy().max()),
      "njmax": int(self.d.njmax),
    }

  # --- host accessors -----------------------------------------------------------------
  def body_pos(self, name: str) -> np.ndarray:
    return self.d.xpos.numpy()[:, self.mjm.body(name).id]

  def set_mocap_pos(self, name: str, pos: np.ndarray) -> None:
    mp = self.d.mocap_pos.numpy()
    mp[:, self.mjm.body(name).mocapid[0]] = np.broadcast_to(pos, (self.nworld, 3))
    self.d.mocap_pos.assign(mp)

  def set_body_free_pose(self, joint: str, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
    jnt = self.mjm.joint(joint)
    qpos, qvel = self.d.qpos.numpy(), self.d.qvel.numpy()
    a, v = jnt.qposadr[0], jnt.dofadr[0]
    qpos[:, a : a + 3] = np.broadcast_to(pos, (self.nworld, 3))
    if quat is not None:
      qpos[:, a + 3 : a + 7] = np.broadcast_to(quat, (self.nworld, 4))
    qvel[:, v : v + 6] = 0
    self.d.qpos.assign(qpos)
    self.d.qvel.assign(qvel)
    mjw.forward(self.m, self.d)

  def to_mjdata(self, world: int = 0, out=None):
    mjd = mujoco.MjData(self.mjm) if out is None else out
    mjw.get_data_into(mjd, self.mjm, self.d, world_id=world)
    return mjd
