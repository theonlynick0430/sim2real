"""MJWarp backend: nworld parallel worlds on a Warp device (CUDA; Warp-CPU works for debugging).

Hot path (`control_substeps`) is pure kernel launches over preallocated
buffers, so on CUDA it is captured once into a CUDA graph and replayed.
"""

from __future__ import annotations

import dataclasses

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from sim2real.dynamics import kernels as K
from sim2real.sim.backend import Backend, as_controller_list
from sim2real.sim.robot_view import RobotView
from sim2real.sim.scene import Scene
from sim2real.sim.state import ALL_NEEDS, NEED_J, NEED_M, RobotState


@dataclasses.dataclass
class _ArmIndex:
  """Per-arm device-side index maps (built once from the arm's RobotView)."""

  qpos_adr: wp.array
  dof_adr: wp.array
  act_ids: wp.array
  m_idx: wp.array
  jac_body: wp.array  # (nworld,) TCP body id, as mjw.jac expects
  grip: wp.array  # (nworld,) host-set gripper values


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
    self.views = tuple(RobotView.from_model(self.mjm, r) for r in scene.robots)
    self.use_graph = use_graph and self.device.is_cuda
    self._graphs: dict = {}

    with wp.ScopedDevice(self.device):
      self.m = mjw.put_model(self.mjm)
      self.d = mjw.make_data(self.mjm, nworld=nworld, nconmax=nconmax or scene.cfg.nconmax,
                             njmax=njmax or scene.cfg.njmax)
      self.states = tuple(RobotState.allocate(nworld, v.ndof, self.device) for v in self.views)
      self._arms = tuple(
        _ArmIndex(
          qpos_adr=wp.array(v.qpos_adr, dtype=int),
          dof_adr=wp.array(v.dof_adr, dtype=int),
          act_ids=wp.array(v.act_ids, dtype=int),
          m_idx=wp.array(v.m_csr_index, dtype=int),
          jac_body=wp.full(nworld, v.tcp_body_id, dtype=int),
          grip=wp.zeros(nworld, dtype=float),
        )
        for v in self.views
      )
      # mjw.jac scratch, shared by arms (each arm gathers its columns right after its jac call)
      self._jacp = wp.zeros((nworld, 3, self.mjm.nv), dtype=float)
      self._jacr = wp.zeros((nworld, 3, self.mjm.nv), dtype=float)
      self._jac_point = wp.zeros(nworld, dtype=wp.vec3)
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
  def robot_state(self, needs=ALL_NEEDS, arm: int = 0) -> RobotState:
    s, v, ix, d, nw = self.states[arm], self.views[arm], self._arms[arm], self.d, self.nworld
    n = v.ndof
    wp.launch(
      K.gather_state_kernel,
      dim=(nw, n),
      inputs=[ix.qpos_adr, ix.dof_adr, v.tcp_site_id, d.qpos, d.qvel, d.qfrc_bias, d.site_xpos, d.site_xmat],
      outputs=[s.q, s.dq, s.bias, s.tcp_pos, s.tcp_rot],
      device=self.device,
    )
    if NEED_M in needs:
      wp.launch(K.gather_mass_kernel, dim=(nw, n, n), inputs=[ix.m_idx, d.M], outputs=[s.M], device=self.device)
    if NEED_J in needs:
      wp.launch(K.tcp_point_kernel, dim=nw, inputs=[v.tcp_site_id, d.site_xpos], outputs=[self._jac_point],
                device=self.device)
      mjw.jac(self.m, d, self._jacp, self._jacr, self._jac_point, ix.jac_body)
      wp.launch(K.gather_jac_kernel, dim=(nw, n), inputs=[ix.dof_adr, self._jacp, self._jacr], outputs=[s.J],
                device=self.device)
    return s

  def set_arm_ctrl(self, cmd: wp.array, arm: int = 0) -> None:
    wp.launch(K.scatter_ctrl_kernel, dim=(self.nworld, self.views[arm].ndof), inputs=[self._arms[arm].act_ids, cmd],
              outputs=[self.d.ctrl], device=self.device)

  def set_gripper(self, value, arm: int = 0) -> None:
    """`value`: python float, numpy (nworld,), or warp array (nworld,) on device."""
    act = self.views[arm].gripper_act_id
    if act < 0:
      return
    if isinstance(value, wp.array):
      arr = value
    else:
      arr = self._arms[arm].grip
      arr.assign(np.broadcast_to(np.asarray(value, np.float32), (self.nworld,)))
    wp.launch(K.set_ctrl_column_kernel, dim=self.nworld, inputs=[act, arr], outputs=[self.d.ctrl], device=self.device)

  def control_substeps(self, controllers, n: int) -> None:
    if not self.use_graph:
      return super().control_substeps(controllers, n)
    ctrls = tuple(as_controller_list(controllers, self.narm))
    key = (tuple(map(id, ctrls)), n)
    if key not in self._graphs:
      super().control_substeps(ctrls, 1)  # warm-up: compile kernels outside capture
      with wp.ScopedCapture(device=self.device) as cap:
        super().control_substeps(ctrls, n)
      # Keep the controllers alive with their graph: the key uses id(), which Python may reuse
      # after garbage collection, and the graph bakes in the controllers' buffer addresses.
      self._graphs[key] = (cap.graph, ctrls)
    wp.capture_launch(self._graphs[key][0])

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

  def body_quat(self, name: str) -> np.ndarray:
    return self.d.xquat.numpy()[:, self.mjm.body(name).id]

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
