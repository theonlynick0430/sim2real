"""MJWarp backend: the one simulation path. nworld parallel worlds on a Warp device.

The same code runs on CUDA (training) and on Warp-CPU (macOS debugging / viewing); only CUDA graph
capture is device-specific. Surface:

  reset()                       -> all worlds to the scene's home keyframe
  step1() / step2()             -> split physics step; controllers run in between on *current* M, J, bias
  kinematics()                  -> refresh positions (site/body poses) without stepping
  robot_state(needs, arm=0)     -> RobotState of one arm (warp arrays, backend device)
  set_arm_ctrl(cmd, arm=0)      -> write (nworld, n) command into that arm's actuators
  set_gripper(value, arm=0)     -> gripper ctrl, scalar or (nworld,) array
  control_substeps(ctrls, n)    -> n x [step1, per-arm controller + set_arm_ctrl, step2]

Multi-arm scenes (e.g. bimanual) have one RobotView / RobotState per arm (`views`, `states`); `view` /
`state` are arm 0. Arms on fixed bases have a block-diagonal mass matrix, so per-arm controllers are exact.
Host-side (numpy) accessors at the bottom are sync points for scripts, tests and viewers, not the RL loop.

Hot path (`control_substeps`) is pure kernel launches over preallocated buffers, so on CUDA it is
captured once into a CUDA graph and replayed.
"""

from __future__ import annotations

import dataclasses

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from sim2real.dynamics import kernels
from sim2real.sim.robot_view import RobotView
from sim2real.sim.scene import Scene
from sim2real.sim.state import ALL_NEEDS, NEED_J, NEED_M, RobotState
from sim2real.utils import broadcast


@wp.kernel
def _or_overflow_kernel(src: wp.array(dtype=int), bits: wp.array(dtype=int)):
  """bits |= src, per world. MJWarp's reset_data_keyframe zeroes d.overflow for the worlds it resets, so the
  backend folds them into `bits` before every reset; `overflow()` reports bits | d.overflow."""
  w = wp.tid()
  bits[w] = bits[w] | src[w]


@dataclasses.dataclass
class _ArmIndex:
  """Per-arm device-side index maps (built once from the arm's RobotView)."""

  qpos_adr: wp.array
  dof_adr: wp.array
  act_ids: wp.array
  m_idx: wp.array
  jac_body: wp.array  # (nworld,) TCP body id, as mjw.jac expects
  grip: wp.array  # (nworld,) host-set gripper values


def _as_tuple(controllers, narm: int) -> tuple:
  ctrls = tuple(controllers) if isinstance(controllers, (list, tuple)) else (controllers,)
  if len(ctrls) > narm:
    raise ValueError(f"{len(ctrls)} controllers for {narm} arm(s)")
  return ctrls


class MjWarpBackend:
  def __init__(
    self,
    scene: Scene,
    nworld: int = 1,
    device: str | None = None,
    nconmax: int | None = None,
    njmax: int | None = None,
    nccdmax: int | None = None,
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
                             njmax=njmax or scene.cfg.njmax,
                             nccdmax=scene.cfg.nccdmax if nccdmax is None else nccdmax)
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
      self._overflow_bits = wp.zeros(nworld, dtype=int)  # see _or_overflow_kernel
      self._home = self.mjm.key(scene.home_key).id
    self.reset()

  @property
  def view(self) -> RobotView:
    return self.views[0]

  @property
  def state(self) -> RobotState:
    return self.states[0]

  @property
  def narm(self) -> int:
    return len(self.views)

  # --- physics -----------------------------------------------------------------
  # Every MJWarp call runs under ScopedDevice: MJWarp launches on the current Warp device, which is cuda:0 on a GPU
  # machine even when this backend lives on the CPU.
  def reset_data(self, world_mask: wp.array | None = None) -> None:
    """Masked keyframe reset of `d` only (no forward): `world_mask` (nworld,) int keyframe ids, -1 = keep.

    Always reset through here, not `mjw.reset_data_keyframe` directly, so the overflow bits MJWarp zeroes on
    reset are saved first (see `overflow()`).
    """
    with wp.ScopedDevice(self.device):
      wp.launch(_or_overflow_kernel, dim=self.nworld, inputs=[self.d.overflow, self._overflow_bits])
      mjw.reset_data_keyframe(self.m, self.d, self._home if world_mask is None else world_mask)

  def reset(self, world_mask: wp.array | None = None) -> None:
    """Reset to home keyframe (`world_mask` as in `reset_data`) and run forward."""
    self.reset_data(world_mask)
    self.forward()

  def forward(self) -> None:
    with wp.ScopedDevice(self.device):
      mjw.forward(self.m, self.d)

  def kinematics(self) -> None:
    with wp.ScopedDevice(self.device):
      mjw.kinematics(self.m, self.d)

  def step1(self) -> None:
    with wp.ScopedDevice(self.device):
      mjw.step1(self.m, self.d)

  def step2(self) -> None:
    with wp.ScopedDevice(self.device):
      mjw.step2(self.m, self.d)

  # --- robot I/O -----------------------------------------------------------------
  def robot_state(self, needs=ALL_NEEDS, arm: int = 0) -> RobotState:
    state, view, index, d = self.states[arm], self.views[arm], self._arms[arm], self.d
    n, nworld = view.ndof, self.nworld
    with wp.ScopedDevice(self.device):
      wp.launch(
        kernels.gather_state_kernel,
        dim=(nworld, n),
        inputs=[index.qpos_adr, index.dof_adr, view.tcp_site_id, d.qpos, d.qvel, d.qfrc_bias, d.site_xpos,
                d.site_xmat],
        outputs=[state.q, state.dq, state.bias, state.tcp_pos, state.tcp_rot],
      )
      if NEED_M in needs:
        wp.launch(kernels.gather_mass_kernel, dim=(nworld, n, n), inputs=[index.m_idx, d.M], outputs=[state.M])
      if NEED_J in needs:
        wp.launch(kernels.tcp_point_kernel, dim=nworld, inputs=[view.tcp_site_id, d.site_xpos],
                  outputs=[self._jac_point])
        mjw.jac(self.m, d, self._jacp, self._jacr, self._jac_point, index.jac_body)
        wp.launch(kernels.gather_jac_kernel, dim=(nworld, n), inputs=[index.dof_adr, self._jacp, self._jacr],
                  outputs=[state.J])
    return state

  def set_arm_ctrl(self, cmd: wp.array, arm: int = 0) -> None:
    wp.launch(kernels.scatter_ctrl_kernel, dim=(self.nworld, self.views[arm].ndof),
              inputs=[self._arms[arm].act_ids, cmd], outputs=[self.d.ctrl], device=self.device)

  def set_gripper(self, value, arm: int = 0) -> None:
    """`value`: python float, numpy (nworld,), or warp array (nworld,) on device."""
    actuator = self.views[arm].gripper_act_id
    if actuator < 0:
      return
    if not isinstance(value, wp.array):
      self._arms[arm].grip.assign(broadcast(value, self.nworld))
      value = self._arms[arm].grip
    wp.launch(kernels.set_ctrl_column_kernel, dim=self.nworld, inputs=[actuator, value], outputs=[self.d.ctrl],
              device=self.device)

  def control_substeps(self, controllers, n: int) -> None:
    """Run `n` physics steps with the low-level controller(s) in the loop.

    `controllers`: one controller (arm 0) or a sequence with one entry per arm (None = leave that arm's
    ctrl unchanged). All arms see the same physics step. On CUDA the n steps are one replayed graph.
    """
    ctrls = _as_tuple(controllers, self.narm)
    if not self.use_graph:
      return self._substeps(ctrls, n)
    key = (tuple(map(id, ctrls)), n)
    if key not in self._graphs:
      self._substeps(ctrls, 1)  # warm-up: compile kernels outside capture
      with wp.ScopedCapture(device=self.device) as cap:
        self._substeps(ctrls, n)
      # Keep the controllers alive with their graph: the key uses id(), which Python may reuse
      # after garbage collection, and the graph bakes in the controllers' buffer addresses.
      self._graphs[key] = (cap.graph, ctrls)
    wp.capture_launch(self._graphs[key][0])

  def _substeps(self, ctrls, n: int) -> None:
    for _ in range(n):
      self.step1()
      for arm, ctrl in enumerate(ctrls):
        if ctrl is not None:
          self.set_arm_ctrl(ctrl.compute(self.robot_state(ctrl.requires, arm)), arm)
      self.step2()

  # --- diagnostics -----------------------------------------------------------------
  def overflow(self, clear: bool = False) -> dict:
    """Host sync: contact / constraint buffer usage plus MJWarp's overflow bits.

    `nacon` / `nefc_max` are the current step's usage. `flags` are the OverflowType names set in any world
    since construction (or the last `clear=True` call). MJWarp ORs bits into `d.overflow` every step but
    zeroes a world's bits when that world is reset, so the backend accumulates them across resets.
    `contacts` / `ccd` / `nefc` are the bits that mean a contact or constraint was silently dropped;
    `ITERATIONS` / `LS_ITERATIONS` only mean the solver hit its iteration cap.
    """
    bits = int(np.bitwise_or.reduce(self.d.overflow.numpy() | self._overflow_bits.numpy()))
    if clear:
      self.d.overflow.zero_()
      self._overflow_bits.zero_()
    flag = mjw.OverflowType
    return {
      "nacon": int(self.d.nacon.numpy()[0]),
      "naconmax": int(self.d.naconmax),
      "naccdmax": int(self.d.naccdmax),
      "nefc_max": int(self.d.nefc.numpy().max()),
      "njmax": int(self.d.njmax),
      "contacts": bool(bits & (flag.BROADPHASE | flag.NARROWPHASE)),
      "ccd": bool(bits & flag.CCD),
      "nefc": bool(bits & (flag.NEFC | flag.NJMAX_NNZ)),
      "flags": [f.name for f in flag if f.value and f != flag.ALL and bits & f.value],
    }

  # --- host accessors -----------------------------------------------------------------
  def body_pos(self, name: str) -> np.ndarray:
    return self.d.xpos.numpy()[:, self.mjm.body(name).id]

  def body_quat(self, name: str) -> np.ndarray:
    return self.d.xquat.numpy()[:, self.mjm.body(name).id]

  def set_mocap_pos(self, name: str, pos: np.ndarray) -> None:
    mocap_pos = self.d.mocap_pos.numpy()
    mocap_pos[:, self.mjm.body(name).mocapid[0]] = broadcast(pos, (self.nworld, 3))
    self.d.mocap_pos.assign(mocap_pos)

  def set_body_free_pose(self, joint: str, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
    """Place a free-joint body (same pose in every world), zero its velocity and run forward."""
    jnt = self.mjm.joint(joint)
    qpos, qvel = self.d.qpos.numpy(), self.d.qvel.numpy()
    q0, v0 = jnt.qposadr[0], jnt.dofadr[0]
    qpos[:, q0 : q0 + 3] = broadcast(pos, (self.nworld, 3))
    if quat is not None:
      qpos[:, q0 + 3 : q0 + 7] = broadcast(quat, (self.nworld, 4))
    qvel[:, v0 : v0 + 6] = 0
    self.d.qpos.assign(qpos)
    self.d.qvel.assign(qvel)
    self.forward()

  def to_mjdata(self, world: int = 0, out=None):
    """MjData of one world (for the passive viewer / debugging); refreshes `out` in place if given."""
    mjd = mujoco.MjData(self.mjm) if out is None else out
    mjw.get_data_into(mjd, self.mjm, self.d, world_id=world)
    return mjd
