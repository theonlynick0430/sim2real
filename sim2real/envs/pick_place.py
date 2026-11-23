"""Batched pick-and-place RL env on the MJWarp backend.

Everything per-step is a Warp kernel over worlds (no host syncs):
  action -> OSC target / gripper  ->  decimation x [step1, OSC, step2] (CUDA graph)
  -> reward/done -> masked reset (keyframe + randomized cube/goal) -> obs.
Observations, rewards and dones are Warp arrays; use `wp.to_torch` for zero-copy torch tensors.

Works for one arm or several (e.g. bimanual YAM): one OSC per arm, all arms in
one CUDA graph per policy step.

Action (nworld, 4 * narm), per arm: [dx, dy, dz] TCP delta relative to the *current*
TCP (scaled, in [-1, 1]) + gripper (>0 closes). Relative-to-current (not accumulated)
avoids target wind-up. Tool orientation is held top-down (extend by also driving target_rot).
"""

import dataclasses

import numpy as np
import warp as wp

from sim2real.control.osc import OSC
from sim2real.envs.scripted import TOP_DOWN_QUAT
from sim2real.sim.mjwarp_backend import MjWarpBackend
from sim2real.utils import broadcast, quat_to_mat


@dataclasses.dataclass(frozen=True)
class ObsLayout:
  """Observation layout: one block per arm, then a shared object block.

  arm block (2n + 7):  q[n] | dq[n] | tcp_pos[3] | finger[1] | cube - tcp[3]
  shared (13):         cube_pos[3] | cube_quat[4] | goal[3] | goal - cube[3]
  A single 6-DoF arm gives 32 dims. Offsets are absolute indices into the obs vector.
  """

  arm_offsets: tuple[int, ...]
  ndofs: tuple[int, ...]
  shared: int
  dim: int

  @staticmethod
  def build(ndofs) -> "ObsLayout":
    offsets, k = [], 0
    for n in ndofs:
      offsets.append(k)
      k += 2 * n + 7
    return ObsLayout(tuple(offsets), tuple(ndofs), k, k + 13)

  def q(self, a): return self.arm_offsets[a]  # noqa: E704
  def dq(self, a): return self.arm_offsets[a] + self.ndofs[a]  # noqa: E704
  def tcp(self, a): return self.arm_offsets[a] + 2 * self.ndofs[a]  # noqa: E704
  def finger(self, a): return self.tcp(a) + 3  # noqa: E704
  def cube_rel(self, a): return self.tcp(a) + 4  # noqa: E704

  @property
  def cube(self): return self.shared  # noqa: E704
  @property
  def cube_quat(self): return self.shared + 3  # noqa: E704
  @property
  def goal(self): return self.shared + 7  # noqa: E704
  @property
  def goal_rel(self): return self.shared + 10  # noqa: E704


ACTIONS_PER_ARM = 4


@dataclasses.dataclass(frozen=True)
class PickPlaceConfig:
  decimation: int = 10  # 500 Hz physics/OSC, 50 Hz policy
  episode_steps: int = 250  # 5 s
  action_scale: float = 0.05  # TCP target offset [m] from current TCP at |a| = 1
  cube_xy_range: float = 0.06  # +/- around scene cube_pos
  goal_xy_range: float = 0.06
  # Per-arm TCP target bounds, in the arm's base frame (translated by its mount position).
  workspace_lo: tuple = (0.15, -0.30, 0.01)
  workspace_hi: tuple = (0.50, 0.30, 0.35)
  success_tol: float = 0.03
  seed: int = 0


@wp.kernel
def _apply_action_kernel(
  action: wp.array2d(dtype=float),
  col: int,
  site_xpos: wp.array2d(dtype=wp.vec3),
  tcp_site: int,
  scale: float,
  lo: wp.vec3,
  hi: wp.vec3,
  grip_open: float,
  grip_closed: float,
  target_pos: wp.array(dtype=wp.vec3),
  grip: wp.array(dtype=float),
):
  w = wp.tid()
  p = site_xpos[w, tcp_site]
  for i in range(3):
    p[i] = wp.clamp(p[i] + scale * wp.clamp(action[w, col + i], -1.0, 1.0), lo[i], hi[i])
  target_pos[w] = p
  grip[w] = wp.where(action[w, col + 3] > 0.0, grip_closed, grip_open)


@wp.kernel
def _reward_done_kernel(
  site_xpos: wp.array2d(dtype=wp.vec3),
  tcp_sites: wp.array(dtype=int),
  xpos: wp.array2d(dtype=wp.vec3),
  cube_body: int,
  goal: wp.array(dtype=wp.vec3),
  cube_half: float,
  tol: float,
  max_steps: int,
  step_count: wp.array(dtype=int),
  reward: wp.array(dtype=float),
  done: wp.array(dtype=int),
  success: wp.array(dtype=int),
  reset_key: wp.array(dtype=int),
):
  w = wp.tid()
  cube = xpos[w, cube_body]
  reach = float(1.0e6)  # distance from the nearest TCP (any arm) to the cube
  for a in range(tcp_sites.shape[0]):
    reach = wp.min(reach, wp.length(site_xpos[w, tcp_sites[a]] - cube))
  to_goal = wp.length(cube - goal[w])
  lifted = wp.where(cube[2] > cube_half + 0.01, 1.0, 0.0)
  ok = to_goal < tol
  r = (1.0 - wp.tanh(10.0 * reach)) + lifted + lifted * (1.0 - wp.tanh(5.0 * to_goal))
  reward[w] = wp.where(ok, r + 5.0, r)
  step_count[w] = step_count[w] + 1
  finished = ok or step_count[w] >= max_steps
  success[w] = wp.where(ok, 1, 0)
  done[w] = wp.where(finished, 1, 0)
  reset_key[w] = wp.where(finished, 0, -1)  # keyframe 0 (home) for finished worlds


@wp.kernel
def _randomize_kernel(
  seed: int,
  epoch: int,
  reset_key: wp.array(dtype=int),
  cube_qadr: int,
  cube_dofadr: int,
  cube0: wp.vec3,
  goal0: wp.vec3,
  cube_range: float,
  goal_range: float,
  goal_mocap: int,
  qpos: wp.array2d(dtype=float),
  qvel: wp.array2d(dtype=float),
  mocap_pos: wp.array2d(dtype=wp.vec3),
  goal: wp.array(dtype=wp.vec3),
  step_count: wp.array(dtype=int),
):
  w = wp.tid()
  if reset_key[w] < 0:
    return
  rng = wp.rand_init(seed, w + epoch * 7919)
  c = cube0 + wp.vec3(wp.randf(rng, -cube_range, cube_range), wp.randf(rng, -cube_range, cube_range), 0.0)
  g = goal0 + wp.vec3(wp.randf(rng, -goal_range, goal_range), wp.randf(rng, -goal_range, goal_range), 0.0)
  for i in range(3):
    qpos[w, cube_qadr + i] = c[i]
  qpos[w, cube_qadr + 3] = 1.0  # identity quaternion (wxyz)
  for i in range(4, 7):
    qpos[w, cube_qadr + i] = 0.0
  for i in range(6):
    qvel[w, cube_dofadr + i] = 0.0
  mocap_pos[w, goal_mocap] = g
  goal[w] = g
  step_count[w] = 0


@wp.kernel
def _reset_target_kernel(reset_key: wp.array(dtype=int), home_tcp: wp.vec3, target_pos: wp.array(dtype=wp.vec3)):
  w = wp.tid()
  if reset_key[w] >= 0:
    target_pos[w] = home_tcp


@wp.kernel
def _arm_obs_kernel(
  q: wp.array2d(dtype=float),
  dq: wp.array2d(dtype=float),
  tcp_pos: wp.array(dtype=wp.vec3),
  qpos: wp.array2d(dtype=float),
  finger_qadr: int,
  xpos: wp.array2d(dtype=wp.vec3),
  cube_body: int,
  offset: int,
  obs: wp.array2d(dtype=float),
):
  w = wp.tid()
  n = q.shape[1]
  for i in range(n):
    obs[w, offset + i] = q[w, i]
    obs[w, offset + n + i] = dq[w, i]
  k = offset + 2 * n
  tcp = tcp_pos[w]
  cube = xpos[w, cube_body]
  for i in range(3):
    obs[w, k + i] = tcp[i]
    obs[w, k + 4 + i] = cube[i] - tcp[i]
  obs[w, k + 3] = qpos[w, finger_qadr]


@wp.kernel
def _shared_obs_kernel(
  xpos: wp.array2d(dtype=wp.vec3),
  xquat: wp.array2d(dtype=wp.quat),
  cube_body: int,
  goal: wp.array(dtype=wp.vec3),
  offset: int,
  obs: wp.array2d(dtype=float),
):
  w = wp.tid()
  cube = xpos[w, cube_body]
  quat = xquat[w, cube_body]  # MuJoCo order (w, x, y, z)
  g = goal[w]
  for i in range(3):
    obs[w, offset + i] = cube[i]
    obs[w, offset + 7 + i] = g[i]
    obs[w, offset + 10 + i] = g[i] - cube[i]
  for i in range(4):
    obs[w, offset + 3 + i] = quat[i]


class PickPlaceEnv:
  """Single- or multi-arm pick-and-place. One OSC per arm (tool held top-down); action is
  `ACTIONS_PER_ARM` per arm, concatenated in arm order. Reward uses the nearest TCP, so any arm
  may move the cube (e.g. a bimanual relay)."""

  def __init__(self, backend: MjWarpBackend, cfg: PickPlaceConfig = PickPlaceConfig()):
    if not isinstance(backend, MjWarpBackend):
      raise TypeError("PickPlaceEnv runs on the MJWarp backend (use device='cpu' for Mac debugging)")
    self.backend, self.cfg = backend, cfg
    self.nworld, self.device = backend.nworld, backend.device
    scene, mjm = backend.scene, backend.mjm
    self.narm = backend.narm
    self.layout = ObsLayout.build([v.ndof for v in backend.views])
    self.obs_dim = self.layout.dim
    self.action_dim = ACTIONS_PER_ARM * self.narm
    self.oscs = [OSC(v, self.nworld, device=self.device) for v in backend.views]
    for osc, robot in zip(self.oscs, scene.robots):
      osc.check_compatible(robot.actuator_mode)

    cube_joint = mjm.joint("cube_free")
    self._cube_qadr, self._cube_dofadr = int(cube_joint.qposadr[0]), int(cube_joint.dofadr[0])
    self._cube_body = mjm.body(scene.cube_body).id
    self._goal_mocap = int(mjm.body(scene.goal_body).mocapid[0])
    self._finger_qadr = [int(mjm.jnt_qposadr[mjm.actuator(r.gripper_actuator).trnid[0]]) for r in scene.robots]
    self._grip_range = [(r.gripper_ctrl(False), r.gripper_ctrl(True)) for r in scene.robots]
    self._workspace = [(wp.vec3(*np.add(cfg.workspace_lo, m.pos)), wp.vec3(*np.add(cfg.workspace_hi, m.pos)))
                       for m in scene.mounts]
    self._cube0 = wp.vec3(*scene.cfg.cube_pos)
    self._goal0 = wp.vec3(*scene.cfg.goal_pos)
    self._epoch = 0

    # TCP poses at home (host, once) -> initial OSC targets after each reset.
    backend.reset()
    backend.step1()
    self.home_tcp = [wp.vec3(*backend.robot_state(arm=a).tcp_pos.numpy()[0]) for a in range(self.narm)]

    with wp.ScopedDevice(self.device):
      self.obs = wp.zeros((self.nworld, self.obs_dim), dtype=float)
      self.reward = wp.zeros(self.nworld, dtype=float)
      self.done = wp.zeros(self.nworld, dtype=int)
      self.success = wp.zeros(self.nworld, dtype=int)
      self.goal = wp.zeros(self.nworld, dtype=wp.vec3)
      self.grip = [wp.full(self.nworld, grip_open, dtype=float) for grip_open, _ in self._grip_range]
      self.step_count = wp.zeros(self.nworld, dtype=int)
      self._reset_key = wp.zeros(self.nworld, dtype=int)  # 0 = home keyframe for all
      self._tcp_sites = wp.array([v.tcp_site_id for v in backend.views], dtype=int)
    down = broadcast(quat_to_mat(TOP_DOWN_QUAT), (self.nworld, 3, 3))
    for osc in self.oscs:
      osc.target_rot.assign(down)

  @property
  def osc(self) -> OSC:  # single-arm convenience
    return self.oscs[0]

  # --- API -----------------------------------------------------------------
  def reset(self) -> wp.array:
    self._reset_key.fill_(0)
    self._masked_reset()
    self._observe()
    return self.obs

  def step(self, action: wp.array):
    """action: (nworld, action_dim) warp array on the env device. Returns (obs, reward, done) warp arrays."""
    backend, cfg, d = self.backend, self.cfg, self.backend.d
    for arm, view in enumerate(backend.views):
      lo, hi = self._workspace[arm]
      grip_open, grip_closed = self._grip_range[arm]
      wp.launch(_apply_action_kernel, dim=self.nworld,
                inputs=[action, ACTIONS_PER_ARM * arm, d.site_xpos, view.tcp_site_id, cfg.action_scale, lo, hi,
                        grip_open, grip_closed],
                outputs=[self.oscs[arm].target_pos, self.grip[arm]], device=self.device)
      backend.set_gripper(self.grip[arm], arm=arm)
    backend.control_substeps(self.oscs, cfg.decimation)
    backend.kinematics()
    wp.launch(_reward_done_kernel, dim=self.nworld,
              inputs=[d.site_xpos, self._tcp_sites, d.xpos, self._cube_body, self.goal,
                      backend.scene.cfg.cube_half_size, cfg.success_tol, cfg.episode_steps, self.step_count],
              outputs=[self.reward, self.done, self.success, self._reset_key], device=self.device)
    self._masked_reset()
    self._observe()
    return self.obs, self.reward, self.done

  # --- internals -------------------------------------------------------------
  def _masked_reset(self):
    backend, cfg, d = self.backend, self.cfg, self.backend.d
    backend.reset_data(self._reset_key)  # not mjw.reset_data_keyframe: the backend keeps the overflow bits
    self._epoch += 1
    wp.launch(_randomize_kernel, dim=self.nworld,
              inputs=[cfg.seed, self._epoch, self._reset_key, self._cube_qadr, self._cube_dofadr, self._cube0,
                      self._goal0, cfg.cube_xy_range, cfg.goal_xy_range, self._goal_mocap],
              outputs=[d.qpos, d.qvel, d.mocap_pos, self.goal, self.step_count], device=self.device)
    for osc, home in zip(self.oscs, self.home_tcp):
      wp.launch(_reset_target_kernel, dim=self.nworld, inputs=[self._reset_key, home], outputs=[osc.target_pos],
                device=self.device)

  def _observe(self):
    backend, layout, d = self.backend, self.layout, self.backend.d
    backend.kinematics()
    for arm in range(self.narm):
      state = backend.robot_state(needs=frozenset(), arm=arm)  # q, dq, tcp only
      wp.launch(_arm_obs_kernel, dim=self.nworld,
                inputs=[state.q, state.dq, state.tcp_pos, d.qpos, self._finger_qadr[arm], d.xpos, self._cube_body,
                        layout.arm_offsets[arm]],
                outputs=[self.obs], device=self.device)
    wp.launch(_shared_obs_kernel, dim=self.nworld, inputs=[d.xpos, d.xquat, self._cube_body, self.goal, layout.shared],
              outputs=[self.obs], device=self.device)
