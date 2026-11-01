"""Batched pick-and-place RL env on the MJWarp backend.

Everything per-step is a Warp kernel over worlds (no host syncs):
  action -> OSC target / gripper  ->  decimation x [step1, OSC, step2] (CUDA graph)
  -> reward/done -> masked reset (keyframe + randomized cube/goal) -> obs.
Observations, rewards and dones are Warp arrays; use `wp.to_torch` for zero-copy torch tensors.

Action (nworld, 4): [dx, dy, dz] TCP delta relative to the *current* TCP (scaled, in [-1, 1])
+ gripper (>0 closes). Relative-to-current (not accumulated) avoids target wind-up.
Tool orientation is held top-down (extend to 7-D actions by also driving target_rot).
"""

import dataclasses

import mujoco_warp as mjw
import numpy as np
import warp as wp

from s2r.control.osc import OSC
from s2r.envs.scripted import TOP_DOWN_QUAT
from s2r.control.osc import quat_wxyz_to_mat
from s2r.sim.mjwarp_backend import MjWarpBackend

OBS_DIM = 32


@dataclasses.dataclass(frozen=True)
class PickPlaceConfig:
  decimation: int = 10  # 500 Hz physics/OSC, 50 Hz policy
  episode_steps: int = 250  # 5 s
  action_scale: float = 0.05  # TCP target offset [m] from current TCP at |a| = 1
  cube_xy_range: float = 0.06  # +/- around scene cube_pos
  goal_xy_range: float = 0.06
  workspace_lo: tuple = (0.15, -0.30, 0.01)
  workspace_hi: tuple = (0.50, 0.30, 0.35)
  success_tol: float = 0.03
  seed: int = 0


@wp.kernel
def _apply_action_kernel(
  action: wp.array2d(dtype=float),
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
    p[i] = wp.clamp(p[i] + scale * wp.clamp(action[w, i], -1.0, 1.0), lo[i], hi[i])
  target_pos[w] = p
  if action[w, 3] > 0.0:
    grip[w] = grip_closed
  else:
    grip[w] = grip_open


@wp.kernel
def _reward_done_kernel(
  site_xpos: wp.array2d(dtype=wp.vec3),
  tcp_site: int,
  qpos: wp.array2d(dtype=float),
  cube_qadr: int,
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
  cube = wp.vec3(qpos[w, cube_qadr], qpos[w, cube_qadr + 1], qpos[w, cube_qadr + 2])
  tcp = site_xpos[w, tcp_site]
  d_reach = wp.length(tcp - cube)
  d_goal = wp.length(cube - goal[w])
  lifted = float(0.0)
  if cube[2] > cube_half + 0.01:
    lifted = 1.0
  ok = d_goal < tol
  r = (1.0 - wp.tanh(10.0 * d_reach)) + lifted + lifted * (1.0 - wp.tanh(5.0 * d_goal))
  if ok:
    r += 5.0
  reward[w] = r
  step_count[w] = step_count[w] + 1
  s = int(0)
  if ok:
    s = 1
  success[w] = s
  dn = int(0)
  if ok or step_count[w] >= max_steps:
    dn = 1
  done[w] = dn
  reset_key[w] = wp.where(dn == 1, 0, -1)  # keyframe 0 (home) for finished worlds


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
  home_tcp: wp.vec3,
  goal_mocap: int,
  qpos: wp.array2d(dtype=float),
  qvel: wp.array2d(dtype=float),
  mocap_pos: wp.array2d(dtype=wp.vec3),
  goal: wp.array(dtype=wp.vec3),
  target_pos: wp.array(dtype=wp.vec3),
  step_count: wp.array(dtype=int),
):
  w = wp.tid()
  if reset_key[w] < 0:
    return
  state = wp.rand_init(seed, w + epoch * 7919)
  c = cube0 + wp.vec3(wp.randf(state, -cube_range, cube_range), wp.randf(state, -cube_range, cube_range), 0.0)
  g = goal0 + wp.vec3(wp.randf(state, -goal_range, goal_range), wp.randf(state, -goal_range, goal_range), 0.0)
  qpos[w, cube_qadr + 0] = c[0]
  qpos[w, cube_qadr + 1] = c[1]
  qpos[w, cube_qadr + 2] = c[2]
  qpos[w, cube_qadr + 3] = 1.0
  qpos[w, cube_qadr + 4] = 0.0
  qpos[w, cube_qadr + 5] = 0.0
  qpos[w, cube_qadr + 6] = 0.0
  for i in range(6):
    qvel[w, cube_dofadr + i] = 0.0
  mocap_pos[w, goal_mocap] = g
  goal[w] = g
  target_pos[w] = home_tcp
  step_count[w] = 0


@wp.kernel
def _obs_kernel(
  qpos_adr: wp.array(dtype=int),
  dof_adr: wp.array(dtype=int),
  finger_qadr: int,
  tcp_site: int,
  cube_qadr: int,
  qpos: wp.array2d(dtype=float),
  qvel: wp.array2d(dtype=float),
  site_xpos: wp.array2d(dtype=wp.vec3),
  goal: wp.array(dtype=wp.vec3),
  obs: wp.array2d(dtype=float),
):
  w = wp.tid()
  k = int(0)
  for i in range(6):
    obs[w, k] = qpos[w, qpos_adr[i]]
    obs[w, k + 6] = qvel[w, dof_adr[i]]
    k += 1
  k = 12
  tcp = site_xpos[w, tcp_site]
  cube = wp.vec3(qpos[w, cube_qadr], qpos[w, cube_qadr + 1], qpos[w, cube_qadr + 2])
  g = goal[w]
  for i in range(3):
    obs[w, k + i] = tcp[i]
    obs[w, k + 4 + i] = cube[i]
    obs[w, k + 11 + i] = g[i]
    obs[w, k + 14 + i] = cube[i] - tcp[i]
    obs[w, k + 17 + i] = g[i] - cube[i]
  obs[w, k + 3] = qpos[w, finger_qadr]
  for i in range(4):
    obs[w, k + 7 + i] = qpos[w, cube_qadr + 3 + i]


class PickPlaceEnv:
  def __init__(self, backend: MjWarpBackend, cfg: PickPlaceConfig = PickPlaceConfig()):
    if not isinstance(backend, MjWarpBackend):
      raise TypeError("PickPlaceEnv runs on the MJWarp backend (use device='cpu' for Mac debugging)")
    if backend.view.ndof != 6:
      raise NotImplementedError("obs layout currently assumes a 6-DoF arm")
    self.b, self.cfg = backend, cfg
    self.nworld, self.device = backend.nworld, backend.device
    scene, mjm, view = backend.scene, backend.mjm, backend.view
    robot = scene.robot
    self.osc = OSC(view, self.nworld, device=self.device)
    self.osc.check_compatible(robot.actuator_mode)

    cube_j = mjm.joint("cube_free")
    self._cube_qadr, self._cube_dofadr = int(cube_j.qposadr[0]), int(cube_j.dofadr[0])
    self._goal_mocap = int(mjm.body(scene.goal_body).mocapid[0])
    gj = mjm.actuator(robot.gripper_actuator).trnid[0]
    self._finger_qadr = int(mjm.jnt_qposadr[gj])
    self._grip_open, self._grip_closed = robot.gripper_open, robot.gripper_closed
    self._cube0 = wp.vec3(*scene.cfg.cube_pos)
    self._goal0 = wp.vec3(*scene.cfg.goal_pos)
    self._epoch = 0

    # TCP pose at home (host, once) -> initial OSC target after each reset.
    backend.reset()
    backend.step1()
    st = backend.robot_state()
    self._home_tcp = wp.vec3(*st.tcp_pos.numpy()[0])

    with wp.ScopedDevice(self.device):
      self.obs = wp.zeros((self.nworld, OBS_DIM), dtype=float)
      self.reward = wp.zeros(self.nworld, dtype=float)
      self.done = wp.zeros(self.nworld, dtype=int)
      self.success = wp.zeros(self.nworld, dtype=int)
      self.goal = wp.zeros(self.nworld, dtype=wp.vec3)
      self.grip = wp.full(self.nworld, self._grip_open, dtype=float)
      self.step_count = wp.zeros(self.nworld, dtype=int)
      self._reset_key = wp.zeros(self.nworld, dtype=int)  # 0 = home keyframe for all
      self._qpos_adr = wp.array(view.qpos_adr, dtype=int)
      self._dof_adr = wp.array(view.dof_adr, dtype=int)
    self.osc.target_rot.assign(np.tile(quat_wxyz_to_mat(TOP_DOWN_QUAT)[0], (self.nworld, 1, 1)).astype(np.float32))

  # --- API -----------------------------------------------------------------
  def reset(self) -> wp.array:
    self._reset_key.fill_(0)
    self._masked_reset()
    self._observe()
    return self.obs

  def step(self, action: wp.array):
    """action: (nworld, 4) warp array on the env device. Returns (obs, reward, done) warp arrays."""
    b, c = self.b, self.cfg
    wp.launch(_apply_action_kernel, dim=self.nworld,
              inputs=[action, b.d.site_xpos, b.view.tcp_site_id, c.action_scale, wp.vec3(*c.workspace_lo), wp.vec3(*c.workspace_hi), self._grip_open,
                      self._grip_closed],
              outputs=[self.osc.target_pos, self.grip], device=self.device)
    b.set_gripper(self.grip)
    b.control_substeps(self.osc, c.decimation)
    mjw.kinematics(b.m, b.d)
    wp.launch(_reward_done_kernel, dim=self.nworld,
              inputs=[b.d.site_xpos, b.view.tcp_site_id, b.d.qpos, self._cube_qadr, self.goal,
                      b.scene.cfg.cube_half_size, c.success_tol, c.episode_steps, self.step_count],
              outputs=[self.reward, self.done, self.success, self._reset_key], device=self.device)
    self._masked_reset()
    self._observe()
    return self.obs, self.reward, self.done

  # --- internals -------------------------------------------------------------
  def _masked_reset(self):
    b, c = self.b, self.cfg
    mjw.reset_data_keyframe(b.m, b.d, self._reset_key)
    self._epoch += 1
    wp.launch(_randomize_kernel, dim=self.nworld,
              inputs=[c.seed, self._epoch, self._reset_key, self._cube_qadr, self._cube_dofadr, self._cube0,
                      self._goal0, c.cube_xy_range, c.goal_xy_range, self._home_tcp, self._goal_mocap],
              outputs=[b.d.qpos, b.d.qvel, b.d.mocap_pos, self.goal, self.osc.target_pos, self.step_count],
              device=self.device)

  def _observe(self):
    b = self.b
    mjw.kinematics(b.m, b.d)
    wp.launch(_obs_kernel, dim=self.nworld,
              inputs=[self._qpos_adr, self._dof_adr, self._finger_qadr, b.view.tcp_site_id, self._cube_qadr, b.d.qpos,
                      b.d.qvel, b.d.site_xpos, self.goal],
              outputs=[self.obs], device=self.device)
