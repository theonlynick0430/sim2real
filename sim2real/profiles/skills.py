"""Compose `joint_move` / `ee_move` skills into trajectories and roll them out on MJWarp, recording per-motor
position, velocity, motor torque and external torque at every physics step.

  joint_move  large move to a sampled joint goal: `GpuMotionPlanner.plan_joint`, tracked by computed torque
              (`JointImpedance(inertia_shaped=True)`) with position / velocity / acceleration targets.
  ee_move     small move to a nearby TCP pose: `GpuMotionPlanner.plan_pose`, tracked by `OSC` with pose /
              twist / acceleration targets (FK of the joint plan).

Plans arrive at the planner's `interpolation_dt` (50 Hz) and are resampled to the physics rate so the controller
gets a new target every step (`sim2real.utils.resample_trajectory`).

Torques, per arm joint, in Nm at the joint output, positive along the joint axis:
  tau_motor  what the actuator applies (`qfrc_actuator`, the controller command after clamping).
  tau_ext    what the rest of the arm (links, payload, gravity, coupling) applies to the motor's output:
             armature * qacc - tau_motor (minus any passive / constraint joint torque, zero for YAM: the
             model has no joint friction, and trajectories stay off the joint limits).
The rotor inertia (armature) belongs to the motor, so it is not part of tau_ext. On a dyno: make the motor
follow (q, dq) while the load side applies tau_ext; an ideal motor then needs tau_motor.
"""

from __future__ import annotations

import contextlib
import dataclasses

import numpy as np
import warp as wp

from sim2real.control import OSC, JointImpedance
from sim2real.planning import rotation_error
from sim2real.profiles.sampling import (Checker, SamplingConfig, sample_ee_goal, sample_joint_goal, sample_payload,
                                        sample_start)
from sim2real.profiles.scene import set_payload
from sim2real.sim.state import ALL_NEEDS
from sim2real.utils import quat_to_mat, resample_trajectory

JOINT_MOVE, EE_MOVE = "joint_move", "ee_move"
HOLD = -1  # value of the per-sample `skill` column between skills


@dataclasses.dataclass(frozen=True)
class RolloutConfig:
  """Controller gains and acceptance thresholds (compared on `--resume`, like `SamplingConfig`)."""

  # Gains are in acceleration units (both controllers scale them by the inertia), critically damped
  # (kd = 2 sqrt(kp)). The model is exact, so the feedforward does the tracking and these only absorb
  # integration error; they are kept moderate so tau_motor is not shaped by a stiff feedback loop.
  joint_kp: float = 100.0  # computed torque: ddq = ddq_des + kp e + kd de (10 rad/s natural frequency)
  osc_kp_pos: float = 400.0  # the OSC class defaults
  osc_kp_rot: float = 150.0
  # Acceptance. Tracking is ~1 mrad / ~1 mm; larger errors mean something went wrong. Any sample with a motor
  # at its torque limit, or any contact in the sim, rejects the trajectory.
  max_joint_error: float = 0.05  # rad, worst joint tracking error allowed in a joint move
  max_ee_pos_error: float = 0.01  # m, worst TCP tracking error allowed in an ee move
  max_ee_rot_error: float = 0.1  # rad
  plan_retries: int = 12  # goals tried per skill before giving up on it
  check_stride: int = 10  # physics-rate samples between validity checks along a plan


class Rejected(Exception):
  """This attempt at a trajectory is unusable (bad tracking, saturation, no plan, ...); try another."""


@wp.kernel
def _record_kernel(
  dof_adr: wp.array(dtype=int),
  q: wp.array2d(dtype=float),
  dq: wp.array2d(dtype=float),
  qacc: wp.array2d(dtype=float),
  qfrc_actuator: wp.array2d(dtype=float),
  qfrc_passive: wp.array2d(dtype=float),
  qfrc_constraint: wp.array2d(dtype=float),
  armature: wp.array(dtype=float),
  out: wp.array2d(dtype=float),  # (4, n): q, dq, tau_motor, tau_ext of world 0
):
  i = wp.tid()
  d = dof_adr[i]
  out[0, i] = q[0, i]
  out[1, i] = dq[0, i]
  out[2, i] = qfrc_actuator[0, d]
  out[3, i] = armature[i] * qacc[0, d] - qfrc_actuator[0, d] - qfrc_passive[0, d] - qfrc_constraint[0, d]


@wp.kernel
def _contact_kernel(nacon: wp.array(dtype=int), most: wp.array(dtype=int)):
  wp.atomic_max(most, 0, nacon[0])


class ProfileGenerator:
  """Generates trajectories on one single-world backend (reused across trajectories)."""

  def __init__(self, backend, planner, fk, sampling: SamplingConfig = SamplingConfig(),
               rollout: RolloutConfig = RolloutConfig(), seed: int = 0, viewer=None, view_stride: int = 10):
    if backend.nworld != 1 or backend.narm != 1:
      raise ValueError("motor profiles use one world and one arm")
    self.backend, self.planner, self.fk = backend, planner, fk
    self.sampling, self.rollout, self.seed = sampling, rollout, seed
    self.viewer, self.view_stride = viewer, view_stride
    self.robot, self.view = backend.scene.robot, backend.view
    self.dt = float(backend.mjm.opt.timestep)
    self.checker = Checker(backend.mjm, self.robot, sampling)
    device = backend.device
    self.joint_ctrl = JointImpedance(self.view, 1, stiffness=rollout.joint_kp, inertia_shaped=True, device=device)
    self.osc = OSC(self.view, 1, kp_pos=rollout.osc_kp_pos, kp_rot=rollout.osc_kp_rot, device=device)
    with wp.ScopedDevice(device):
      self._dof_adr = wp.array(self.view.dof_adr, dtype=int)
      self._armature = wp.array(backend.mjm.dof_armature[self.view.dof_adr], dtype=float)
      self._sample = wp.zeros((4, self.view.ndof), dtype=float)
      self._contacts = wp.zeros(1, dtype=int)  # most contacts seen in any step of the current trajectory

  # --- public -----------------------------------------------------------------
  def generate(self, index: int, attempt: int = 0) -> tuple[dict, dict]:
    """Trajectory `index` -> (arrays, metadata). Raises `Rejected`.

    Everything sampled here (start, goals, payloads, speeds) depends only on (seed, index, attempt). The
    planner keeps its own random state, so regenerating an index gives the same goals but plans that can
    differ slightly."""
    cfg, rng = self.sampling, np.random.default_rng([self.seed, index, attempt])
    self._rows, self._skill_col = [], []
    q_start = sample_start(self.checker, index, attempt, self.seed)
    payload = sample_payload(cfg, rng)
    self._reset(q_start, payload)

    weights = np.array([cfg.joint_weight, cfg.ee_weight])
    skills, visited = [], [q_start]
    for k in range(int(rng.integers(cfg.min_skills, cfg.max_skills + 1))):
      if k > 0 and rng.uniform() < cfg.payload_change_prob:
        payload = sample_payload(cfg, rng)
        self._set_payload(payload)
      self._hold(rng.uniform(*cfg.hold_range))
      kind = (JOINT_MOVE, EE_MOVE)[rng.choice(2, p=weights / weights.sum())]
      first = len(self._rows)
      info = self._ee_move(rng, k) if kind == EE_MOVE else None
      if info is None:  # joint move, or no usable ee move from here (e.g. near a singularity)
        info = self._joint_move(rng, k, visited)
        if kind == EE_MOVE:
          kind, info = JOINT_MOVE, {**info, "fallback_from": EE_MOVE}
      visited.append(self._q())
      skills.append({"type": kind, "first_sample": first, "num_samples": len(self._rows) - first,
                     "payload_mass": payload[0], "payload_com_tool": np.round(payload[1], 6).tolist(), **info})
    self._hold(cfg.hold_range[1])

    rows = np.asarray(self._rows, np.float32)  # (T, 4, n)
    self._accept(rows)
    arrays = {
      "t": (np.arange(len(rows)) * self.dt).astype(np.float32),
      "q": rows[:, 0], "dq": rows[:, 1], "tau_motor": rows[:, 2], "tau_ext": rows[:, 3],
      "skill": np.asarray(self._skill_col, np.int16),
    }
    meta = {"index": index, "attempt": attempt, "dt": self.dt, "num_samples": len(rows),
            "start_q": np.round(q_start, 6).tolist(),
            "skills": skills}
    return arrays, meta

  # --- skills -----------------------------------------------------------------
  def _joint_move(self, rng, k: int, visited) -> dict:
    for _ in range(self.rollout.plan_retries):
      goal = sample_joint_goal(self.checker, rng, self._q(), visited)
      scale = float(rng.uniform(*self.sampling.time_scale_range))
      plan = self._plan(self.planner.plan_joint(self._q(), goal), scale)
      if plan is None or self.checker.path_problem(plan[0], self.rollout.check_stride):
        continue
      q, dq, ddq = plan
      first = len(self._rows)
      self._run(self.joint_ctrl, k, len(q), lambda i: self.joint_ctrl.set_target(q[i], dq[i], ddq[i]))
      error = float(np.abs(self._recorded_q(first) - q).max())
      if error > self.rollout.max_joint_error:
        raise Rejected(f"joint_move tracking error {error:.3f} rad")
      return {"goal_q": np.round(goal, 6).tolist(), "time_scale": scale, "max_joint_error": error}
    raise Rejected("no valid joint_move plan")

  def _ee_move(self, rng, k: int) -> dict | None:
    """None if no valid plan was found from the current configuration (the caller falls back to a joint move)."""
    cfg = self.sampling
    q_now = self._q()
    if self.checker.min_sigma(q_now) < cfg.min_sigma:
      return None
    pos_now, quat_now = (x[0] for x in self.fk(q_now))
    for _ in range(self.rollout.plan_retries):
      pos, quat = sample_ee_goal(cfg, rng, pos_now, quat_now)
      scale = float(rng.uniform(*cfg.time_scale_range))
      plan = self._plan(self.planner.plan_pose(q_now, pos, quat), scale)
      if plan is None or np.abs(plan[0] - q_now).max() > cfg.ee_max_joint_step:
        continue  # no plan, or the planner picked another IK branch (not a small move any more)
      if self.checker.path_problem(plan[0], self.rollout.check_stride, sigma=True):
        continue
      q, dq, _ = plan
      ee_pos, ee_quat, ee_vel = self.fk.ee_trajectory(q, dq)
      ee_rot, ee_acc = quat_to_mat(ee_quat), np.gradient(ee_vel, self.dt, axis=0) if len(q) > 1 else 0 * ee_vel
      first = len(self._rows)
      self._run(self.osc, k, len(q),
                lambda i: self.osc.set_target(ee_pos[i], rot=ee_rot[i], vel=ee_vel[i], acc=ee_acc[i]))
      got_pos, got_quat = self.fk(self._recorded_q(first))
      pos_error = float(np.linalg.norm(got_pos - ee_pos, axis=-1).max())
      rot_error = float(rotation_error(got_quat, ee_quat).max())
      if pos_error > self.rollout.max_ee_pos_error or rot_error > self.rollout.max_ee_rot_error:
        raise Rejected(f"ee_move tracking error {pos_error * 1e3:.1f} mm / {rot_error:.3f} rad")
      return {"goal_pos": np.round(pos, 6).tolist(), "goal_quat_wxyz": np.round(quat, 6).tolist(),
              "time_scale": scale, "max_pos_error": pos_error, "max_rot_error": rot_error}
    return None

  def _plan(self, traj, time_scale: float):
    """Planner result for one problem -> (q, dq, ddq) at the physics rate, or None if planning failed."""
    traj = traj.numpy()
    if not traj.success[0] or traj.length[0] < 2:
      return None
    n = int(traj.length[0])
    return resample_trajectory(traj.q[0, :n], traj.dq[0, :n], traj.ddq[0, :n], traj.dt, self.dt, time_scale)

  # --- rollout -----------------------------------------------------------------
  def _reset(self, q_start, payload) -> None:
    backend = self.backend
    self._set_payload(payload)
    backend.reset()
    backend.set_gripper(self.robot.gripper_ctrl(closed=True))
    backend.set_arm_state(q_start)
    self._contacts.zero_()
    self.joint_ctrl.set_target(q_start)
    self._active = self.joint_ctrl

  def _set_payload(self, payload) -> None:
    # The passive viewer renders from the same host model on another thread: edit it under the viewer's lock.
    handle = getattr(self.viewer, "handle", None)
    with handle.lock() if handle is not None else contextlib.nullcontext():
      set_payload(self.backend, payload[0], payload[1], self.sampling.payload_mass_max)

  def _hold(self, seconds: float) -> None:
    """Keep the last target of the controller that ran last (zero velocity / acceleration targets)."""
    ctrl = self._active
    if ctrl is self.osc:
      wp.copy(ctrl.target_vel, wp.zeros_like(ctrl.target_vel))
      wp.copy(ctrl.target_acc, wp.zeros_like(ctrl.target_acc))
    else:
      ctrl.dq_des.zero_()
      ctrl.ddq_des.zero_()
    self._run(ctrl, HOLD, int(round(seconds / self.dt)), None)

  def _run(self, ctrl, skill: int, steps: int, set_target) -> None:
    """`steps` physics steps under `ctrl`; `set_target(i)` updates its targets before step i. Records each step."""
    backend, d = self.backend, self.backend.d
    self._active = ctrl
    for i in range(steps):
      if set_target is not None:
        set_target(i)
      backend.step1()
      state = backend.robot_state(ctrl.requires or ALL_NEEDS)
      backend.set_arm_ctrl(ctrl.compute(state))
      backend.step2()
      # q, dq in `state` are still the values at the start of the step; the forces and qacc were computed there.
      wp.launch(_record_kernel, dim=self.view.ndof,
                inputs=[self._dof_adr, state.q, state.dq, d.qacc, d.qfrc_actuator, d.qfrc_passive, d.qfrc_constraint,
                        self._armature],
                outputs=[self._sample], device=backend.device)
      wp.launch(_contact_kernel, dim=1, inputs=[d.nacon], outputs=[self._contacts], device=backend.device)
      self._rows.append(self._sample.numpy())
      self._skill_col.append(skill)
      if self.viewer is not None and len(self._rows) % self.view_stride == 0:
        if not self.viewer.running():
          raise KeyboardInterrupt("viewer closed")
        self.viewer.sync(self.view_stride * self.dt)

  def _q(self) -> np.ndarray:
    """Current arm joint positions (host sync)."""
    return self.backend.d.qpos.numpy()[0, self.view.qpos_adr].astype(np.float64)

  def _recorded_q(self, first: int) -> np.ndarray:
    return np.asarray([row[0] for row in self._rows[first:]], np.float64)

  def _accept(self, rows: np.ndarray) -> None:
    """Whole-trajectory checks on what was actually simulated."""
    if not np.isfinite(rows).all():
      raise Rejected("non-finite samples")
    saturated = (np.abs(rows[:, 2]) >= 0.999 * self.view.torque_limits).any(axis=1).mean()
    if saturated > 0:
      raise Rejected(f"torque saturated in {saturated:.1%} of samples")
    if self._contacts.numpy()[0] > 0:
      raise Rejected("contact in the simulation")
    if not all(self.checker.in_limits(q, 0.5) for q in rows[:, 0]):
      raise Rejected("simulated motion: joint_limit")
