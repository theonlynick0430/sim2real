"""GPU motion planning (cuRobo v2 `MotionPlanner` / `BatchMotionPlanner`), collision checking disabled.

`plan_pose` solves IK internally (several goal configurations, best trajectory wins); `plan_joint`
plans to a given joint configuration. Both return time-parameterized joint trajectories sampled at
`cfg.interpolation_dt`, respecting the URDF velocity limits and `cfg.max_acc` / `cfg.max_jerk`.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .config import PlannerConfig


@dataclasses.dataclass
class Trajectory:
  q: "torch.Tensor | np.ndarray"  # (B, T, dof); steps past length[b] hold the final configuration  # noqa: F821
  dq: "torch.Tensor | np.ndarray"  # (B, T, dof); zero past length[b]
  ddq: "torch.Tensor | np.ndarray"  # (B, T, dof); zero past length[b]
  dt: float  # s between samples
  length: "torch.Tensor | np.ndarray"  # (B,) valid samples per problem
  success: "torch.Tensor | np.ndarray"  # (B,) bool
  solve_time: float = 0.0  # s, wall clock for the planning call (excludes the end-effector FK below)
  # Planned end-effector (tool frame) trajectory, base frame; only filled with `return_ee=True`.
  ee_pos: "torch.Tensor | np.ndarray | None" = None  # (B, T, 3)  # noqa: F821
  ee_quat: "torch.Tensor | np.ndarray | None" = None  # (B, T, 4) wxyz  # noqa: F821
  ee_vel: "torch.Tensor | np.ndarray | None" = None  # (B, T, 6) [linear, angular]; zero past length[b]  # noqa: F821

  def numpy(self) -> "Trajectory":
    f = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)  # noqa: E731
    g = lambda x: None if x is None else f(x)  # noqa: E731
    return Trajectory(f(self.q), f(self.dq), f(self.ddq), self.dt, f(self.length).astype(int),
                      f(self.success).astype(bool), self.solve_time, g(self.ee_pos), g(self.ee_quat), g(self.ee_vel))


class GpuMotionPlanner:
  """batch=False: `MotionPlanner`, problems solved one after another (full retry/graph fallback).
  batch=True: `BatchMotionPlanner`, up to `cfg.max_batch_size` problems solved together on the GPU.

  `plan_kwargs` are forwarded to cuRobo's plan_pose / plan_cspace (e.g. max_attempts).
  """

  def __init__(self, cfg: PlannerConfig, batch: bool = False, **plan_kwargs):
    from curobo.batch_motion_planner import BatchMotionPlanner
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    from . import _curobo

    self.cfg, self.batch, self.plan_kwargs = cfg, batch, plan_kwargs
    self._fk = None  # host FK for return_ee, built on first use
    self._c = _curobo
    mp_cfg = MotionPlannerCfg.create(
      robot=_curobo.robot_cfg(cfg),
      scene_model=None,
      self_collision_check=False,
      device_cfg=_curobo.device_cfg(cfg),
      num_ik_seeds=cfg.num_ik_seeds,
      position_tolerance=cfg.position_tolerance,
      orientation_tolerance=cfg.orientation_tolerance,
      max_batch_size=cfg.max_batch_size if batch else 1,
      interpolation_dt=cfg.interpolation_dt,
      use_cuda_graph=True,
      enable_inverse_dynamics=False,  # no torque constraints (needs cuRobo main after v0.9.0)
    )
    self.planner = BatchMotionPlanner(mp_cfg) if batch else MotionPlanner(mp_cfg)
    if list(self.planner.joint_names) != list(cfg.joint_names):
      raise ValueError(f"cuRobo joint order {self.planner.joint_names} != {cfg.joint_names}")

  def warmup(self, num_iterations: int = 5) -> None:
    """Capture CUDA graphs up front so the first real plan call is fast."""
    self.planner.warmup(enable_graph=True, num_warmup_iterations=num_iterations)

  def plan_pose(self, q_start, pos, quat_wxyz, return_ee: bool = False) -> Trajectory:
    """q_start (B,dof); target tool pose pos (B,3), quat_wxyz (B,4) in the base frame.
    return_ee: also fill `ee_pos` / `ee_quat` / `ee_vel` (planned tool-frame trajectory, base frame)."""
    c, cfg = self._c, self.cfg
    q0, pos, quat = c.as_tensor(q_start, cfg, 2), c.as_tensor(pos, cfg, 2), c.as_tensor(quat_wxyz, cfg, 2)
    traj = self._run(q0, lambda q, i: self.planner.plan_pose(
      c.goal_pose(cfg, pos[i], quat[i]), c.joint_state(q, cfg.joint_names), **self.plan_kwargs))
    return self._add_ee(traj) if return_ee else traj

  def plan_joint(self, q_start, q_goal, return_ee: bool = False) -> Trajectory:
    """q_start (B,dof) -> q_goal (B,dof). return_ee as in `plan_pose`."""
    c, cfg = self._c, self.cfg
    q0, qg = c.as_tensor(q_start, cfg, 2), c.as_tensor(q_goal, cfg, 2)
    traj = self._run(q0, lambda q, i: self.planner.plan_cspace(
      c.joint_state(qg[i], cfg.joint_names), c.joint_state(q, cfg.joint_names), **self.plan_kwargs))
    return self._add_ee(traj) if return_ee else traj

  # -- internals --

  def _add_ee(self, traj: Trajectory) -> Trajectory:
    """Fill the end-effector fields by host FK (MuJoCo on the URDF cuRobo plans with), not cuRobo's kinematics,
    so they match `MujocoFK` exactly. Outputs are torch on the planner device, like the joint fields."""
    import torch

    from .fk import MujocoFK

    if self._fk is None:
      self._fk = MujocoFK(self.cfg.urdf_path, self.cfg.joint_names, self.cfg.tool_frame)
    t = traj.numpy()
    pos, quat, vel = self._fk.ee_trajectory(t.q, t.dq)  # dq is zero past length[b], so vel is too
    dev = traj.q.device
    as_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    return dataclasses.replace(traj, ee_pos=as_t(pos), ee_quat=as_t(quat), ee_vel=as_t(vel))

  def _run(self, q0, plan_one) -> Trajectory:
    """plan_one(q_start (n,dof), index) -> cuRobo result for those problems."""
    import time

    import torch

    tic = time.perf_counter()
    batch = q0.shape[0]
    if self.batch:
      idx = self._c.pad_batch(torch.arange(batch, device=q0.device), self.cfg.max_batch_size)
      parts = [self._extract(plan_one(q0[idx], idx), q0[idx])]
    else:
      parts = [self._extract(plan_one(q0[i:i + 1], slice(i, i + 1)), q0[i:i + 1]) for i in range(batch)]
    traj = self._concat(parts, batch)
    if q0.device.type == "cuda":
      torch.cuda.synchronize(q0.device)
    traj.solve_time = time.perf_counter() - tic
    return traj

  def _extract(self, res, q0) -> Trajectory:
    """cuRobo result for n problems -> Trajectory (n, T, dof). A None result means all failed."""
    import torch

    n, dof = q0.shape
    if res is None or res.interpolated_trajectory is None:
      zeros = torch.zeros_like(q0).unsqueeze(1)
      return Trajectory(q0.unsqueeze(1).clone(), zeros, zeros.clone(), self.cfg.interpolation_dt,
                        torch.ones(n, dtype=torch.long, device=q0.device),
                        torch.zeros(n, dtype=torch.bool, device=q0.device))
    js = res.interpolated_trajectory

    # cuRobo reuses its result buffers across plan calls, so everything kept here must be a copy (clone):
    # a view would be overwritten by the next call (the non-batch path makes one call per problem).
    def per_problem(x):  # [n, seeds, T, dof] | [n, T, dof] | [T, dof] -> [n, T, dof]
      x = x.reshape(1, *x.shape) if x.ndim == 2 else x
      return (x[:, 0] if x.ndim == 4 else x)[:n].float().clone()

    q = per_problem(js.position)
    dq = per_problem(js.velocity) if js.velocity is not None else torch.zeros_like(q)
    ddq = per_problem(js.acceleration) if js.acceleration is not None else torch.zeros_like(q)
    T = q.shape[1]
    if res.interpolated_last_tstep is not None:
      length = self._c.first_per_problem(torch.as_tensor(res.interpolated_last_tstep, device=q.device), n)
      length = length.long().clamp(1, T)  # clamp copies
    else:
      length = torch.full((n,), T, dtype=torch.long, device=q.device)
    dt = float(js.dt.reshape(-1)[0]) if js.dt is not None else self.cfg.interpolation_dt
    success = self._c.first_per_problem(res.success, n).bool().to(q.device).clone()
    return Trajectory(q, dq, ddq, dt, length, success)

  @staticmethod
  def _concat(parts: list[Trajectory], batch: int) -> Trajectory:
    """Stack per-call trajectories, padding to a common T (hold final q, zero dq/ddq) past each length."""
    import torch

    T = int(torch.cat([p.length for p in parts])[:batch].max())  # ignore batch-padding rows
    qs, dqs, ddqs = [], [], []
    for p in parts:
      steps = torch.arange(T, device=p.q.device)
      valid = steps[None, :] < p.length[:, None]  # (n, T)
      src = torch.minimum(steps[None, :], p.length[:, None] - 1).clamp(max=p.q.shape[1] - 1)
      gather = lambda x: torch.gather(x, 1, src[..., None].expand(-1, -1, x.shape[-1]))  # noqa: E731
      qs.append(gather(p.q))
      dqs.append(gather(p.dq) * valid[..., None])
      ddqs.append(gather(p.ddq) * valid[..., None])
    return Trajectory(
      q=torch.cat(qs)[:batch], dq=torch.cat(dqs)[:batch], ddq=torch.cat(ddqs)[:batch], dt=parts[0].dt,
      length=torch.cat([p.length for p in parts])[:batch], success=torch.cat([p.success for p in parts])[:batch])
