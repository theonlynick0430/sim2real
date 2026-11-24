"""Sampling of start configurations, skill goals and payloads, plus the host-side validity checks.

Everything here is numpy + MuJoCo C on the scene model: no GPU. The planner has collision checking off, so
`Checker` is what keeps plans free of contacts (arm with itself or the table), off the joint limits and away from
singularities. After the rollout, contacts are checked on the simulation itself (`skills.py`).
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
  """Every knob that changes which trajectories are generated (compared on `--resume`)."""

  joint_weight: float = 1.0  # relative probability of a joint move
  ee_weight: float = 1.0  # relative probability of an ee move
  min_skills: int = 3
  max_skills: int = 6
  limit_margin: float = 0.1  # fraction of each joint range kept free at both limits
  min_sigma: float = 0.02  # smallest singular value of the TCP Jacobian at starts / joint goals / along ee moves
  joint_candidates: int = 16  # valid candidates per joint goal; the farthest one is used
  ee_pos_range: tuple[float, float] = (0.02, 0.12)  # m, length of the position offset of an ee goal
  ee_rot_range: tuple[float, float] = (0.087, 0.52)  # rad (5..30 deg), rotation of an ee goal
  ee_max_joint_step: float = 1.0  # rad: an ee plan moving any joint further than this changed IK branch
  time_scale_range: tuple[float, float] = (1.0, 3.0)  # per-skill slow-down of the plan
  hold_range: tuple[float, float] = (0.2, 0.5)  # s, pause before each skill
  payload_mass_max: float = 2.0  # kg (rated payload of the base YAM)
  payload_com_range: tuple[float, float, float] = (0.03, 0.03, 0.05)  # m, +- box in the tool frame
  empty_prob: float = 0.25  # probability that a sampled payload is "nothing"
  payload_change_prob: float = 0.4  # probability that the payload changes before a skill (after the first)

  def __post_init__(self):
    if self.joint_weight < 0 or self.ee_weight < 0 or self.joint_weight + self.ee_weight <= 0:
      raise ValueError("skill weights must be >= 0 and not both zero")
    if not 1 <= self.min_skills <= self.max_skills:
      raise ValueError("need 1 <= min_skills <= max_skills")


def halton(index: int, dim: int, shift: np.ndarray) -> np.ndarray:
  """Point `index` of the `dim`-dimensional Halton sequence in [0, 1)^dim, rotated by `shift` (mod 1)."""
  point = np.empty(dim)
  for d in range(dim):
    base, i, f, x = _PRIMES[d], index + 1, 1.0, 0.0
    while i > 0:
      f /= base
      x += f * (i % base)
      i //= base
    point[d] = x
  return (point + shift) % 1.0


class Checker:
  """Validity of arm configurations on the scene model: joint margins, contacts, TCP Jacobian conditioning."""

  def __init__(self, mjm: mujoco.MjModel, robot, cfg: SamplingConfig):
    self.cfg, self.m = cfg, mjm
    self.d = mujoco.MjData(self.m)
    joints = [self.m.joint(j).id for j in robot.arm_joints]
    self.qpos_adr, self.dof_adr = self.m.jnt_qposadr[joints], self.m.jnt_dofadr[joints]
    self.site = self.m.site(robot.tcp_site).id
    lower, upper = self.m.jnt_range[joints].T
    span = upper - lower
    self.lower, self.upper, self.span = lower + cfg.limit_margin * span, upper - cfg.limit_margin * span, span
    self.hard_lower, self.hard_upper = lower.copy(), upper.copy()

  def _forward(self, q) -> None:
    self.d.qpos[:] = self.m.qpos0
    self.d.qpos[self.qpos_adr] = q
    mujoco.mj_forward(self.m, self.d)

  def in_limits(self, q, margin_scale: float = 1.0) -> bool:
    """All joints at least `margin_scale * limit_margin` (fraction of range) inside their limits."""
    q, pad = np.asarray(q), self.cfg.limit_margin * margin_scale * self.span
    return bool(np.all(q >= self.hard_lower + pad) and np.all(q <= self.hard_upper - pad))

  def clear(self, q) -> bool:
    """MuJoCo finds no contact (arm with itself or the table) at `q`, same collision model as the sim."""
    self._forward(q)
    return self.d.ncon == 0

  def min_sigma(self, q) -> float:
    """Smallest singular value of the 6 x n TCP Jacobian [linear; angular]."""
    self._forward(q)
    jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
    mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.site)
    return float(np.linalg.svd(np.vstack([jacp, jacr])[:, self.dof_adr], compute_uv=False)[-1])

  def endpoint_ok(self, q) -> bool:
    """Usable as a start or joint goal: clear, and well enough conditioned that an ee move can start there."""
    return self.clear(q) and self.min_sigma(q) >= self.cfg.min_sigma

  def path_problem(self, q_path, stride: int = 1, sigma: bool = False) -> str | None:
    """First reason a joint path (T, n) is invalid, or None. Checked every `stride` samples and at the end."""
    q_path = np.asarray(q_path)
    idx = np.unique(np.append(np.arange(0, len(q_path), stride), len(q_path) - 1))
    for q in q_path[idx]:
      if not self.in_limits(q):
        return "joint_limit"
      if not self.clear(q):
        return "contact"
      if sigma and self.min_sigma(q) < self.cfg.min_sigma:
        return "singular"
    return None

  def sample(self, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(self.lower, self.upper)

  def distance(self, q_a, q_b) -> float:
    """Mean joint distance as a fraction of each joint's range."""
    return float(np.mean(np.abs(np.asarray(q_a) - np.asarray(q_b)) / self.span))


def sample_start(checker: Checker, index: int, attempt: int, seed: int, tries: int = 2000) -> np.ndarray:
  """Start configuration of trajectory `index`: the first valid point of a Halton sequence over the joint box
  (rotated by a seed-dependent shift), so the starts of a dataset cover the workspace evenly. Each (index,
  attempt) reads its own stretch of the sequence."""
  n = len(checker.span)
  shift = np.random.default_rng([seed, 0xC0FFEE]).uniform(size=n)
  base = (index * 7919 + attempt * 104729) % 2**31
  for k in range(tries):
    q = checker.lower + halton(base + k * 15485863 % 2**31, n, shift) * (checker.upper - checker.lower)
    if checker.endpoint_ok(q):
      return q
  raise RuntimeError("no valid start configuration found")


def sample_joint_goal(checker: Checker, rng: np.random.Generator, q_now, visited, tries: int = 4000) -> np.ndarray:
  """Joint goal for a large move: among `joint_candidates` valid samples, the one farthest from the current
  configuration and from the goals already used in this trajectory (best-candidate / maximin sampling)."""
  avoid = [np.asarray(q_now), *visited]
  best, best_score, found = None, -1.0, 0
  for _ in range(tries):
    q = checker.sample(rng)
    if not checker.endpoint_ok(q):
      continue
    found += 1
    score = min(checker.distance(q, other) for other in avoid)
    if score > best_score:
      best, best_score = q, score
    if found >= checker.cfg.joint_candidates:
      return best
  if best is None:
    raise RuntimeError("no valid joint goal found")
  return best


def sample_ee_goal(cfg: SamplingConfig, rng: np.random.Generator, pos, quat_wxyz) -> tuple[np.ndarray, np.ndarray]:
  """Nearby TCP goal: the current pose moved along a random direction (all three axes) by a length in
  `ee_pos_range`, and rotated about a random axis by an angle in `ee_rot_range`."""
  direction = rng.normal(size=3)
  direction /= np.linalg.norm(direction)
  axis = rng.normal(size=3)
  axis /= np.linalg.norm(axis)
  spin, quat = np.zeros(4), np.zeros(4)
  mujoco.mju_axisAngle2Quat(spin, axis, rng.uniform(*cfg.ee_rot_range))
  mujoco.mju_mulQuat(quat, spin, np.asarray(quat_wxyz, np.float64))
  return np.asarray(pos) + direction * rng.uniform(*cfg.ee_pos_range), quat


def sample_payload(cfg: SamplingConfig, rng: np.random.Generator) -> tuple[float, np.ndarray]:
  """(mass [kg], center of mass [m] in the tool frame). Mass 0 means no payload."""
  if rng.uniform() < cfg.empty_prob:
    return 0.0, np.zeros(3)
  box = np.asarray(cfg.payload_com_range)
  return float(rng.uniform(0.05 * cfg.payload_mass_max, cfg.payload_mass_max)), rng.uniform(-box, box)
