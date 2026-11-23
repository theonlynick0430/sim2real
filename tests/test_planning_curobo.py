"""cuRobo-backed IK / motion planning on YAM. Needs `pip install -e ".[planning]"` and a CUDA GPU (Linux);
skipped otherwise. Targets come from host FK (MuJoCo on the exported URDF), so they are reachable, and
every solver output is re-checked with that same independent FK."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")
if not torch.cuda.is_available():
  pytest.skip("cuRobo needs a CUDA GPU", allow_module_level=True)

from sim2real.planning import GpuIK, GpuMotionPlanner, MujocoFK, rotation_error  # noqa: E402
from sim2real.planning.adapters import planner_config_from_robot  # noqa: E402
from sim2real.robots import get_robot  # noqa: E402

NIK, NBATCH = 256, 16


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
  return planner_config_from_robot(get_robot("yam"), tmp_path_factory.mktemp("planning"))


@pytest.fixture(scope="module")
def fk(cfg):
  return MujocoFK(cfg.urdf_path, cfg.joint_names, cfg.tool_frame)


@pytest.fixture(scope="module")
def ik(cfg):
  return GpuIK(cfg.replace(max_batch_size=NIK))


@pytest.fixture(scope="module")
def planner(cfg):
  mp = GpuMotionPlanner(cfg)
  mp.warmup()
  return mp


@pytest.fixture(scope="module")
def batch_planner(cfg):
  mp = GpuMotionPlanner(cfg.replace(max_batch_size=NBATCH), batch=True)
  mp.warmup()
  return mp


def _targets(fk, n, seed):
  q = fk.sample_q(n, np.random.default_rng(seed), margin=0.1)
  pos, quat = fk(q)
  return q, pos, quat


def _assert_reaches(fk, cfg, q, pos, quat, slack=1.5):
  p, r = fk(q)
  assert np.linalg.norm(p - pos, axis=-1).max() < slack * cfg.position_tolerance
  assert rotation_error(r, quat).max() < slack * cfg.orientation_tolerance


def _assert_valid_traj(fk, cfg, traj, q_start, check_end):
  """Starts at q_start, stays in limits, respects velocity limits, no jumps, ends where `check_end` says."""
  t = traj.numpy()
  assert t.success.all(), f"planning failed for {np.flatnonzero(~t.success)}"
  vmax = np.asarray(cfg.max_vel)
  for b in range(len(t.q)):
    q = t.q[b, :t.length[b]]
    assert len(q) >= 2
    np.testing.assert_allclose(q[0], q_start[b], atol=1e-3)
    assert (q >= fk.lower - 1e-3).all() and (q <= fk.upper + 1e-3).all()
    assert (np.abs(t.dq[b, :t.length[b]]) <= 1.05 * vmax).all()
    assert (np.abs(np.diff(q, axis=0)) <= 1.05 * vmax * t.dt + 1e-4).all()
    check_end(b, q[-1])


def test_ik_batch_reaches_targets(ik, fk, cfg):
  _, pos, quat = _targets(fk, NIK, seed=0)
  res = ik.solve(pos, quat).numpy()
  assert res.q.shape == (NIK, cfg.dof)
  assert res.success.mean() >= 0.95, f"IK success {res.success.mean():.3f}"
  ok = res.success
  _assert_reaches(fk, cfg, res.q[ok], pos[ok], quat[ok])
  assert (res.q[ok] >= fk.lower - 1e-4).all() and (res.q[ok] <= fk.upper + 1e-4).all()
  assert (res.pos_err[ok] <= cfg.position_tolerance).all()


def test_ik_current_q_prefers_nearby_solution(ik, fk, cfg):
  q_true, pos, quat = _targets(fk, 32, seed=1)
  res = ik.solve(pos, quat, seed_q=q_true, current_q=q_true).numpy()
  assert res.success.all()
  _assert_reaches(fk, cfg, res.q, pos, quat)
  assert np.median(np.abs(res.q - q_true).max(-1)) < 0.05  # converges to the seeded branch


def test_ik_accepts_unbatched_input(ik, fk, cfg):
  _, pos, quat = _targets(fk, 1, seed=2)
  res = ik.solve(pos[0], quat[0]).numpy()
  assert res.q.shape == (1, cfg.dof) and res.success.all()


def test_plan_joint(planner, fk, cfg):
  q0, _, _ = _targets(fk, 3, seed=3)
  qg, _, _ = _targets(fk, 3, seed=4)
  traj = planner.plan_joint(q0, qg)
  _assert_valid_traj(fk, cfg, traj, q0, lambda b, q_end: np.testing.assert_allclose(q_end, qg[b], atol=0.02))


def test_plan_pose(planner, fk, cfg):
  q0, _, _ = _targets(fk, 3, seed=5)
  _, pos, quat = _targets(fk, 3, seed=6)
  traj = planner.plan_pose(q0, pos, quat)
  _assert_valid_traj(fk, cfg, traj, q0, lambda b, q_end: _assert_reaches(fk, cfg, q_end[None], pos[b:b + 1],
                                                                          quat[b:b + 1]))


def test_batch_plan_pose(batch_planner, fk, cfg):
  q0, _, _ = _targets(fk, NBATCH, seed=7)
  _, pos, quat = _targets(fk, NBATCH, seed=8)
  traj = batch_planner.plan_pose(q0, pos, quat)
  assert traj.q.shape[0] == NBATCH
  _assert_valid_traj(fk, cfg, traj, q0, lambda b, q_end: _assert_reaches(fk, cfg, q_end[None], pos[b:b + 1],
                                                                          quat[b:b + 1]))


def test_batch_plan_joint_partial_batch(batch_planner, fk, cfg):
  """Fewer problems than max_batch_size are padded internally and trimmed from the output."""
  q0, _, _ = _targets(fk, 5, seed=9)
  qg, _, _ = _targets(fk, 5, seed=10)
  traj = batch_planner.plan_joint(q0, qg)
  assert traj.q.shape[0] == 5
  _assert_valid_traj(fk, cfg, traj, q0, lambda b, q_end: np.testing.assert_allclose(q_end, qg[b], atol=0.02))
