"""Motor reference profiles (sim2real/profiles): resampling, payload scene, recorded torques, sampling, session
files and resume run anywhere; the planner-backed end-to-end tests need cuRobo and a CUDA GPU."""

import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim2real.profiles import sampling
from sim2real.profiles.scene import PAYLOAD_BODY, PAYLOAD_GEOM, build_profile_scene, set_payload
from sim2real.profiles.session import Session, load_trajectory, traj_path
from sim2real.profiles.skills import HOLD, ProfileGenerator
from sim2real.robots import get_robot
from sim2real.sim.mjwarp_backend import MjWarpBackend
from sim2real.utils import resample_trajectory

ROBOT = get_robot("yam")


def _backend(device):
  return MjWarpBackend(build_profile_scene(ROBOT), nworld=1, device=device, use_graph=False)


# --- resampling ---------------------------------------------------------------------------------
def test_resample_reproduces_samples_and_cubics():
  t = np.arange(11)[:, None] * 0.02
  coef = np.array([[0.3, -1.0, 2.0, 5.0], [-0.2, 0.5, -3.0, 1.0]]).T  # two joints, cubic in t
  q = coef[0] + coef[1] * t + coef[2] * t**2 + coef[3] * t**3
  dq = coef[1] + 2 * coef[2] * t + 3 * coef[3] * t**2
  ddq = 2 * coef[2] + 6 * coef[3] * t
  same = resample_trajectory(q, dq, ddq, 0.02, 0.02)
  for got, want in zip(same, (q, dq, ddq)):
    np.testing.assert_allclose(got, want, atol=1e-12)
  fine = resample_trajectory(q, dq, ddq, 0.02, 0.002)
  tf = np.arange(101)[:, None] * 0.002
  assert len(fine[0]) == 101
  np.testing.assert_allclose(fine[0], coef[0] + coef[1] * tf + coef[2] * tf**2 + coef[3] * tf**3, atol=1e-12)
  np.testing.assert_allclose(fine[1], coef[1] + 2 * coef[2] * tf + 3 * coef[3] * tf**2, atol=1e-10)
  np.testing.assert_allclose(fine[2], 2 * coef[2] + 6 * coef[3] * tf, atol=1e-10)


def test_resample_time_scale_and_end_point():
  rng = np.random.default_rng(0)
  q, dq, ddq = rng.normal(size=(3, 6, 4))
  slow = resample_trajectory(q, dq, ddq, 0.02, 0.002, time_scale=2.0)
  assert len(slow[0]) == 101  # 0.2 s instead of 0.1 s
  np.testing.assert_allclose(slow[0][::20], q, atol=1e-12)  # same path
  np.testing.assert_allclose(slow[1][::20], dq / 2, atol=1e-12)
  np.testing.assert_allclose(slow[2][::20], ddq / 4, atol=1e-12)
  odd = resample_trajectory(q, dq, ddq, 0.02, 0.003, time_scale=1.3)  # duration not a multiple of dt_out
  np.testing.assert_allclose(odd[0][-1], q[-1], atol=1e-12)


# --- scene / payload ----------------------------------------------------------------------------
def test_motors_are_ideal():
  mjm = build_profile_scene(ROBOT).mjm
  dofs = [mjm.joint(j).dofadr[0] for j in ROBOT.arm_joints]
  assert np.all(mjm.dof_frictionloss[dofs] == 0.0) and np.all(mjm.dof_damping[dofs] == 0.0)
  np.testing.assert_allclose(mjm.dof_armature[dofs], [0.032] * 3 + [0.0018] * 3)  # reflected rotor inertia


def test_set_payload(warp_device):
  backend = _backend(warp_device)
  mjm = backend.mjm
  body, site = mjm.body(PAYLOAD_BODY).id, mjm.site(ROBOT.tcp_site).id
  total = mjm.body_subtreemass[1]
  com = np.array([0.02, -0.03, 0.05])
  set_payload(backend, 0.6, com, max_mass=1.0)
  assert mjm.body_mass[body] == pytest.approx(0.6)
  assert mjm.body_subtreemass[1] == pytest.approx(total + 0.6)
  assert mjm.geom_rgba[mjm.geom(PAYLOAD_GEOM).id, 3] == pytest.approx(0.6)
  np.testing.assert_allclose(backend.m.body_mass.numpy()[0, body], 0.6, rtol=1e-6)  # uploaded to MJWarp

  d = mujoco.MjData(mjm)
  d.qpos[:6] = [0.4, 1.2, 0.9, -0.3, 0.5, 1.0]
  mujoco.mj_forward(mjm, d)
  want = d.site_xpos[site] + d.site_xmat[site].reshape(3, 3) @ com  # COM given in the tool frame
  np.testing.assert_allclose(d.xipos[body], want, atol=1e-12)
  np.testing.assert_allclose(d.geom_xpos[mjm.geom(PAYLOAD_GEOM).id], want, atol=1e-12)  # sphere is at the COM

  set_payload(backend, 0.0, np.zeros(3), max_mass=1.0)
  assert mjm.body_subtreemass[1] == pytest.approx(total)
  assert mjm.geom_rgba[mjm.geom(PAYLOAD_GEOM).id, 3] == 0.0


# --- recorded torques ---------------------------------------------------------------------------
def test_recorded_torques_match_mujoco(warp_device):
  """Roll a joint sinusoid with a payload and compare every recorded sample with MuJoCo C at that state."""
  backend = _backend(warp_device)
  gen = ProfileGenerator(backend, planner=None, fk=None)
  q0 = np.array([0.3, 1.2, 1.3, 0.2, -0.4, 0.5])
  gen._rows, gen._skill_col = [], []
  gen._reset(q0, (0.7, np.array([0.02, 0.01, 0.04])))
  amp, w, dt = np.array([0.3, 0.2, 0.2, 0.4, 0.4, 0.5]), 4.0, gen.dt
  gen._run(gen.joint_ctrl, 0, 300, lambda i: gen.joint_ctrl.set_target(
    q0 + amp * np.sin(w * i * dt), amp * w * np.cos(w * i * dt), -amp * w * w * np.sin(w * i * dt)))
  rows = np.asarray(gen._rows, np.float64)
  assert rows.shape == (300, 4, 6) and np.isfinite(rows).all()
  assert np.abs(rows[:, 1]).max() > 0.5  # it moved

  mjm = backend.mjm  # holds the payload set above
  d, arm = mujoco.MjData(mjm), slice(0, 6)
  full = np.zeros((mjm.nv, mjm.nv))
  for q, dq, tau_motor, tau_ext in rows[::7]:
    d.qpos[:], d.qvel[:], d.ctrl[:] = mjm.qpos0, 0.0, 0.0
    d.qpos[arm], d.qvel[arm], d.ctrl[arm] = q, dq, tau_motor
    mujoco.mj_forward(mjm, d)
    np.testing.assert_allclose(d.qfrc_actuator[arm], tau_motor, atol=1e-5)
    mujoco.mj_fullM(mjm, d, full)
    # Load on the motor output = minus the rigid-body torque without the rotor: -(M_links qacc + bias).
    load = -(full @ d.qacc - mjm.dof_armature * d.qacc + d.qfrc_bias)[arm]
    np.testing.assert_allclose(tau_ext, load, atol=0.02)
    # Motor balance of an ideal motor: armature * qacc = tau_motor + tau_ext.
    np.testing.assert_allclose(mjm.dof_armature[arm] * d.qacc[arm] - tau_motor - tau_ext, 0.0, atol=0.02)


# --- sampling -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def checker():
  return sampling.Checker(build_profile_scene(ROBOT).mjm, ROBOT, sampling.SamplingConfig())


def test_checker(checker):
  assert checker.in_limits(ROBOT.home_qpos) and checker.clear(ROBOT.home_qpos)
  assert not checker.in_limits(checker.hard_lower + 0.01 * checker.span)
  rng = np.random.default_rng(0)
  samples = [checker.sample(rng) for _ in range(200)]
  blocked = [q for q in samples if not checker.clear(q)]
  assert 0.1 < len(blocked) / len(samples) < 0.6  # some in-limit configurations hit the arm or the table
  assert checker.in_limits(blocked[0]) and checker.path_problem([ROBOT.home_qpos, blocked[0]]) == "contact"
  wrist_hit = [-1.08, 0.89, 1.07, -0.18, -1.17, -1.69]  # link_3 touches link_5 (two joints apart)
  assert checker.in_limits(wrist_hit, 0.5) and not checker.clear(wrist_hit)
  under = [0.0, 3.4, 1.0, 0.0, 0.0, 0.0]  # tool 0.3 m below the table
  assert not checker.clear(under)
  assert checker.path_problem([ROBOT.home_qpos] * 3, sigma=True) is None


def test_starts_and_goals(checker):
  cfg = checker.cfg
  starts = np.array([sampling.sample_start(checker, i, 0, seed=0) for i in range(64)])
  assert all(checker.in_limits(q) and checker.endpoint_ok(q) for q in starts)
  np.testing.assert_array_equal(starts[5], sampling.sample_start(checker, 5, 0, seed=0))  # reproducible
  assert not np.allclose(starts[5], sampling.sample_start(checker, 5, 1, seed=0))  # attempts differ
  spread = (starts.max(0) - starts.min(0)) / (checker.upper - checker.lower)
  assert spread.min() > 0.8  # covers each joint's allowed range

  rng = np.random.default_rng(0)
  goal = sampling.sample_joint_goal(checker, rng, starts[0], [])
  assert checker.in_limits(goal) and checker.endpoint_ok(goal)
  assert checker.distance(goal, starts[0]) > 0.2  # a large move

  pos, quat = np.array([0.3, 0.0, 0.2]), np.array([0.0, 0.0, 1.0, 0.0])
  moved, turned = [], []
  for _ in range(50):
    p, q = sampling.sample_ee_goal(cfg, rng, pos, quat)
    diff = np.zeros(3)
    mujoco.mju_subQuat(diff, q, quat)
    moved.append(p - pos)
    turned.append(diff)
  moved, turned = np.array(moved), np.array(turned)
  dist, angle = np.linalg.norm(moved, axis=1), np.linalg.norm(turned, axis=1)
  assert cfg.ee_pos_range[0] - 1e-9 <= dist.min() and dist.max() <= cfg.ee_pos_range[1] + 1e-9
  assert cfg.ee_rot_range[0] - 1e-6 <= angle.min() and angle.max() <= cfg.ee_rot_range[1] + 1e-6
  assert np.all(moved.std(0) > 0.01) and np.all(turned.std(0) > 0.05)  # every axis varies

  payloads = [sampling.sample_payload(cfg, rng) for _ in range(200)]
  mass = np.array([m for m, _ in payloads])
  com = np.array([c for _, c in payloads])
  assert 0.1 < np.mean(mass == 0.0) < 0.45 and mass.max() <= cfg.payload_mass_max
  assert np.all(np.abs(com) <= np.asarray(cfg.payload_com_range))
  with pytest.raises(ValueError):
    sampling.SamplingConfig(joint_weight=0.0, ee_weight=0.0)


# --- session / resume ---------------------------------------------------------------------------
def _fake(index, n=20):
  rng = np.random.default_rng(index)
  arrays = {k: rng.normal(size=(n, 6)).astype(np.float32) for k in ("q", "dq", "tau_motor", "tau_ext")}
  arrays.update(t=np.arange(n, dtype=np.float32) * 0.002, skill=np.zeros(n, np.int16))
  meta = {"index": index, "attempt": 0, "dt": 0.002, "num_samples": n,
          "skills": [{"type": "joint_move", "payload_mass": 0.5}]}
  return arrays, meta


def test_session_roundtrip_and_resume(tmp_path):
  config = {"seed": 0, "robot": "yam", "sampling": {"ee_pos_range": (0.02, 0.12)}}
  session = Session(tmp_path, config)
  for i in range(3):
    session.add(i, *_fake(i), rejects=[])
  session.fail(3, ["attempt 0: no plan"])
  session.close("failed", RuntimeError("boom"))

  arrays, meta = load_trajectory(traj_path(tmp_path, 1))
  np.testing.assert_array_equal(arrays["tau_ext"], _fake(1)[0]["tau_ext"])
  assert meta["skills"][0]["type"] == "joint_move" and "friction" not in arrays
  data = json.loads((tmp_path / "session.json").read_text())
  assert data["status"] == "failed" and data["num_ok"] == 3 and data["num_rejected"] == 1
  assert "RuntimeError: boom" in data["runs"][0]["error"]
  assert not list(tmp_path.glob("*.tmp"))

  with pytest.raises(FileExistsError):
    Session(tmp_path, config)  # refuses to overwrite without resume
  with pytest.raises(ValueError, match="seed"):
    Session(tmp_path, {**config, "seed": 1}, resume=True)
  traj_path(tmp_path, 0).unlink()  # lost file
  traj_path(tmp_path, 2).write_bytes(b"truncated")  # corrupt file
  resumed = Session(tmp_path, config, resume=True)
  assert [resumed.done(i) for i in range(4)] == [False, True, False, False]
  resumed.add(0, *_fake(0), rejects=[])
  resumed.close("complete")
  data = json.loads((tmp_path / "session.json").read_text())
  assert data["status"] == "complete" and len(data["runs"]) == 2 and data["trajectories"]["0"]["status"] == "ok"
  with pytest.raises(FileNotFoundError):
    Session(tmp_path / "missing", config, resume=True)


# --- end to end (planner: cuRobo + CUDA) --------------------------------------------------------
def _needs_planner():
  torch = pytest.importorskip("torch")
  pytest.importorskip("curobo")
  if not torch.cuda.is_available():
    pytest.skip("the motion planner needs a CUDA GPU")


@pytest.fixture(scope="module")
def planning(tmp_path_factory):
  _needs_planner()
  from sim2real.planning import GpuMotionPlanner, MujocoFK
  from sim2real.planning.adapters import planner_config_from_robot

  cfg = planner_config_from_robot(ROBOT, tmp_path_factory.mktemp("profiles_planner"))
  planner = GpuMotionPlanner(cfg)
  planner.warmup()
  return planner, MujocoFK(cfg.urdf_path, cfg.joint_names, cfg.tool_frame)


def test_generate_trajectory(planning):
  cfg = sampling.SamplingConfig(min_skills=3, max_skills=3, payload_change_prob=1.0, empty_prob=0.0)
  gen = ProfileGenerator(_backend("cuda:0"), *planning, sampling=cfg, seed=1)
  arrays, meta = gen.generate(0)
  n = meta["num_samples"]
  assert arrays["q"].shape == arrays["tau_ext"].shape == (n, 6) and arrays["t"].shape == arrays["skill"].shape == (n,)
  assert all(np.isfinite(a).all() for a in arrays.values())
  assert len(meta["skills"]) == 3 and {s["type"] for s in meta["skills"]} <= {"joint_move", "ee_move"}
  assert len({s["payload_mass"] for s in meta["skills"]}) == 3  # payload changed before each skill
  for k, skill in enumerate(meta["skills"]):
    rows = slice(skill["first_sample"], skill["first_sample"] + skill["num_samples"])
    assert np.all(arrays["skill"][rows] == k) and np.abs(arrays["dq"][rows]).max() > 0.05
  assert arrays["skill"][0] == HOLD and arrays["skill"][-1] == HOLD
  assert np.all(np.abs(arrays["tau_motor"]) <= np.asarray(ROBOT.torque_limits) + 1e-4)
  assert all(gen.checker.in_limits(q, 0.5) and gen.checker.clear(q) for q in arrays["q"][::25])
  np.testing.assert_allclose(arrays["tau_motor"][-1], -arrays["tau_ext"][-1], atol=0.02)  # at rest: balance
  assert np.abs(arrays["tau_ext"]).max() > 1.0  # gravity load is there
  assert np.all(np.abs(arrays["tau_motor"]) < 0.999 * np.asarray(ROBOT.torque_limits))  # never saturated


def test_skill_weights(planning):
  cfg = sampling.SamplingConfig(joint_weight=1.0, ee_weight=0.0, min_skills=2, max_skills=2)
  gen = ProfileGenerator(_backend("cuda:0"), *planning, sampling=cfg)
  assert [s["type"] for s in gen.generate(0)[1]["skills"]] == ["joint_move"] * 2


def test_cli_and_resume(tmp_path):
  _needs_planner()
  script = Path(__file__).resolve().parents[1] / "examples" / "motor_profiles.py"
  cmd = [sys.executable, str(script), "--out", str(tmp_path), "--min-skills", "2", "--max-skills", "2"]

  def run(*extra):
    return subprocess.run([*cmd, *extra], capture_output=True, text=True)

  first = run("--num-trajs", "2")
  assert first.returncode == 0, first.stderr[-2000:]
  assert sorted(p.name for p in tmp_path.iterdir()) == ["session.json", "traj_00000.npz", "traj_00001.npz"]
  assert run("--num-trajs", "2").returncode != 0  # no silent overwrite
  assert "seed" in run("--num-trajs", "2", "--resume", "--seed", "5").stderr

  traj_path(tmp_path, 0).unlink()
  again = run("--num-trajs", "3", "--resume")
  assert again.returncode == 0, again.stderr[-2000:]
  assert "[2/3]" not in again.stdout and "generated 2 trajectories" in again.stdout  # index 1 was kept
  data = json.loads((tmp_path / "session.json").read_text())
  assert data["status"] == "complete" and data["num_ok"] == 3 and len(data["runs"]) == 2  # refused runs leave no trace
  _, meta = load_trajectory(traj_path(tmp_path, 0))
  assert "friction" not in meta and meta["motor_models"][0] == "DM4340" and meta["motor_models"][5] == "DM4310"
