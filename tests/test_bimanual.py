"""Bimanual YAM: scene composition, per-arm control, scripted relay, real-robot loop."""

import mujoco
import numpy as np

from sim2real.control import OSC
from sim2real.envs.scripted import RELAY_HANDOFF, TOP_DOWN_QUAT, BimanualRelay
from sim2real.real.driver import RealArm, RealControlLoop, RobotDynamicsModel, SimulatedRig
from sim2real.sim.mujoco_backend import MujocoBackend


def test_scene_structure(yam_bimanual_scene):
  scene = yam_bimanual_scene
  m = scene.mjm
  assert scene.narm == 2 and m.nu == 14
  b = MujocoBackend(scene)
  left, right = b.views
  assert left.cfg.tcp_site == "left/pinch_site" and right.cfg.tcp_site == "right/pinch_site"
  for field in ("qpos_adr", "dof_adr", "act_ids"):
    assert not set(getattr(left, field)) & set(getattr(right, field))
  # mounts: bases 0.4 m apart along y, identical home poses -> mirrored TCPs
  pl, pr = (b.robot_state(arm=a).tcp_pos.numpy()[0] for a in range(2))
  np.testing.assert_allclose(pl - pr, [0.0, 0.4, 0.0], atol=1e-5)
  # per-robot contact excludes survive attach (prefixed)
  assert m.nexclude == 2 * len(scene.mounts[0].robot.contact_excludes)


def test_mass_matrix_block_diagonal(yam_bimanual_scene):
  """Fixed-base arms share no joints -> no inertial coupling -> per-arm controllers are exact."""
  b = MujocoBackend(yam_bimanual_scene)
  rng = np.random.default_rng(0)
  b.mjd.qpos[np.r_[b.views[0].qpos_adr, b.views[1].qpos_adr]] = rng.uniform(-0.5, 0.5, 12)
  mujoco.mj_forward(b.mjm, b.mjd)
  M = np.zeros((b.mjm.nv, b.mjm.nv))
  mujoco.mj_fullM(b.mjm, b.mjd, M)
  assert np.abs(M[np.ix_(b.views[0].dof_adr, b.views[1].dof_adr)]).max() == 0.0


def test_two_arms_track_independent_targets(yam_bimanual_scene):
  b = MujocoBackend(yam_bimanual_scene)
  oscs = [OSC(v, 1, device="cpu") for v in b.views]
  targets = [np.array([0.30, 0.30, 0.12]), np.array([0.32, -0.15, 0.12])]  # reachable top-down
  for osc, tgt in zip(oscs, targets):
    osc.set_target(tgt, quat_wxyz=TOP_DOWN_QUAT)
  b.control_substeps(oscs, 1500)
  for arm, tgt in enumerate(targets):
    st = b.robot_state(arm=arm)
    assert np.linalg.norm(st.tcp_pos.numpy()[0] - tgt) < 0.01, arm
    assert st.tcp_rot.numpy()[0][2, 2] < -0.99


def test_scripted_relay_cpu(yam_bimanual_scene):
  scene = yam_bimanual_scene
  b = MujocoBackend(scene)
  oscs = [OSC(v, 1, device="cpu") for v in b.views]
  park = [b.robot_state(arm=a).tcp_pos.numpy()[0].copy() for a in range(2)]
  relay = BimanualRelay(1, handoff=np.array(RELAY_HANDOFF), park=park)
  cube0, goal = np.array(scene.cfg.cube_pos), np.array(scene.cfg.goal_pos)
  t, handed_off = 0.0, None
  while t < relay.duration:
    quat = b.mjd.joint("cube_free").qpos[3:7]
    for arm, (pos, closed, q) in enumerate(relay(t, cube0, b.body_pos("cube")[0], goal, quat)):
      oscs[arm].set_target(pos, quat_wxyz=q)
      r = scene.robots[arm]
      b.set_gripper(r.gripper_closed if closed else r.gripper_open, arm=arm)
    b.control_substeps(oscs, 10)
    t += 0.02
    if handed_off is None and t >= relay.t_switch:
      handed_off = b.body_pos("cube")[0].copy()
  assert np.linalg.norm(handed_off[:2] - np.array(RELAY_HANDOFF)[:2]) < 0.02  # left arm's part
  assert np.linalg.norm(b.body_pos("cube")[0] - goal) < 0.02  # right arm's part


def test_dynamics_model_at_mount_matches_sim(yam_bimanual_scene):
  """Real-robot dynamics model (robot-only, placed at its mount) == that arm in the sim scene."""
  scene = yam_bimanual_scene
  b = MujocoBackend(scene)
  rng = np.random.default_rng(1)
  for arm, mount in enumerate(scene.mounts):
    v = b.views[arm]
    q, dq = rng.uniform(-0.5, 0.5, 6) + v.home_qpos, rng.normal(0, 0.5, 6)
    b.mjd.qpos[v.qpos_adr], b.mjd.qvel[v.dof_adr] = q, dq
    mujoco.mj_forward(b.mjm, b.mjd)
    sim = b.robot_state(arm=arm)
    real = RobotDynamicsModel(mount).update(q, dq)
    for k in ("tcp_pos", "tcp_rot", "J", "M", "bias"):
      np.testing.assert_allclose(getattr(real, k).numpy(), getattr(sim, k).numpy(), rtol=1e-4, atol=1e-5, err_msg=k)


def test_real_loop_two_arms(yam_bimanual_scene):
  """Real-robot loop over two arms (one tick = read both, compute both, command both) on a simulated rig."""
  scene = yam_bimanual_scene
  b = MujocoBackend(scene)
  rig = SimulatedRig(b)
  targets = [np.array([0.32, 0.25, 0.15]), np.array([0.32, -0.25, 0.15])]
  arms = []
  for i, (mount, tgt) in enumerate(zip(scene.mounts, targets)):
    osc = OSC(b.views[i], 1, device="cpu")
    osc.set_target(tgt, quat_wxyz=TOP_DOWN_QUAT)  # world-frame targets, same as in sim
    arms.append(RealArm(mount, rig.hardware[i], osc))
  RealControlLoop(arms, rate_hz=500).run(3.0, realtime=False, after_tick=rig.advance)
  for i, tgt in enumerate(targets):
    assert np.linalg.norm(b.mjd.site(scene.robots[i].tcp_site).xpos - tgt) < 0.01, i
