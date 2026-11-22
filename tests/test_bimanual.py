"""Bimanual YAM: scene composition, per-arm control, scripted relay."""

import numpy as np

from sim2real.control import OSC
from sim2real.envs.scripted import RELAY_HANDOFF, TOP_DOWN_QUAT, BimanualRelay
from sim2real.sim.mjwarp_backend import MjWarpBackend
from sim2real.sim.robot_view import RobotView


def test_scene_structure(yam_bimanual_scene, warp_device):
  scene = yam_bimanual_scene
  m = scene.mjm
  assert scene.narm == 2 and m.nu == 14
  b = MjWarpBackend(scene, nworld=1, device=warp_device)
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
  """Fixed-base arms share no joints -> no inertial coupling -> per-arm controllers are exact.

  Checked on the sparsity pattern of M (the CSR layout MJWarp's `d.M` uses): no stored entry couples the arms.
  """
  m = yam_bimanual_scene.mjm
  left, right = (set(RobotView.from_model(m, r).dof_adr) for r in yam_bimanual_scene.robots)
  for row in range(m.nv):
    cols = m.M_colind[m.M_rowadr[row] : m.M_rowadr[row] + m.M_rownnz[row]]
    for col in cols:
      assert not ({row, col} & left and {row, col} & right), (row, col)


def test_two_arms_track_independent_targets(yam_bimanual_scene, warp_device):
  b = MjWarpBackend(yam_bimanual_scene, nworld=1, device=warp_device)
  oscs = [OSC(v, 1, device=b.device) for v in b.views]
  targets = [np.array([0.30, 0.30, 0.12]), np.array([0.32, -0.15, 0.12])]  # reachable top-down
  for osc, tgt in zip(oscs, targets):
    osc.set_target(tgt, quat_wxyz=TOP_DOWN_QUAT)
  b.control_substeps(oscs, 1500)
  for arm, tgt in enumerate(targets):
    st = b.robot_state(arm=arm)
    assert np.linalg.norm(st.tcp_pos.numpy()[0] - tgt) < 0.01, arm
    assert st.tcp_rot.numpy()[0][2, 2] < -0.99


def test_scripted_relay(yam_bimanual_scene, warp_device):
  scene = yam_bimanual_scene
  b = MjWarpBackend(scene, nworld=1, device=warp_device)
  oscs = [OSC(v, 1, device=b.device) for v in b.views]
  park = [b.robot_state(arm=a).tcp_pos.numpy()[0].copy() for a in range(2)]
  relay = BimanualRelay(1, handoff=np.array(RELAY_HANDOFF), park=park)
  cube0, goal = np.array(scene.cfg.cube_pos), np.array(scene.cfg.goal_pos)
  t, handed_off = 0.0, None
  while t < relay.duration:
    quat = b.body_quat(scene.cube_body)[0]
    for arm, (pos, closed, q) in enumerate(relay(t, cube0, b.body_pos(scene.cube_body)[0], goal, quat)):
      oscs[arm].set_target(pos, quat_wxyz=q)
      r = scene.robots[arm]
      b.set_gripper(r.gripper_closed if closed else r.gripper_open, arm=arm)
    b.control_substeps(oscs, 10)
    t += 0.02
    if handed_off is None and t >= relay.t_switch:
      handed_off = b.body_pos(scene.cube_body)[0].copy()
  assert np.linalg.norm(handed_off[:2] - np.array(RELAY_HANDOFF)[:2]) < 0.02  # left arm's part
  assert np.linalg.norm(b.body_pos(scene.cube_body)[0] - goal) < 0.02  # right arm's part
