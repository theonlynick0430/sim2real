"""Controller kernels vs NumPy references, plus closed-loop behaviour on MJWarp."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest

from sim2real.control import OSC, JointImpedance, JointPD
from sim2real.envs.scripted import TOP_DOWN_QUAT, ScriptedPickPlace
from sim2real.robots import ActuatorMode
from sim2real.sim.mjwarp_backend import MjWarpBackend


def _random_state(scene, seed, device):
  b = MjWarpBackend(scene, nworld=1, device=device)
  rng = np.random.default_rng(seed)
  v = b.view
  lo, hi = scene.mjm.jnt_range[:6].T
  qpos, qvel = b.d.qpos.numpy(), b.d.qvel.numpy()
  qpos[0, v.qpos_adr] = rng.uniform(lo + 0.2, hi - 0.2)
  qvel[0, v.dof_adr] = rng.normal(0, 0.5, 6)
  b.d.qpos.assign(qpos)
  b.d.qvel.assign(qvel)
  mjw.forward(b.m, b.d)
  return b, b.robot_state()


def _np(s):
  return {k: getattr(s, k).numpy()[0].astype(np.float64) for k in ("q", "dq", "tcp_pos", "tcp_rot", "J", "M", "bias")}


def _rotvec(R):
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, R.reshape(-1))
  if q[0] < 0:
    q = -q
  v = np.zeros(3)
  mujoco.mju_quat2Vel(v, q, 1.0)
  return v


def osc_reference(s, tp, tR, kp, kd, q_null, kpn, kdn, lam, lim, vel=np.zeros(6), acc=np.zeros(6)):
  J, M = s["J"], s["M"]
  Minv = np.linalg.inv(M)
  Lam = np.linalg.inv(J @ Minv @ J.T + lam * np.eye(6))
  e = np.concatenate([tp - s["tcp_pos"], _rotvec(tR @ s["tcp_rot"].T)])
  F = Lam @ (acc + kp * e + kd * (vel - J @ s["dq"]))
  tau0 = kpn * (q_null - s["q"]) - kdn * s["dq"]
  Jbar = Minv @ J.T @ Lam
  tau = J.T @ F + (np.eye(len(tau0)) - J.T @ Jbar.T) @ tau0 + s["bias"]
  return np.clip(tau, -lim, lim)


@pytest.mark.parametrize("seed", range(5))
def test_osc_matches_reference(yam_torque_scene, warp_device, seed):
  b, st = _random_state(yam_torque_scene, seed, warp_device)
  osc = OSC(b.view, 1, kp_pos=150, kp_rot=60, device=b.device)
  rng = np.random.default_rng(100 + seed)
  tp = st.tcp_pos.numpy()[0] + rng.normal(0, 0.03, 3)
  osc.set_target(tp, quat_wxyz=TOP_DOWN_QUAT)
  tau = osc.compute(st).numpy()[0]
  tR = osc.target_rot.numpy()[0].astype(np.float64)
  ref = osc_reference(_np(st), tp, tR, osc.kp.numpy(), osc.kd.numpy(), osc.q_null.numpy(), osc.kp_null, osc.kd_null,
                      osc.lambda_damping, b.view.torque_limits)
  np.testing.assert_allclose(tau, ref, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("seed", range(3))
def test_osc_feedforward_matches_reference(yam_torque_scene, warp_device, seed):
  b, st = _random_state(yam_torque_scene, seed, warp_device)
  osc = OSC(b.view, 1, kp_pos=150, kp_rot=60, device=b.device)
  rng = np.random.default_rng(200 + seed)
  tp = st.tcp_pos.numpy()[0] + rng.normal(0, 0.03, 3)
  vel, acc = rng.normal(0, 0.3, 6), rng.normal(0, 2.0, 6)
  osc.set_target(tp, quat_wxyz=TOP_DOWN_QUAT, vel=vel, acc=acc)
  tau = osc.compute(st).numpy()[0]
  tR = osc.target_rot.numpy()[0].astype(np.float64)
  ref = osc_reference(_np(st), tp, tR, osc.kp.numpy(), osc.kd.numpy(), osc.q_null.numpy(), osc.kp_null, osc.kd_null,
                      osc.lambda_damping, b.view.torque_limits, vel, acc)
  np.testing.assert_allclose(tau, ref, rtol=2e-3, atol=2e-3)
  osc.set_target(tp)  # pose-only call resets the feedforward to zero
  assert not osc.target_vel.numpy().any() and not osc.target_acc.numpy().any()


def test_joint_controllers_match_reference(yam_torque_scene, warp_device):
  b, st = _random_state(yam_torque_scene, 7, warp_device)
  s = _np(st)
  qd = s["q"] + 0.1
  lim = b.view.torque_limits
  imp = JointImpedance(b.view, 1, stiffness=50.0, device=b.device)
  imp.set_target(qd)
  ref = np.clip(imp.kp.numpy() * (qd - s["q"]) - imp.kd.numpy() * s["dq"] + s["bias"], -lim, lim)
  np.testing.assert_allclose(imp.compute(st).numpy()[0], ref, rtol=1e-4, atol=1e-4)

  ct = JointImpedance(b.view, 1, stiffness=100.0, inertia_shaped=True, device=b.device)
  ct.set_target(qd)
  ref = np.clip(s["M"] @ (100 * (qd - s["q"]) - 20 * s["dq"]) + s["bias"], -lim, lim)
  np.testing.assert_allclose(ct.compute(st).numpy()[0], ref, rtol=1e-4, atol=1e-4)
  dqd, ddqd = np.full(6, 0.2), np.linspace(-1, 1, 6)
  ct.set_target(qd, dqd, ddqd)
  ref = np.clip(s["M"] @ (ddqd + 100 * (qd - s["q"]) + 20 * (dqd - s["dq"])) + s["bias"], -lim, lim)
  np.testing.assert_allclose(ct.compute(st).numpy()[0], ref, rtol=1e-4, atol=1e-4)
  with pytest.raises(ValueError):
    imp.set_target(qd, ddq_des=ddqd)  # acceleration feedforward needs M (inertia_shaped)

  pd = JointPD(b.view, 1, mode=ActuatorMode.POSITION, device=b.device)
  pd.set_target(qd)
  np.testing.assert_allclose(pd.compute(st).numpy()[0], qd, atol=1e-6)
  with pytest.raises(ValueError):
    pd.check_compatible(ActuatorMode.TORQUE)


def test_osc_converges(yam_torque_scene, warp_device):
  b = MjWarpBackend(yam_torque_scene, nworld=1, device=warp_device)
  osc = OSC(b.view, 1, device=b.device)
  target = np.array([0.30, 0.10, 0.12])
  osc.set_target(target, quat_wxyz=TOP_DOWN_QUAT)
  b.control_substeps(osc, 1500)
  st = b.robot_state()
  assert np.linalg.norm(st.tcp_pos.numpy()[0] - target) < 0.01
  z = st.tcp_rot.numpy()[0][:, 2]
  assert z[2] < -0.99  # tool axis pointing down


def _track_circle(b, ctrl, use_ff, c0, steps=750):
  """Settle top-down at the circle start, then follow a 4 cm-radius TCP circle (0.5 Hz) around `c0`.
  Returns the max position error over the second half of the circle."""
  dt, w, r = b.mjm.opt.timestep, 2 * np.pi * 0.5, 0.04
  b.reset()
  ctrl.set_target(c0 + [r, 0.0, 0.0], quat_wxyz=TOP_DOWN_QUAT)
  b.control_substeps(ctrl, 1000)
  errs = []
  for k in range(steps):
    t = (k + 1) * dt  # the target for the state reached after this step
    p = c0 + r * np.array([np.cos(w * t), np.sin(w * t), 0.0])
    v = np.concatenate([r * w * np.array([-np.sin(w * t), np.cos(w * t), 0.0]), np.zeros(3)])
    a = np.concatenate([-r * w * w * np.array([np.cos(w * t), np.sin(w * t), 0.0]), np.zeros(3)])
    ctrl.set_target(p, quat_wxyz=TOP_DOWN_QUAT, vel=v if use_ff else None, acc=a if use_ff else None)
    b.control_substeps(ctrl, 1)
    if k >= steps // 2:
      errs.append(np.linalg.norm(b.robot_state().tcp_pos.numpy()[0] - p))
  return max(errs)


def test_osc_feedforward_improves_tracking(yam_torque_scene, warp_device):
  b = MjWarpBackend(yam_torque_scene, nworld=1, device=warp_device, use_graph=False)
  osc = OSC(b.view, 1, device=b.device)
  c0 = np.array([0.30, 0.0, 0.15])
  err_ff, err_reg = _track_circle(b, osc, True, c0), _track_circle(b, osc, False, c0)
  assert err_ff < 0.5 * err_reg, (err_ff, err_reg)
  assert err_ff < 0.01, err_ff


def test_scripted_pick_place(yam_torque_scene, warp_device):
  scene = yam_torque_scene
  robot = scene.robot
  b = MjWarpBackend(scene, nworld=1, device=warp_device)
  osc = OSC(b.view, 1, device=b.device)
  script = ScriptedPickPlace(1)
  cube0 = np.array([scene.cfg.cube_pos])
  goal = np.array(scene.cfg.goal_pos)
  t = 0.0
  while t < script.duration:
    pos, closed = script(t, cube0, goal)
    osc.set_target(pos, quat_wxyz=TOP_DOWN_QUAT)
    b.set_gripper(robot.gripper_closed if closed else robot.gripper_open)
    b.control_substeps(osc, 10)
    t += 0.02
  assert np.linalg.norm(b.body_pos("cube")[0] - goal) < 0.02

