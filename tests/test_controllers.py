"""Controller kernels vs NumPy references, plus closed-loop behaviour on the CPU backend."""

import mujoco
import numpy as np
import pytest

from sim2real.control import OSC, JointImpedance, JointPD
from sim2real.envs.scripted import TOP_DOWN_QUAT, ScriptedPickPlace
from sim2real.robots import ActuatorMode
from sim2real.sim.mujoco_backend import MujocoBackend


def _random_state(scene, seed):
  b = MujocoBackend(scene)
  rng = np.random.default_rng(seed)
  v = b.view
  lo, hi = scene.mjm.jnt_range[:6].T
  b.mjd.qpos[v.qpos_adr] = rng.uniform(lo + 0.2, hi - 0.2)
  b.mjd.qvel[v.dof_adr] = rng.normal(0, 0.5, 6)
  mujoco.mj_forward(b.mjm, b.mjd)
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


def osc_reference(s, tp, tR, kp, kd, q_null, kpn, kdn, lam, lim):
  J, M = s["J"], s["M"]
  Minv = np.linalg.inv(M)
  Lam = np.linalg.inv(J @ Minv @ J.T + lam * np.eye(6))
  e = np.concatenate([tp - s["tcp_pos"], _rotvec(tR @ s["tcp_rot"].T)])
  F = Lam @ (kp * e - kd * (J @ s["dq"]))
  tau0 = kpn * (q_null - s["q"]) - kdn * s["dq"]
  Jbar = Minv @ J.T @ Lam
  tau = J.T @ F + (np.eye(len(tau0)) - J.T @ Jbar.T) @ tau0 + s["bias"]
  return np.clip(tau, -lim, lim)


@pytest.mark.parametrize("seed", range(5))
def test_osc_matches_reference(yam_torque_scene, seed):
  b, st = _random_state(yam_torque_scene, seed)
  osc = OSC(b.view, 1, kp_pos=150, kp_rot=60, device="cpu")
  rng = np.random.default_rng(100 + seed)
  tp = st.tcp_pos.numpy()[0] + rng.normal(0, 0.03, 3)
  osc.set_target(tp, quat_wxyz=TOP_DOWN_QUAT)
  tau = osc.compute(st).numpy()[0]
  tR = osc.target_rot.numpy()[0].astype(np.float64)
  ref = osc_reference(_np(st), tp, tR, osc.kp.numpy(), osc.kd.numpy(), osc.q_null.numpy(), osc.kp_null, osc.kd_null,
                      osc.lambda_damping, b.view.torque_limits)
  np.testing.assert_allclose(tau, ref, rtol=2e-3, atol=2e-3)


def test_joint_controllers_match_reference(yam_torque_scene):
  b, st = _random_state(yam_torque_scene, 7)
  s = _np(st)
  qd = s["q"] + 0.1
  lim = b.view.torque_limits
  imp = JointImpedance(b.view, 1, stiffness=50.0, device="cpu")
  imp.set_target(qd)
  ref = np.clip(imp.kp.numpy() * (qd - s["q"]) - imp.kd.numpy() * s["dq"] + s["bias"], -lim, lim)
  np.testing.assert_allclose(imp.compute(st).numpy()[0], ref, rtol=1e-4, atol=1e-4)

  ct = JointImpedance(b.view, 1, stiffness=100.0, inertia_shaped=True, device="cpu")
  ct.set_target(qd)
  ref = np.clip(s["M"] @ (100 * (qd - s["q"]) - 20 * s["dq"]) + s["bias"], -lim, lim)
  np.testing.assert_allclose(ct.compute(st).numpy()[0], ref, rtol=1e-4, atol=1e-4)

  pd = JointPD(b.view, 1, mode=ActuatorMode.POSITION, device="cpu")
  pd.set_target(qd)
  np.testing.assert_allclose(pd.compute(st).numpy()[0], qd, atol=1e-6)
  with pytest.raises(ValueError):
    pd.check_compatible(ActuatorMode.TORQUE)


def test_osc_converges(yam_torque_scene):
  b = MujocoBackend(yam_torque_scene)
  osc = OSC(b.view, 1, device="cpu")
  target = np.array([0.30, 0.10, 0.12])
  osc.set_target(target, quat_wxyz=TOP_DOWN_QUAT)
  b.control_substeps(osc, 1500)
  st = b.robot_state()
  assert np.linalg.norm(st.tcp_pos.numpy()[0] - target) < 0.01
  z = st.tcp_rot.numpy()[0][:, 2]
  assert z[2] < -0.99  # tool axis pointing down


def test_scripted_pick_place_cpu(yam_torque_scene):
  scene = yam_torque_scene
  robot = scene.robot
  b = MujocoBackend(scene)
  osc = OSC(b.view, 1, device="cpu")
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


def test_real_loop_with_simulated_hardware(yam_torque_scene):
  """Real-robot code path (separate dynamics model + MIT-mode conversion) drives the sim."""
  from sim2real.real.driver import RealArm, RealControlLoop, SimulatedRig

  b = MujocoBackend(yam_torque_scene)
  rig = SimulatedRig(b)
  osc = OSC(b.view, 1, device="cpu")
  target = np.array([0.32, -0.05, 0.15])
  osc.set_target(target, quat_wxyz=TOP_DOWN_QUAT)
  loop = RealControlLoop([RealArm(yam_torque_scene.mounts[0], rig.hardware[0], osc)], rate_hz=500)
  loop.run(3.0, realtime=False, after_tick=rig.advance)
  assert np.linalg.norm(b.mjd.site(yam_torque_scene.robot.tcp_site).xpos - target) < 0.01
