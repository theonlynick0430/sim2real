"""Real-robot side: hardware interface + MuJoCo-C dynamics model + control loop.

The same Controller objects used in sim run here, on Warp-CPU:

  hardware.read() -> (q, dq) -> RobotDynamicsModel (MuJoCo C on a robot-only model, same MJCF)
  -> RobotState -> controller.compute -> hardware.command(...)

Command convention is MIT mode (DM motors / most QDD actuators):
  tau = kp (q_des - q) + kd (dq_des - dq) + tau_ff
  TORQUE controllers -> kp = kd = 0, tau_ff = cmd
  POSITION controllers -> q_des = cmd with the robot's configured kp/kd
"""

import abc
import time

import mujoco
import numpy as np
import warp as wp

from s2r.robots.config import ActuatorMode, RobotConfig
from s2r.sim.mujoco_backend import fill_state_from_mjdata
from s2r.sim.robot_view import RobotView
from s2r.sim.state import ALL_NEEDS, RobotState


class RobotHardware(abc.ABC):
  """Minimal hardware contract. Implement per robot SDK (e.g. i2rt for YAM)."""

  @abc.abstractmethod
  def read(self) -> tuple[np.ndarray, np.ndarray]:
    """Measured arm (q, dq), each (ndof,)."""

  @abc.abstractmethod
  def command(self, q_des, dq_des, kp, kd, tau_ff) -> None:
    """MIT-mode command, all (ndof,)."""

  def set_gripper(self, value: float) -> None:  # optional
    pass


class RobotDynamicsModel:
  """Robot-only MuJoCo C model that turns measured (q, dq) into a RobotState."""

  def __init__(self, cfg: RobotConfig):
    spec = mujoco.MjSpec.from_file(str(cfg.model_path))
    for name, body, pos, quat in cfg.extra_sites:
      spec.body(body).add_site(name=name, pos=list(pos), quat=list(quat))
    for body in spec.bodies:  # real robot has no simulator gravity compensation
      body.gravcomp = 0.0
    self.mjm = spec.compile()
    self.mjd = mujoco.MjData(self.mjm)
    self.view = RobotView.from_model(self.mjm, cfg)
    self.state = RobotState.allocate(1, self.view.ndof, "cpu")

  def update(self, q: np.ndarray, dq: np.ndarray, needs=ALL_NEEDS) -> RobotState:
    self.mjd.qpos[self.view.qpos_adr] = q
    self.mjd.qvel[self.view.dof_adr] = dq
    mujoco.mj_forward(self.mjm, self.mjd)  # kinematics, M, bias (M/bias/J only need fwd position+velocity)
    fill_state_from_mjdata(self.mjm, self.mjd, self.view, self.state, needs)
    return self.state


class RealControlLoop:
  def __init__(self, cfg: RobotConfig, hardware: RobotHardware, controller, rate_hz: float = 500.0):
    self.cfg, self.hw, self.ctrl = cfg, hardware, controller
    self.model = RobotDynamicsModel(cfg)
    self.dt = 1.0 / rate_hz
    n = cfg.ndof
    self._zeros = np.zeros(n)
    self._kp = np.asarray(cfg.kp or np.zeros(n))
    self._kd = np.asarray(cfg.kd or np.zeros(n))

  def step(self) -> None:
    q, dq = self.hw.read()
    state = self.model.update(q, dq, self.ctrl.requires)
    cmd = self.ctrl.compute(state).numpy()[0]
    if self.ctrl.output == ActuatorMode.TORQUE:
      self.hw.command(q, self._zeros, self._zeros, self._zeros, cmd)
    else:
      self.hw.command(cmd, self._zeros, self._kp, self._kd, self._zeros)

  def run(self, seconds: float, realtime: bool = True) -> None:
    t_next = time.perf_counter()
    for _ in range(int(seconds / self.dt)):
      self.step()
      if realtime:
        t_next += self.dt
        time.sleep(max(0.0, t_next - time.perf_counter()))


class SimulatedHardware(RobotHardware):
  """`RobotHardware` backed by the CPU sim (torque-mode scene): exercises the real-robot
  code path (separate dynamics model, MIT command conversion) without hardware."""

  def __init__(self, backend):
    self.b = backend
    self.n = backend.view.ndof

  def read(self):
    d, v = self.b.mjd, self.b.view
    return d.qpos[v.qpos_adr].copy(), d.qvel[v.dof_adr].copy()

  def command(self, q_des, dq_des, kp, kd, tau_ff):
    q, dq = self.read()
    tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
    self.b.step1()
    self.b.set_arm_ctrl(wp.array(tau[None].astype(np.float32), dtype=float, device="cpu"))
    self.b.step2()

  def set_gripper(self, value):
    self.b.set_gripper(value)


class I2RTYamHardware(RobotHardware):
  """Placeholder for the i2rt SDK (github.com/i2rt-robotics/i2rt): wire `read`/`command` to its
  MIT-mode motor chain. Left unimplemented until hardware is available."""

  def __init__(self, *_, **__):
    raise NotImplementedError("connect i2rt SDK here")

  def read(self):  # pragma: no cover
    raise NotImplementedError

  def command(self, q_des, dq_des, kp, kd, tau_ff):  # pragma: no cover
    raise NotImplementedError
