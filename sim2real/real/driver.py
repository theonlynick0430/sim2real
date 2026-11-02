"""Real-robot side: hardware interface + MuJoCo-C dynamics model + control loop.

The same Controller objects used in sim run here, on Warp-CPU:

  hardware.read() -> (q, dq) -> RobotDynamicsModel (MuJoCo C on a robot-only model, same MJCF)
  -> RobotState -> controller.compute -> hardware.command(...)

Multi-arm rigs (e.g. bimanual YAM) are a list of `RealArm`s, one hardware handle +
controller per arm, ticked together. Each arm's dynamics model is placed at its
mount pose, so RobotState (TCP pose, J) is in the same world frame as the sim
scene and the same targets work in both. Fixed-base arms share no joints, so
per-arm models are exact.

Command convention is MIT mode (DM motors / most QDD actuators):
  tau = kp (q_des - q) + kd (dq_des - dq) + tau_ff
  TORQUE controllers -> kp = kd = 0, tau_ff = cmd
  POSITION controllers -> q_des = cmd with the robot's configured kp/kd
"""

import abc
import dataclasses
import time
from collections.abc import Callable, Sequence

import mujoco
import numpy as np
import warp as wp

from sim2real.robots.config import ActuatorMode, RobotConfig, RobotMount
from sim2real.sim.mujoco_backend import fill_state_from_mjdata
from sim2real.sim.robot_view import RobotView
from sim2real.sim.scene import compose_robots
from sim2real.sim.state import ALL_NEEDS, RobotState


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
  """Robot-only MuJoCo C model (at its mount pose) that turns measured (q, dq) into a RobotState."""

  def __init__(self, robot: RobotConfig | RobotMount):
    mount = robot if isinstance(robot, RobotMount) else RobotMount(robot)
    spec = compose_robots([mount])  # same robot-level edits as the sim scene (sites, excludes, ...)
    for body in spec.bodies:  # real robot has no simulator gravity compensation
      body.gravcomp = 0.0
    self.mjm = spec.compile()
    self.mjd = mujoco.MjData(self.mjm)
    self.view = RobotView.from_model(self.mjm, mount.resolved)
    self.state = RobotState.allocate(1, self.view.ndof, "cpu")

  def update(self, q: np.ndarray, dq: np.ndarray, needs=ALL_NEEDS) -> RobotState:
    self.mjd.qpos[self.view.qpos_adr] = q
    self.mjd.qvel[self.view.dof_adr] = dq
    mujoco.mj_forward(self.mjm, self.mjd)  # kinematics, M, bias (M/bias/J only need fwd position+velocity)
    fill_state_from_mjdata(self.mjm, self.mjd, self.view, self.state, needs)
    return self.state


@dataclasses.dataclass
class RealArm:
  """One arm of a real rig: where it is mounted, how to talk to it, what controls it."""

  mount: RobotMount
  hardware: RobotHardware
  controller: object  # sim2real.control.Controller

  def __post_init__(self):
    self.model = RobotDynamicsModel(self.mount)
    n = self.mount.robot.ndof
    self._zeros = np.zeros(n)
    self._kp = np.asarray(self.mount.robot.kp or np.zeros(n))
    self._kd = np.asarray(self.mount.robot.kd or np.zeros(n))

  def send(self, q: np.ndarray, cmd: np.ndarray) -> None:
    """MIT-mode conversion of a controller command."""
    if self.controller.output == ActuatorMode.TORQUE:
      self.hardware.command(q, self._zeros, self._zeros, self._zeros, cmd)
    else:
      self.hardware.command(cmd, self._zeros, self._kp, self._kd, self._zeros)


class RealControlLoop:
  """Ticks all arms together: read every arm, then compute, then command every arm."""

  def __init__(self, arms: Sequence[RealArm], rate_hz: float = 500.0):
    self.arms = list(arms)
    self.dt = 1.0 / rate_hz

  def step(self) -> None:
    meas = [a.hardware.read() for a in self.arms]  # read all first: one consistent snapshot
    for a, (q, dq) in zip(self.arms, meas):
      state = a.model.update(q, dq, a.controller.requires)
      a.send(q, a.controller.compute(state).numpy()[0])

  def run(self, seconds: float, realtime: bool = True, after_tick: Callable[[], None] | None = None) -> None:
    """`after_tick`: called after each tick (e.g. `SimulatedRig.advance` to step a simulated rig)."""
    t_next = time.perf_counter()
    for _ in range(int(seconds / self.dt)):
      self.step()
      if after_tick is not None:
        after_tick()
      if realtime:
        t_next += self.dt
        time.sleep(max(0.0, t_next - time.perf_counter()))


class SimulatedRig:
  """A CPU-sim scene standing in for a real rig: one `SimulatedHardware` per arm, physics
  advanced once per control tick (after all arms are commanded). Exercises the real-robot
  code path (separate dynamics models, MIT command conversion) without hardware."""

  def __init__(self, backend):
    self.b = backend
    self.hardware = [SimulatedHardware(backend, arm) for arm in range(backend.narm)]

  def advance(self) -> None:
    self.b.step1()
    self.b.step2()


class SimulatedHardware(RobotHardware):
  """`RobotHardware` for one arm of a CPU-sim (torque-mode) scene. Does not step physics;
  `SimulatedRig.advance` does, once per tick for all arms."""

  def __init__(self, backend, arm: int = 0):
    self.b, self.arm = backend, arm
    self.v = backend.views[arm]

  def read(self):
    d = self.b.mjd
    return d.qpos[self.v.qpos_adr].copy(), d.qvel[self.v.dof_adr].copy()

  def command(self, q_des, dq_des, kp, kd, tau_ff):
    q, dq = self.read()  # motor-side PD runs on current measurements, as on the drive
    tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
    self.b.set_arm_ctrl(wp.array(tau[None].astype(np.float32), dtype=float, device="cpu"), self.arm)

  def set_gripper(self, value):
    self.b.set_gripper(value, self.arm)


class I2RTYamHardware(RobotHardware):
  """Placeholder for the i2rt SDK (github.com/i2rt-robotics/i2rt): wire `read`/`command` to its
  MIT-mode motor chain. Left unimplemented until hardware is available."""

  def __init__(self, *_, **__):
    raise NotImplementedError("connect i2rt SDK here")

  def read(self):  # pragma: no cover
    raise NotImplementedError

  def command(self, q_des, dq_des, kp, kd, tau_ff):  # pragma: no cover
    raise NotImplementedError
