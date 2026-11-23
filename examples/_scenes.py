"""Scene presets and viewer helpers shared by the examples."""

import threading
import time

import mujoco

from sim2real.envs.scripted import RELAY_CUBE_POS, RELAY_GOAL_POS
from sim2real.robots import get_robot, yam_bimanual
from sim2real.sim.scene import SceneConfig, build_scene


def make_scene(bimanual: bool, separation: float = 0.4):
  """Single YAM pick-place, or bimanual YAM relay (cube in front of left arm, goal in front of right)."""
  if bimanual:
    return build_scene(yam_bimanual(separation), SceneConfig(cube_pos=RELAY_CUBE_POS, goal_pos=RELAY_GOAL_POS))
  return build_scene(get_robot("yam"))


def close_viewer(viewer, timeout: float = 5.0) -> None:
  """Close a `mujoco.viewer.launch_passive` handle and wait for its render thread to exit.

  `Handle.close()` only requests exit; the viewer keeps running on a daemon thread. Returning from main
  while it is still tearing down GL races the interpreter's atexit hooks (glfw.terminate, Warp's CUDA
  context release) and segfaults intermittently. Joining the thread first makes exit clean.
  """
  viewer.close()
  main = threading.main_thread()
  for t in threading.enumerate():
    if t is not main and t.daemon and "_launch_internal" in repr(getattr(t, "_target", "")):
      t.join(timeout)


class Viewer:
  """Passive MuJoCo viewer showing world 0 of a backend, paced to real time; does nothing when headless.

  perturb=True: ctrl-drag in the viewer pushes the simulated robot (world 0). Double-click a body to select it,
  then ctrl + right-drag applies a force and ctrl + left-drag a torque. Without it the viewer is display-only.
  """

  def __init__(self, backend, headless: bool, perturb: bool = False):
    self.backend, self.handle, self.perturb = backend, None, perturb
    if not headless:
      import mujoco.viewer

      self.data = backend.to_mjdata(0)
      self.handle = mujoco.viewer.launch_passive(backend.mjm, self.data)
    self._last = time.time()
    self._pushing = False

  def running(self) -> bool:
    return self.handle is None or self.handle.is_running()

  def sync(self, dt: float) -> None:
    """Show the current state, then sleep so frames `dt` seconds of sim apart are shown `dt` apart."""
    if self.handle is None:
      return
    self.backend.to_mjdata(0, out=self.data)
    self.handle.sync()
    if self.perturb:
      self._apply_perturbation()
    time.sleep(max(0.0, dt - (time.time() - self._last)))
    self._last = time.time()

  def _apply_perturbation(self) -> None:
    """Turn the viewer's ctrl-drag into a force on the selected body (MuJoCo's `mjv_applyPerturbForce`) and write it
    into world 0 of the simulation; it acts until the next frame."""
    data = self.data
    data.xfrc_applied[:] = 0.0
    with self.handle.lock():  # the viewer thread updates `perturb` while the mouse moves
      pert = self.handle.perturb
      active = bool(pert.active) and pert.select > 0
      if active:
        mujoco.mjv_applyPerturbForce(self.backend.mjm, data, pert)
    if active or self._pushing:  # write while dragging, and once more on release to clear the force
      xfrc = self.backend.d.xfrc_applied.numpy()
      xfrc[0] = data.xfrc_applied
      self.backend.d.xfrc_applied.assign(xfrc)
    self._pushing = active

  def close(self) -> None:
    if self.handle is not None:
      close_viewer(self.handle)
