"""Backend-agnostic scripted pick-and-place (host-side numpy, batched over worlds).

Produces OSC TCP targets + gripper commands from cube/goal positions. Used to
test physics + controllers end-to-end on CPU and GPU with identical code.
"""

import numpy as np

# Tool z-axis pointing down (180 deg about world y), MuJoCo wxyz.
TOP_DOWN_QUAT = np.array([0.0, 0.0, 1.0, 0.0])

# (name, duration [s], gripper closed?)
PHASES = (
  ("above_cube", 1.2, False),
  ("descend", 0.8, False),
  ("grasp", 0.5, True),
  ("lift", 0.8, True),
  ("to_goal", 1.2, True),
  ("place", 0.8, True),
  ("release", 0.5, False),
  ("retreat", 0.6, False),
)


class ScriptedPickPlace:
  def __init__(self, nworld: int, hover: float = 0.12, grasp_z_offset: float = 0.0):
    self.nworld = nworld
    self.hover = hover
    self.grasp_z_offset = grasp_z_offset
    self.ends = np.cumsum([p[1] for p in PHASES])
    self.duration = float(self.ends[-1])

  def phase(self, t: float) -> int:
    return int(min(np.searchsorted(self.ends, t, side="right"), len(PHASES) - 1))

  def __call__(self, t: float, cube_pos0: np.ndarray, goal_pos: np.ndarray):
    """t: time since episode start. cube_pos0: cube pos at episode start (nworld,3). Returns (pos, gripper_closed)."""
    i = self.phase(t)
    name, _, closed = PHASES[i]
    up = np.array([0.0, 0.0, self.hover])
    grasp = cube_pos0 + np.array([0.0, 0.0, self.grasp_z_offset])
    target = {
      "above_cube": cube_pos0 + up,
      "descend": grasp,
      "grasp": grasp,
      "lift": grasp + up,
      "to_goal": goal_pos + up,
      "place": goal_pos,
      "release": goal_pos,
      "retreat": goal_pos + up,
    }[name]
    return np.broadcast_to(target, (self.nworld, 3)), closed
