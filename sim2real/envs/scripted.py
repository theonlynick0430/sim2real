"""Backend-agnostic scripted pick-and-place (host-side numpy, batched over worlds).

Produces OSC TCP targets + gripper commands from cube/goal positions. Used to
test physics + controllers end-to-end on CPU and GPU with identical code.
"""

import numpy as np

# Tool z-axis pointing down (180 deg about world y), MuJoCo wxyz.
TOP_DOWN_QUAT = np.array([0.0, 0.0, 1.0, 0.0])


def top_down_quat(yaw: float = 0.0) -> np.ndarray:
  """Tool pointing down, rotated by `yaw` about world z (wxyz)."""
  qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
  w1, x1, y1, z1 = qz
  w2, x2, y2, z2 = TOP_DOWN_QUAT
  return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def box_grasp_yaw(quat_wxyz: np.ndarray) -> float:
  """Yaw that aligns a parallel-jaw gripper with the faces of an upright box, folded into
  [-45, 45] deg (box faces repeat every 90 deg)."""
  w, x, y, z = np.asarray(quat_wxyz, np.float64).reshape(4)
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)

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


class BimanualRelay:
  """Two-arm relay: left arm picks the cube and sets it down at a handoff spot, moves to its park
  pose, then the right arm picks it at the handoff spot and places it at the goal.

  Built from two `ScriptedPickPlace` scripts. Coordination (who moves when, keeping the shared
  workspace clear) lives here, above the per-arm controllers.
  """

  def __init__(self, nworld: int, handoff: np.ndarray, park: tuple[np.ndarray, np.ndarray], hover: float = 0.12,
               clear_time: float = 0.8):
    self.nworld = nworld
    self.handoff = np.asarray(handoff, np.float64)
    self.park = tuple(np.broadcast_to(np.asarray(p, np.float64), (nworld, 3)) for p in park)
    self.left = ScriptedPickPlace(nworld, hover)
    self.right = ScriptedPickPlace(nworld, hover)
    self.t_switch = self.left.duration + clear_time  # left is parked, right starts
    self.duration = self.t_switch + self.right.duration
    self._cube_at_handoff = None

  def __call__(self, t: float, cube_pos0: np.ndarray, cube_pos_now: np.ndarray, goal_pos: np.ndarray,
               cube_quat_now: np.ndarray | None = None):
    """Returns [(pos, closed, quat_wxyz) for left, right].

    The cube pose is latched when the right arm starts; the right arm then grasps aligned with
    the cube's yaw (the left arm's set-down leaves it slightly rotated).
    """
    cube_pos0 = np.broadcast_to(cube_pos0, (self.nworld, 3))
    if t < self.left.duration:
      left = (*self.left(t, cube_pos0, self.handoff), TOP_DOWN_QUAT)
    else:
      left = (self.park[0], False, TOP_DOWN_QUAT)
    if t < self.t_switch:
      right = (self.park[1], False, TOP_DOWN_QUAT)
    else:
      if self._cube_at_handoff is None:
        self._cube_at_handoff = np.broadcast_to(cube_pos_now, (self.nworld, 3)).copy()
        self._right_quat = TOP_DOWN_QUAT if cube_quat_now is None else top_down_quat(box_grasp_yaw(cube_quat_now))
      right = (*self.right(t - self.t_switch, self._cube_at_handoff, goal_pos), self._right_quat)
    return [left, right]


# Default bimanual relay layout for `yam_bimanual(0.4)` (bases at y = +-0.2): the cube starts
# in front of the left arm, the goal is in front of the right arm, the handoff spot is between.
RELAY_CUBE_POS = (0.30, 0.28, 0.02)
RELAY_GOAL_POS = (0.30, -0.28, 0.02)
RELAY_HANDOFF = (0.32, 0.0, 0.02)
