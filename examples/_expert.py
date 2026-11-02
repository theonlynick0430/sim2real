"""Time-phased experts for PickPlaceEnv, as Warp kernels over worlds (used by examples/tests).

Offsets come from `env.layout` (ObsLayout), so the experts don't hard-code the obs vector.
"""

import warp as wp


@wp.func
def _vec(obs: wp.array2d(dtype=float), w: int, off: int) -> wp.vec3:
  return wp.vec3(obs[w, off], obs[w, off + 1], obs[w, off + 2])


@wp.func
def _write(action: wp.array2d(dtype=float), w: int, col: int, target: wp.vec3, tcp: wp.vec3, scale: float,
           grip: float):
  d = (target - tcp) / scale
  for i in range(3):
    action[w, col + i] = wp.clamp(d[i], -1.0, 1.0)
  action[w, col + 3] = grip


@wp.kernel
def expert_kernel(obs: wp.array2d(dtype=float), step: wp.array(dtype=int), scale: float, hover: float, tcp_off: int,
                  cube_off: int, goal_off: int, action: wp.array2d(dtype=float)):
  """Single arm: pick the cube, carry it to the goal."""
  w = wp.tid()
  t = step[w]
  tcp = _vec(obs, w, tcp_off)
  cube = _vec(obs, w, cube_off)
  goal = _vec(obs, w, goal_off)
  up = wp.vec3(0.0, 0.0, hover)
  target = cube + up
  grip = float(-1.0)
  if t >= 60:
    target = cube
  if t >= 100:
    grip = 1.0
  if t >= 125:
    target = wp.vec3(tcp[0], tcp[1], goal[2] + hover)
  if t >= 165:
    target = goal + up
  if t >= 225:
    target = goal
  _write(action, w, 0, target, tcp, scale, grip)


RELAY_STEPS = 640  # policy steps the relay expert needs (at 50 Hz)


@wp.kernel
def relay_expert_kernel(obs: wp.array2d(dtype=float), step: wp.array(dtype=int), scale: float, hover: float,
                        tcp_off_l: int, tcp_off_r: int, cube_off: int, goal_off: int, handoff: wp.vec3,
                        park_l: wp.vec3, park_r: wp.vec3, action: wp.array2d(dtype=float)):
  """Bimanual relay: left picks the cube and sets it down at `handoff`, parks; right picks it
  there and carries it to the goal. Arm 0 = left (action cols 0-3), arm 1 = right (4-7)."""
  w = wp.tid()
  t = step[w]
  tl = _vec(obs, w, tcp_off_l)
  tr = _vec(obs, w, tcp_off_r)
  cube = _vec(obs, w, cube_off)
  goal = _vec(obs, w, goal_off)
  up = wp.vec3(0.0, 0.0, hover)

  # left arm: cube -> handoff, then park
  tgt_l = cube + up
  grip_l = float(-1.0)
  if t >= 60:
    tgt_l = cube
  if t >= 100:
    grip_l = 1.0
  if t >= 125:
    tgt_l = wp.vec3(tl[0], tl[1], handoff[2] + hover)
  if t >= 165:
    tgt_l = handoff + up
  if t >= 225:
    tgt_l = handoff
  if t >= 265:
    grip_l = -1.0
  if t >= 285:
    tgt_l = handoff + up
  if t >= 315:
    tgt_l = park_l

  # right arm: park, then handoff (wherever the cube is) -> goal
  tgt_r = park_r
  grip_r = float(-1.0)
  if t >= 355:
    tgt_r = cube + up
  if t >= 415:
    tgt_r = cube
  if t >= 455:
    grip_r = 1.0
  if t >= 480:
    tgt_r = wp.vec3(tr[0], tr[1], goal[2] + hover)
  if t >= 520:
    tgt_r = goal + up
  if t >= 580:
    tgt_r = goal

  _write(action, w, 0, tgt_l, tl, scale, grip_l)
  _write(action, w, 4, tgt_r, tr, scale, grip_r)


def launch_expert(env, action, hover: float = 0.10):
  """Pick the expert matching the env (1 arm: pick-place, 2 arms: relay) and launch it."""
  L = env.layout
  if env.narm == 1:
    wp.launch(expert_kernel, dim=env.nworld,
              inputs=[env.obs, env.step_count, env.cfg.action_scale, hover, L.tcp(0), L.cube, L.goal],
              outputs=[action], device=env.device)
  elif env.narm == 2:
    from s2r.envs.scripted import RELAY_HANDOFF

    wp.launch(relay_expert_kernel, dim=env.nworld,
              inputs=[env.obs, env.step_count, env.cfg.action_scale, hover, L.tcp(0), L.tcp(1), L.cube, L.goal,
                      wp.vec3(*RELAY_HANDOFF), env.home_tcp[0], env.home_tcp[1]],
              outputs=[action], device=env.device)
  else:
    raise NotImplementedError("expert defined for 1 or 2 arms")
