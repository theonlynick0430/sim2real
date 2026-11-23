"""Time-phased experts for PickPlaceEnv, as Warp kernels over worlds (used by examples/tests).

Offsets come from `env.layout` (ObsLayout), so the experts don't hard-code the obs vector.
"""

import warp as wp


@wp.func
def _vec(obs: wp.array2d(dtype=float), w: int, offset: int) -> wp.vec3:
  return wp.vec3(obs[w, offset], obs[w, offset + 1], obs[w, offset + 2])


@wp.func
def _write(action: wp.array2d(dtype=float), w: int, col: int, target: wp.vec3, tcp: wp.vec3, scale: float,
           grip: float):
  delta = (target - tcp) / scale
  for i in range(3):
    action[w, col + i] = wp.clamp(delta[i], -1.0, 1.0)
  action[w, col + 3] = grip


@wp.func
def _pick_place(t: int, tcp: wp.vec3, cube: wp.vec3, place: wp.vec3, hover: float):
  """One arm's pick schedule, `t` policy steps after it starts: above the cube, descend, grasp, lift,
  carry to above `place`, lower onto it (still holding). Returns (TCP target, grip command)."""
  up = wp.vec3(0.0, 0.0, hover)
  target = cube + up
  grip = float(-1.0)
  if t >= 60:
    target = cube
  if t >= 100:
    grip = 1.0
  if t >= 125:
    target = wp.vec3(tcp[0], tcp[1], place[2] + hover)
  if t >= 165:
    target = place + up
  if t >= 225:
    target = place
  return target, grip


@wp.kernel
def expert_kernel(obs: wp.array2d(dtype=float), step: wp.array(dtype=int), scale: float, hover: float,
                  tcp_offset: int, cube_offset: int, goal_offset: int, action: wp.array2d(dtype=float)):
  """Single arm: pick the cube, carry it to the goal."""
  w = wp.tid()
  tcp = _vec(obs, w, tcp_offset)
  target, grip = _pick_place(step[w], tcp, _vec(obs, w, cube_offset), _vec(obs, w, goal_offset), hover)
  _write(action, w, 0, target, tcp, scale, grip)


RELAY_STEPS = 640  # policy steps the relay expert needs (at 50 Hz)
RIGHT_START = wp.constant(355)  # policy step at which the right arm starts its pick (the left arm is parked by then)


@wp.kernel
def relay_expert_kernel(obs: wp.array2d(dtype=float), step: wp.array(dtype=int), scale: float, hover: float,
                        tcp_offset_left: int, tcp_offset_right: int, cube_offset: int, goal_offset: int,
                        handoff: wp.vec3, park_left: wp.vec3, park_right: wp.vec3, action: wp.array2d(dtype=float)):
  """Bimanual relay: left picks the cube and sets it down at `handoff`, parks; right picks it
  there and carries it to the goal. Arm 0 = left (action cols 0-3), arm 1 = right (4-7)."""
  w = wp.tid()
  t = step[w]
  tcp_left = _vec(obs, w, tcp_offset_left)
  tcp_right = _vec(obs, w, tcp_offset_right)
  cube = _vec(obs, w, cube_offset)
  goal = _vec(obs, w, goal_offset)

  # left arm: cube -> handoff, release, lift, park
  target_left, grip_left = _pick_place(t, tcp_left, cube, handoff, hover)
  if t >= 265:
    grip_left = -1.0
  if t >= 285:
    target_left = handoff + wp.vec3(0.0, 0.0, hover)
  if t >= 315:
    target_left = park_left

  # right arm: parked, then handoff (wherever the cube is) -> goal
  target_right = park_right
  grip_right = float(-1.0)
  if t >= RIGHT_START:
    target_right, grip_right = _pick_place(t - RIGHT_START, tcp_right, cube, goal, hover)

  _write(action, w, 0, target_left, tcp_left, scale, grip_left)
  _write(action, w, 4, target_right, tcp_right, scale, grip_right)


def launch_expert(env, action, hover: float = 0.10):
  """Pick the expert matching the env (1 arm: pick-place, 2 arms: relay) and launch it."""
  layout = env.layout
  if env.narm == 1:
    wp.launch(expert_kernel, dim=env.nworld,
              inputs=[env.obs, env.step_count, env.cfg.action_scale, hover, layout.tcp(0), layout.cube, layout.goal],
              outputs=[action], device=env.device)
  elif env.narm == 2:
    from sim2real.envs.scripted import RELAY_HANDOFF

    wp.launch(relay_expert_kernel, dim=env.nworld,
              inputs=[env.obs, env.step_count, env.cfg.action_scale, hover, layout.tcp(0), layout.tcp(1), layout.cube,
                      layout.goal, wp.vec3(*RELAY_HANDOFF), env.home_tcp[0], env.home_tcp[1]],
              outputs=[action], device=env.device)
  else:
    raise NotImplementedError("expert defined for 1 or 2 arms")
