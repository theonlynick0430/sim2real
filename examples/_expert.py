"""Time-phased expert for PickPlaceEnv, as a Warp kernel over worlds (used by GPU examples/tests)."""

import warp as wp


@wp.kernel
def expert_kernel(obs: wp.array2d(dtype=float), step: wp.array(dtype=int), scale: float, hover: float,
                  action: wp.array2d(dtype=float)):
  w = wp.tid()
  t = step[w]
  tcp = wp.vec3(obs[w, 12], obs[w, 13], obs[w, 14])
  cube = wp.vec3(obs[w, 16], obs[w, 17], obs[w, 18])
  goal = wp.vec3(obs[w, 23], obs[w, 24], obs[w, 25])
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
  d = (target - tcp) / scale
  for i in range(3):
    action[w, i] = wp.clamp(d[i], -1.0, 1.0)
  action[w, 3] = grip
