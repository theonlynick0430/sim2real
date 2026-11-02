---
name: sim2real-real-robot
description: How sim2real deploys the same controllers on physical robots — RobotHardware MIT-mode contract, RobotDynamicsModel, RealArm, RealControlLoop, SimulatedRig — and how to wire a new robot SDK (e.g. i2rt YAM). Use when working on sim2real/real/, hardware drivers, or real-robot control loops.
---

# Real-robot deployment (`sim2real/real/driver.py`)

## Loop

```
hardware.read() → (q, q̇) → RobotDynamicsModel.update → RobotState → controller.compute (Warp-CPU)
→ RealArm.send → hardware.command(q_des, dq_des, kp, kd, tau_ff)
```

## Pieces

- **`RobotHardware`** (ABC) — implement per SDK:
  `read() -> (q, dq)` (arm joints, config order), `command(q_des, dq_des, kp, kd, tau_ff)` (MIT mode,
  `τ = kp(q_des−q) + kd(q̇_des−q̇) + τ_ff`), optional `set_gripper(value)`.
- **`RobotDynamicsModel(robot | mount)`** — robot-only MuJoCo C model from the same asset via
  `compose_robots([mount])` (same sites/excludes as sim), `gravcomp=0`, placed at the mount pose.
  `update(q, dq, needs)` → `mj_forward` → `fill_state_from_mjdata`. Same function as the CPU sim backend.
- **`RealArm(mount, hardware, controller)`** — builds its model; `send()` maps controller output:
  TORQUE → `kp=kd=0, τ_ff=cmd`; POSITION → `q_des=cmd` with config `kp/kd`.
- **`RealControlLoop(arms, rate_hz=500)`** — `step()` reads **all** arms first (consistent snapshot), then
  computes and commands each. `run(seconds, realtime=True, after_tick=None)`.
- **`SimulatedRig(backend)` / `SimulatedHardware(backend, arm)`** — CPU-sim stand-in implementing
  `RobotHardware`; `SimulatedRig.advance` steps physics once per tick (pass as `after_tick`). Exercises the
  full deployment path in tests (`tests/test_controllers.py`, `tests/test_bimanual.py`).
- **`I2RTYamHardware`** — stub; wire the i2rt SDK (github.com/i2rt-robotics/i2rt) MIT-mode motor chain here.

## Wiring a new SDK

1. Subclass `RobotHardware`; map SDK joint order to `RobotConfig.arm_joints` order; units rad, rad/s, Nm.
2. Enforce safety in the driver too: torque limits, joint limits, a watchdog that zeros torque if the loop stalls.
3. Validate against `SimulatedRig` first with the same controller and targets, then on hardware with low gains
   (`OSC(kp_pos=..., kp_rot=...)`) and the arm near home.
4. Calibrate mount poses (`RobotMount.pos/quat`) on multi-arm rigs so sim targets transfer directly.

## Sim/real consistency

- Torque mode in sim has no gravcomp, so gravity compensation comes from `bias` in both.
- Controllers run float32 on Warp-CPU in sim CPU and on hardware alike.
- Model mismatch (masses, friction, motor dynamics) is the main gap — handle with domain randomization in the
  MJWarp env (per-world model fields) and system identification of the asset.
