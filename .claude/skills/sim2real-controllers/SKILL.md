---
name: sim2real-controllers
description: How sim2real controllers (OSC, joint impedance, PD) are written as batched Warp kernels that run unchanged on GPU, Warp-CPU and the real robot, with their equations and conventions. Use when writing a new controller, changing control math or gains, or debugging tracking.
---

# Controllers as Warp kernels (`sim2real/control/`)

## Contract (`base.py`)

```python
class MyCtrl(Controller):
  requires = frozenset({NEED_M, NEED_J, NEED_BIAS})   # backends skip what isn't listed
  output = ActuatorMode.TORQUE                        # or POSITION; check_compatible() enforces it
  def __init__(self, view, nworld, ..., device=None):
    super().__init__(view, nworld, device)            # allocates self.cmd (nworld, n), self.tau_limit
    with wp.ScopedDevice(self.device): ...            # targets/gains as warp arrays, (nworld, ...) or (n,)
  def compute(self, state) -> wp.array:
    wp.launch(kernel, dim=self.nworld, inputs=[...], outputs=[self.cmd], device=self.device)
    return self.cmd
```

Rules:
- `compute` = kernel launches only. No `.numpy()`, no allocation, no Python branching on device data — it is
  captured into a CUDA graph (keyed by controller identity) and replayed.
- Targets live in preallocated buffers; update them with `.assign()` (host) or a kernel (RL env), never
  by rebinding the attribute (the graph holds the old buffer address).
- Always clamp torque output to `tau_limit`.
- Host helpers (`set_target`, `hold_current`) are for scripts; `_arr(value, shape)` broadcasts to float32.

## Fixed-size linear algebra (`linalg.py`)

`la = linalg(n)` (cached per n) gives `vec`, `mat`, `cholesky`, `cho_solve`, `spd_inverse`, `load_vec`,
`load_mat`. Per-world matrices live in registers; one thread per world. Kernels that depend on n are built in a
`@functools.cache` factory with `@wp.kernel(module="unique")` (see `_osc_kernel(n)`), so 6- and 7-DoF arms share
code. Use `wp.types.matrix(shape=(6, n))` for J-shaped types.

## Conventions (`RobotState`)

`J` is `(6, n)`, rows 0–2 translational, 3–5 rotational, world frame. `tcp_rot` is world-from-TCP. `bias` =
`C(q,q̇)q̇ + g(q)` (MuJoCo `qfrc_bias`). Everything float32.

## Equations (implemented)

- **JointPD** — POSITION: `cmd = q_des`. TORQUE: `τ = kp(q_des−q) + kd(q̇_des−q̇) [+ b]`.
- **JointImpedance** — `τ = K(q_des−q) + D(q̇_des−q̇) + b`; inertia-shaped: `τ = M[kp e + kd ė] + b`.
  Default `D = kd_cfg·√(K/kp_cfg)` (keeps configured damping ratio); inertia-shaped default `kd = 2√kp`.
- **OSC** —
  `e = [p_des−p ; log(R_des Rᵀ)]`, `v = J q̇`,
  `Λ = (J M⁻¹ Jᵀ + εI)⁻¹` (M⁻¹Jᵀ via one Cholesky of M),
  `F = Λ(Kp e − Kd v)`, `Kd = 2ζ√Kp`,
  `τ₀ = kp_null(q_null−q) − kd_null q̇`, `J̄ = M⁻¹JᵀΛ`,
  `τ = JᵀF + (I − JᵀJ̄ᵀ)τ₀ + b`. J̇q̇ neglected.
  Defaults: `kp_pos=400`, `kp_rot=150` (high because YAM joint friction 0.1 makes a deadband), `ε=1e-3`.

## Adding a controller

1. Implement kernel + class in `sim2real/control/<name>.py`, export in `control/__init__.py`.
2. Add a NumPy reference and compare on random states (`tests/test_controllers.py::osc_reference` pattern),
   tolerance ~1e-3 relative (float32 vs float64).
3. Add a closed-loop test on `MujocoBackend` (tracking error) and, if it uses M/J, run it on MJWarp too.
4. It automatically works on the real robot: `RealArm.send` maps TORQUE → MIT `kp=kd=0, τ_ff=τ`,
   POSITION → MIT `q_des=cmd` with config `kp/kd`.

## Multi-arm

Pass a list to `backend.control_substeps([ctrl_left, ctrl_right], n)` (None skips an arm). Separate per-arm
controllers are exact for fixed-base arms (block-diagonal M). A shared torso would need a whole-body controller.
