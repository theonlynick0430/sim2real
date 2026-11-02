"""Fixed-size dense linear algebra for per-world controller kernels.

Robot DoF is small (6-7), so each world's M (n x n), J (6 x n) and Lambda (6 x 6)
live in registers as Warp fixed-size matrices. Types/functions are generated
per n and cached, so one code path serves YAM (6), Franka (7), etc.
"""


import functools
from types import SimpleNamespace

import warp as wp


@functools.cache
def linalg(n: int) -> SimpleNamespace:
  vec = wp.types.vector(length=n, dtype=wp.float32)
  mat = wp.types.matrix(shape=(n, n), dtype=wp.float32)

  @wp.func
  def cholesky(A: mat) -> mat:
    """Lower-triangular L with A = L L^T (A SPD; diagonal floored for robustness)."""
    L = mat()
    for j in range(n):
      s = A[j, j]
      for k in range(j):
        s -= L[j, k] * L[j, k]
      L[j, j] = wp.sqrt(wp.max(s, 1.0e-12))
      for i in range(j + 1, n):
        t = A[i, j]
        for k in range(j):
          t -= L[i, k] * L[j, k]
        L[i, j] = t / L[j, j]
    return L

  @wp.func
  def cho_solve(L: mat, b: vec) -> vec:
    """Solve (L L^T) x = b."""
    y = vec()
    for i in range(n):
      s = b[i]
      for k in range(i):
        s -= L[i, k] * y[k]
      y[i] = s / L[i, i]
    x = vec()
    for ii in range(n):
      i = n - 1 - ii
      s = y[i]
      for k in range(i + 1, n):
        s -= L[k, i] * x[k]
      x[i] = s / L[i, i]
    return x

  @wp.func
  def spd_inverse(A: mat) -> mat:
    L = cholesky(A)
    inv = mat()
    for c in range(n):
      e = vec()
      e[c] = 1.0
      col = cho_solve(L, e)
      for r in range(n):
        inv[r, c] = col[r]
    return inv

  @wp.func
  def load_vec(a: wp.array2d(dtype=float), w: int) -> vec:
    v = vec()
    for i in range(n):
      v[i] = a[w, i]
    return v

  @wp.func
  def load_mat(a: wp.array3d(dtype=float), w: int) -> mat:
    m = mat()
    for i in range(n):
      for j in range(n):
        m[i, j] = a[w, i, j]
    return m

  return SimpleNamespace(
    n=n, vec=vec, mat=mat, cholesky=cholesky, cho_solve=cho_solve, spd_inverse=spd_inverse, load_vec=load_vec,
    load_mat=load_mat,
  )
