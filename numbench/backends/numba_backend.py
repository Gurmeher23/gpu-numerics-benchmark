"""Optional Numba JIT CPU backend (float64 only), parallelised over the outer loop."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .base import Backend, check_softening, verlet


@njit(parallel=True, cache=True)
def _accel_kernel(pos, mass, eps2, g, out):  # pragma: no cover - exercised via backend
    n = pos.shape[0]
    for i in prange(n):
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            r2 = dx * dx + dy * dy + dz * dz + eps2
            w = mass[j] / (r2 * np.sqrt(r2))
            ax += w * dx
            ay += w * dy
            az += w * dz
        out[i, 0] = g * ax
        out[i, 1] = g * ay
        out[i, 2] = g * az


@njit(parallel=True, cache=True)
def _heat_kernel(u, v, r, steps):  # pragma: no cover - exercised via backend
    n, m = u.shape
    for _ in range(steps):
        for i in prange(1, n - 1):
            for j in range(1, m - 1):
                lap = u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4.0 * u[i, j]
                v[i, j] = u[i, j] + r * lap
        for i in prange(1, n - 1):
            for j in range(1, m - 1):
                u[i, j] = v[i, j]


class NumbaBackend(Backend):
    name = "numba-f64"
    dtype = "float64"
    device = "cpu"
    family = "numba"

    def _accel(self, pos, mass, eps2, g):
        out = np.empty_like(pos)
        _accel_kernel(pos, mass, eps2, g, out)
        return out

    def nbody_accel(self, pos, mass, softening, g=1.0):
        check_softening(softening)
        p = np.array(pos, dtype=np.float64)
        return self._accel(p, np.asarray(mass, dtype=np.float64), softening**2, g)

    def nbody_simulate(self, pos, vel, mass, softening, dt, steps, g=1.0):
        check_softening(softening)
        p, v = np.array(pos, dtype=np.float64), np.array(vel, dtype=np.float64)
        m = np.asarray(mass, dtype=np.float64)
        verlet(lambda x: self._accel(x, m, softening**2, g), p, v, dt, steps)
        return p, v

    def heat_solve(self, u0, r, steps):
        u = np.array(u0, dtype=np.float64)
        v = u.copy()  # boundary values are copied once and never rewritten
        _heat_kernel(u, v, r, steps)
        return u
