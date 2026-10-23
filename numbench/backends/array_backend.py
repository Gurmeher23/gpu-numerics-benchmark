"""NumPy and CuPy backends: one implementation over the shared array API.

`ArrayBackend` takes the array module (`numpy` or `cupy`) as a parameter, so the
CuPy path executes exactly the same expressions as the NumPy reference, just on the
device. NOTE: the CuPy path is written against the CuPy API but has NOT been executed
on this machine (no CUDA); the NumPy path is what the tests exercise.
"""

from __future__ import annotations

import numpy as np

from .base import Backend, check_softening, heat_ftcs, verlet


def accel_xp(xp, pos, mass, eps2, g):
    """O(N^2) softened gravity, fully vectorised.

    diff[i, j] = pos[j] - pos[i]; a_i = G * sum_j m_j * diff[i, j] / (|diff|^2 + eps^2)^1.5.
    The i == j term has diff == 0, so it adds exactly zero and needs no masking.
    The difference form (rather than |a|^2 + |b|^2 - 2 a.b) avoids catastrophic
    cancellation for nearby particles in float32.
    """
    diff = pos[None, :, :] - pos[:, None, :]
    r2 = (diff * diff).sum(axis=-1) + eps2
    w = mass[None, :] * r2**-1.5
    return g * (w[:, :, None] * diff).sum(axis=1)


class ArrayBackend(Backend):
    def __init__(self, xp, dtype: str, name: str, family: str, device: str = "cpu"):
        self.xp = xp
        self.dtype = dtype
        self.name = name
        self.family = family
        self.device = device
        self._dt = np.dtype(dtype)

    # host <-> device (identity for NumPy)
    def _to_dev(self, a):
        return self.xp.array(a, dtype=self._dt)  # always copies: callers' arrays are never mutated

    def _to_host(self, a) -> np.ndarray:
        if self.family == "cupy":
            return self.xp.asnumpy(a).astype(np.float64)
        return np.asarray(a, dtype=np.float64)

    def _sync(self) -> None:
        if self.family == "cupy":
            self.xp.cuda.Stream.null.synchronize()

    def nbody_accel(self, pos, mass, softening, g=1.0):
        check_softening(softening)
        p, m = self._to_dev(pos), self._to_dev(mass)
        out = accel_xp(self.xp, p, m, self._dt.type(softening**2), self._dt.type(g))
        self._sync()
        return self._to_host(out)

    def nbody_simulate(self, pos, vel, mass, softening, dt, steps, g=1.0):
        check_softening(softening)
        p, v, m = self._to_dev(pos), self._to_dev(vel), self._to_dev(mass)
        eps2, gg = self._dt.type(softening**2), self._dt.type(g)
        verlet(lambda x: accel_xp(self.xp, x, m, eps2, gg), p, v, dt, steps)
        self._sync()
        return self._to_host(p), self._to_host(v)

    def heat_solve(self, u0, r, steps):
        u = self._to_dev(u0)
        heat_ftcs(u, r, steps)
        self._sync()
        return self._to_host(u)

    def reset_peak_memory(self) -> None:
        if self.family == "cupy":
            self.xp.get_default_memory_pool().free_all_blocks()

    def peak_memory_mb(self):
        if self.family == "cupy":
            pool = self.xp.get_default_memory_pool()
            return pool.total_bytes() / 2**20, "cupy_pool_total"
        return None
