"""Backend interface, registry, and the backend-agnostic algorithm loops.

A backend turns float64 host arrays into its own device/dtype, runs a kernel and
returns float64 host arrays. Keeping the host<->device boundary inside the backend call
means every backend is timed on the same, end-to-end contract. The time-stepping
loops (`verlet`, `heat_ftcs`) use only operators that NumPy, CuPy and PyTorch share, so
the numerics are literally the same code across those backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class BackendUnavailable(RuntimeError):
    """Raised when a registered backend cannot run on this machine."""


class Backend(ABC):
    name: str
    dtype: str  # "float64" | "float32": the working precision of the backend
    device: str  # "cpu" | "mps" | "cuda"
    family: str  # "numpy" | "torch" | "cupy" | "numba"

    @abstractmethod
    def nbody_accel(self, pos: np.ndarray, mass: np.ndarray, softening: float, g: float = 1.0) -> np.ndarray:
        """Softened gravitational accelerations, shape (N, 3)."""

    @abstractmethod
    def nbody_simulate(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mass: np.ndarray,
        softening: float,
        dt: float,
        steps: int,
        g: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Velocity-Verlet integration; returns final (pos, vel)."""

    @abstractmethod
    def heat_solve(self, u0: np.ndarray, r: float, steps: int) -> np.ndarray:
        """FTCS steps of u_t = alpha*laplace(u), r = alpha*dt/dx^2, Dirichlet edges fixed."""

    # Optional device-memory hooks (None means "not available for this backend").
    def reset_peak_memory(self) -> None:  # pragma: no cover - trivial default
        return None

    def peak_memory_mb(self) -> tuple[float, str] | None:  # pragma: no cover
        return None


def check_softening(softening: float) -> float:
    """Softening must be > 0: with eps == 0 the i == j term evaluates 0 * inf = NaN."""
    if not softening > 0.0:
        raise ValueError(f"softening must be > 0 (got {softening})")
    return float(softening)


def verlet(accel: Callable, pos, vel, dt: float, steps: int):
    """Kick-drift-kick velocity Verlet (symplectic, 2nd order). In-place on pos/vel.

    Reuses the acceleration from the end of the previous step, so each step costs a
    single force evaluation.
    """
    half = 0.5 * dt
    a = accel(pos)
    for _ in range(steps):
        vel += half * a
        pos += dt * vel
        a = accel(pos)
        vel += half * a
    return pos, vel


def heat_ftcs(u, r: float, steps: int):
    """Explicit 5-point stencil, in place. Boundary nodes are never written (Dirichlet).

    The Laplacian is accumulated into a temporary *before* it is added to the interior,
    so reading and writing the same array is safe.
    """
    for _ in range(steps):
        c = u[1:-1, 1:-1]
        lap = u[2:, 1:-1] + u[:-2, 1:-1]
        lap += u[1:-1, 2:]
        lap += u[1:-1, :-2]
        lap -= 4.0 * c
        c += r * lap
    return u


# ------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class BackendInfo:
    name: str
    family: str
    dtype: str
    device: str
    available: bool
    reason: str  # why unavailable (empty when available)
    factory: Callable[[], Backend]


_REGISTRY: dict[str, BackendInfo] = {}


def register(info: BackendInfo) -> None:
    if info.name in _REGISTRY:
        raise ValueError(f"backend {info.name!r} registered twice")
    _REGISTRY[info.name] = info


def list_backends() -> list[BackendInfo]:
    return list(_REGISTRY.values())


def backend_names(available_only: bool = False) -> list[str]:
    return [b.name for b in _REGISTRY.values() if b.available or not available_only]


def get_info(name: str) -> BackendInfo:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(_REGISTRY)}") from None


def get_backend(name: str) -> Backend:
    info = get_info(name)
    if not info.available:
        raise BackendUnavailable(f"{name}: {info.reason}")
    return info.factory()
