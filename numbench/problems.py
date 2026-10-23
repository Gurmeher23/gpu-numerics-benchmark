"""Problem definitions: initial conditions, diagnostics and analytic references.

Everything here is computed in float64 on the host, independent of any backend, so it
can be used as ground truth for the accuracy tests and for the job service.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- N-body


@dataclass(frozen=True)
class NBodyIC:
    pos: np.ndarray  # (N, 3) float64
    vel: np.ndarray  # (N, 3) float64
    mass: np.ndarray  # (N,) float64
    softening: float
    g: float
    dt: float


def make_nbody_ic(n: int, seed: int = 0, softening: float = 0.05, dt: float = 0.005) -> NBodyIC:
    """Gaussian blob of N equal masses (total mass 1, G=1), roughly virialised.

    Centre of mass and total momentum are removed so that momentum conservation is
    a meaningful check. Plummer softening keeps close encounters well resolved by dt.
    """
    rng = np.random.default_rng(seed)
    pos = rng.normal(0.0, 1.0, size=(n, 3))
    vel = rng.normal(0.0, 0.3, size=(n, 3))
    mass = np.full(n, 1.0 / n)
    pos -= pos.mean(axis=0)
    vel -= vel.mean(axis=0)
    return NBodyIC(pos, vel, mass, softening, 1.0, dt)


def total_energy(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, softening: float, g: float) -> float:
    """Kinetic + Plummer-softened potential energy in float64.

    The softened pair potential -G m_i m_j / sqrt(r^2 + eps^2) is exactly the potential
    whose gradient is the force the kernels compute, so velocity Verlet conserves it
    up to O(dt^2) fluctuations.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    kinetic = 0.5 * float(np.sum(mass * np.sum(vel * vel, axis=1)))
    diff = pos[None, :, :] - pos[:, None, :]
    r2 = np.sum(diff * diff, axis=-1) + softening**2
    pair = mass[:, None] * mass[None, :] / np.sqrt(r2)
    np.fill_diagonal(pair, 0.0)
    potential = -0.5 * g * float(pair.sum())
    return kinetic + potential


def total_momentum(vel: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(mass, dtype=np.float64)[:, None] * np.asarray(vel, dtype=np.float64), axis=0)


# ------------------------------------------------------------------------------ Heat

HEAT_SIGMA0 = 0.05
HEAT_ALPHA = 1.0
HEAT_R = 0.2  # alpha*dt/dx^2; FTCS in 2D is stable for r <= 0.25


def heat_grid(n: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Unit square sampled on an n x n node grid (dx = 1/(n-1))."""
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return xx, yy, 1.0 / (n - 1)


def heat_analytic(n: int, t: float, alpha: float = HEAT_ALPHA, sigma0: float = HEAT_SIGMA0) -> np.ndarray:
    """Exact free-space solution for a Gaussian initial condition.

    u(x, y, t) = s0^2 / (s0^2 + 2 alpha t) * exp(-r^2 / (2 (s0^2 + 2 alpha t)))

    The Gaussian is centred in the unit square and stays ~10 sigma from the boundary for
    the times used here, so homogeneous Dirichlet conditions agree with free space to
    far better than the discretisation error being tested.
    """
    xx, yy, _ = heat_grid(n)
    var = sigma0**2 + 2.0 * alpha * t
    r2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
    return (sigma0**2 / var) * np.exp(-r2 / (2.0 * var))


def heat_setup(n: int, t_end: float, r: float = HEAT_R, alpha: float = HEAT_ALPHA):
    """Return (u0, steps, t_actual) for the fixed-r FTCS scheme reaching ~t_end.

    dt = r dx^2 / alpha, so refining the grid refines dt quadratically and the global
    error is O(dx^2) (the O(dt) term is proportional to dx^2 at fixed r).
    """
    _, _, dx = heat_grid(n)
    dt = r * dx * dx / alpha
    steps = max(1, round(t_end / dt))
    return heat_analytic(n, 0.0, alpha), steps, steps * dt


def default_heat_ic(n: int) -> np.ndarray:
    return heat_analytic(n, 0.0)
