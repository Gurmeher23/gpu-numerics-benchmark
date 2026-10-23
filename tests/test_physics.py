"""Physics-level checks: conservation laws, analytic solution, convergence order."""

import numpy as np
import pytest

from numbench.backends import get_backend
from numbench.problems import (
    HEAT_R,
    heat_analytic,
    heat_grid,
    heat_setup,
    make_nbody_ic,
    total_energy,
    total_momentum,
)

# Maximum |dE/E| allowed after 200 steps (t = 1.0, about one crossing time).
# Measured: ~5e-7 in both float64 and float32 for this setup, so the limits leave >100x margin
# while still failing on a wrong force law or a non-symplectic integrator.
ENERGY_DRIFT_LIMIT = {"float64": 1e-4, "float32": 5e-4}


def test_energy_conserved_by_velocity_verlet(backend):
    ic = make_nbody_ic(96, seed=11)
    e0 = total_energy(ic.pos, ic.vel, ic.mass, ic.softening, ic.g)
    p, v = backend.nbody_simulate(ic.pos, ic.vel, ic.mass, ic.softening, ic.dt, 200, ic.g)
    e1 = total_energy(p, v, ic.mass, ic.softening, ic.g)
    assert abs(e1 - e0) / abs(e0) < ENERGY_DRIFT_LIMIT[backend.dtype]


def test_momentum_conserved(backend):
    # Pairwise forces are equal and opposite, so total momentum is conserved to rounding.
    ic = make_nbody_ic(96, seed=12)
    _, v = backend.nbody_simulate(ic.pos, ic.vel, ic.mass, ic.softening, ic.dt, 100, ic.g)
    p0 = np.linalg.norm(total_momentum(ic.vel, ic.mass))
    p1 = np.linalg.norm(total_momentum(v, ic.mass))
    assert p0 < 1e-15 and p1 < {"float64": 1e-12, "float32": 1e-5}[backend.dtype]


def test_verlet_is_second_order():
    """Halving dt cuts the energy fluctuation by ~4x: evidence the integrator is 2nd order,
    not merely that the energy happens to look fine."""
    ref = get_backend("numpy64")
    ic = make_nbody_ic(48, seed=2)
    amplitude = []
    for dt in (0.01, 0.005):
        steps = round(0.5 / dt)
        e0 = total_energy(ic.pos, ic.vel, ic.mass, ic.softening, ic.g)
        worst = 0.0
        p, v = ic.pos.copy(), ic.vel.copy()
        for _ in range(5):  # sample the fluctuation along the trajectory
            p, v = ref.nbody_simulate(p, v, ic.mass, ic.softening, dt, steps // 5, ic.g)
            worst = max(worst, abs(total_energy(p, v, ic.mass, ic.softening, ic.g) - e0) / abs(e0))
        amplitude.append(worst)
    ratio = amplitude[0] / amplitude[1]
    assert 2.5 < ratio < 6.0, amplitude


def test_two_body_force_matches_closed_form(reference):
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    mass = np.array([2.0, 5.0])
    a = reference.nbody_accel(pos, mass, softening=1e-6, g=1.0)
    # r = 5; |a_0| = G m_1 / r^2 = 5/25 toward +(3,4)/5
    np.testing.assert_allclose(a[0], 0.2 * np.array([0.6, 0.8, 0.0]), rtol=1e-12)
    np.testing.assert_allclose(a[1], -(2.0 / 25.0) * np.array([0.6, 0.8, 0.0]), rtol=1e-12)


def test_heat_matches_analytic_gaussian(backend):
    n = 129
    u0, steps, t = heat_setup(n, 1e-3)
    u = backend.heat_solve(u0, HEAT_R, steps)
    exact = heat_analytic(n, t)
    # peak decays 1.0 -> 0.555; measured max error at n=129 is 5.0e-4 (O(dx^2) truncation error)
    assert np.max(np.abs(u - exact)) < 1e-3
    assert u.max() < u0.max()  # diffusion lowers the peak
    assert u.min() >= -1e-6 * u0.max()  # discrete maximum principle (r <= 1/4)


def test_heat_grid_refinement_converges_second_order(reference):
    """With r fixed, dt ~ dx^2 and the FTCS error is O(dx^2): errors drop ~4x per refinement."""
    errs, hs = [], []
    for n in (33, 65, 129):
        u0, steps, t = heat_setup(n, 1e-3)
        u = reference.heat_solve(u0, HEAT_R, steps)
        errs.append(np.max(np.abs(u - heat_analytic(n, t))))
        hs.append(heat_grid(n)[2])
    orders = [np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1]) for i in range(2)]
    assert errs[0] > errs[1] > errs[2]
    assert all(1.7 < o < 2.3 for o in orders), (errs, orders)


def test_heat_conserves_mass_away_from_boundary(reference):
    n = 129
    u0, steps, _ = heat_setup(n, 1e-3)
    u = reference.heat_solve(u0, HEAT_R, steps)
    assert abs(u.sum() - u0.sum()) / u0.sum() < 1e-8


@pytest.mark.parametrize("r", [0.3])
def test_unstable_step_ratio_blows_up(reference, r):
    """Sanity check on the stability limit the docs cite (r <= 0.25 in 2D)."""
    u0, _, _ = heat_setup(33, 1e-3)
    u = reference.heat_solve(u0, r, 400)
    assert np.max(np.abs(u)) > 10 * u0.max()


def test_zero_softening_is_rejected(backend):
    # eps == 0 would make the self-interaction 0 * inf = NaN, so it is refused up front.
    ic = make_nbody_ic(8)
    with pytest.raises(ValueError, match="softening"):
        backend.nbody_accel(ic.pos, ic.mass, 0.0)
