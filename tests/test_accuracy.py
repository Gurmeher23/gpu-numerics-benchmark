"""Every backend against the float64 NumPy reference, with explicit per-dtype tolerances."""

import numpy as np

from numbench.problems import heat_setup, make_nbody_ic
from numbench.tolerances import is_close, scaled_max_error, tolerance_for


def test_accel_matches_reference(backend, reference):
    ic = make_nbody_ic(192, seed=3)
    got = backend.nbody_accel(ic.pos, ic.mass, ic.softening, ic.g)
    want = reference.nbody_accel(ic.pos, ic.mass, ic.softening, ic.g)
    assert got.shape == want.shape == (192, 3)
    assert is_close(got, want, tolerance_for(backend.dtype)), scaled_max_error(got, want)


def test_accel_does_not_mutate_inputs(backend):
    ic = make_nbody_ic(32, seed=5)
    before = (ic.pos.copy(), ic.mass.copy())
    backend.nbody_accel(ic.pos, ic.mass, ic.softening)
    assert np.array_equal(ic.pos, before[0]) and np.array_equal(ic.mass, before[1])


def test_simulation_matches_reference(backend, reference):
    ic = make_nbody_ic(128, seed=4)
    args = (ic.mass, ic.softening, ic.dt, 50, ic.g)
    p, v = backend.nbody_simulate(ic.pos, ic.vel, *args)
    pr, vr = reference.nbody_simulate(ic.pos, ic.vel, *args)
    tol = tolerance_for(backend.dtype, multi_step=True)
    assert is_close(p, pr, tol), scaled_max_error(p, pr)
    assert is_close(v, vr, tol), scaled_max_error(v, vr)


def test_heat_matches_reference(backend, reference):
    u0, steps, _ = heat_setup(65, 1e-3)
    got = backend.heat_solve(u0, 0.2, steps)
    want = reference.heat_solve(u0, 0.2, steps)
    assert is_close(got, want, tolerance_for(backend.dtype, multi_step=True)), scaled_max_error(got, want)


def test_heat_keeps_dirichlet_boundary(backend):
    u0, steps, _ = heat_setup(33, 1e-3)
    u0 = u0.copy()
    u0[0, :] = 0.25  # distinct, non-zero boundary data must be preserved exactly
    got = backend.heat_solve(u0, 0.2, steps)
    assert np.allclose(got[0, :], 0.25) and np.allclose(got[:, -1], u0[:, -1])


def test_float32_tolerance_is_looser_than_float64_for_good_reason():
    """float32 really is less accurate than float64: the two tolerances are not arbitrary."""
    from numbench.backends import get_backend

    ic = make_nbody_ic(192, seed=3)
    ref = get_backend("numpy64").nbody_accel(ic.pos, ic.mass, ic.softening)
    err32 = scaled_max_error(get_backend("numpy32").nbody_accel(ic.pos, ic.mass, ic.softening), ref)
    assert 1e-9 < err32 < tolerance_for("float32").atol_scale  # visibly float32, but inside tolerance
    assert tolerance_for("float64").atol_scale < err32  # a float32 result would fail the float64 bar
