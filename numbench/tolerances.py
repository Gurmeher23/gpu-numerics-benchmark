"""Explicit, documented comparison tolerances.

All backends are compared to the float64 NumPy reference with
    |actual - ref| <= atol_scale * max|ref| + rtol * |ref|        (elementwise)

Why an `atol` tied to the field's scale: forces and temperatures contain entries that
are close to zero because positive and negative contributions cancel. A pure relative
test would then measure the rounding error of the *terms* (large) against a tiny
result and fail for reasons that are not bugs. Scaling the absolute tolerance by the
largest reference magnitude makes the test about the field as a whole.

Why float32 differs: float32 has a unit roundoff of ~6e-8 versus ~1.1e-16 for float64.
A reduction over N terms accumulates roughly sqrt(N) roundoff errors (worst case N), and
the inputs themselves are rounded when cast. The float32 values below leave a margin of
more than an order of magnitude over what is observed (see README table), whereas the
float64 values are ~7 orders of magnitude tighter and would catch any algorithmic error
or reordering bug.

`SIM_FACTOR` loosens the tolerance for multi-step runs, where rounding differences are
amplified by the dynamics before the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol_scale: float


TOLERANCES: dict[str, Tolerance] = {
    "float64": Tolerance(rtol=1e-10, atol_scale=1e-12),
    "float32": Tolerance(rtol=1e-4, atol_scale=1e-5),
}
SIM_FACTOR: dict[str, float] = {"float64": 100.0, "float32": 10.0}


def tolerance_for(dtype: str, multi_step: bool = False) -> Tolerance:
    tol = TOLERANCES[dtype]
    if multi_step:
        f = SIM_FACTOR[dtype]
        return Tolerance(tol.rtol * f, tol.atol_scale * f)
    return tol


def scaled_max_error(actual: np.ndarray, ref: np.ndarray) -> float:
    """max|a - ref| / max|ref|, the headline error number reported in tables."""
    ref = np.asarray(ref, dtype=np.float64)
    scale = float(np.max(np.abs(ref)))
    if scale == 0.0:
        scale = 1.0
    return float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - ref)) / scale)


def is_close(actual: np.ndarray, ref: np.ndarray, tol: Tolerance) -> bool:
    ref = np.asarray(ref, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    atol = tol.atol_scale * float(np.max(np.abs(ref)))
    return bool(np.all(np.abs(actual - ref) <= atol + tol.rtol * np.abs(ref)))
