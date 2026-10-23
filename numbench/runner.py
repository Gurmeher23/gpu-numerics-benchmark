"""Worker: executes one run directory. Shared by the thread and SLURM executors.

python -m numbench.runner <run_dir>
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .backends import get_backend
from .bench import read_csv, run_benchmark, write_csv  # noqa: F401  (read_csv re-exported for tests)
from .jobs import JobSpec, JobState, now_iso, update_status
from .problems import HEAT_R, heat_analytic, heat_grid, make_nbody_ic, total_energy, total_momentum


def run_simulation(spec: JobSpec) -> dict:
    """One backend, one size: run the kernel once and report physical diagnostics."""
    name, size = spec.backends[0], spec.sizes[0]
    b = get_backend(name)
    ref = get_backend("numpy64")
    t0 = time.perf_counter()
    if spec.kernel == "nbody":
        steps = spec.steps or 200
        ic = make_nbody_ic(size, spec.seed)
        pos, vel = b.nbody_simulate(ic.pos, ic.vel, ic.mass, ic.softening, ic.dt, steps, ic.g)
        elapsed = time.perf_counter() - t0
        e0 = total_energy(ic.pos, ic.vel, ic.mass, ic.softening, ic.g)
        e1 = total_energy(pos, vel, ic.mass, ic.softening, ic.g)
        rpos, _ = ref.nbody_simulate(ic.pos, ic.vel, ic.mass, ic.softening, ic.dt, steps, ic.g)
        scale = float(np.max(np.abs(rpos)))
        return {
            "kernel": "nbody", "backend": name, "n": size, "steps": steps, "dt": ic.dt, "elapsed_s": elapsed,
            "energy_initial": e0, "energy_final": e1, "energy_rel_drift": abs(e1 - e0) / abs(e0),
            "momentum_norm": float(np.linalg.norm(total_momentum(vel, ic.mass))),
            "pos_max_err_vs_float64_ref": float(np.max(np.abs(pos - rpos)) / scale),
        }  # fmt: skip
    steps = spec.steps or 200
    _, _, dx = heat_grid(size)
    dt = HEAT_R * dx * dx
    t_end = steps * dt
    u0 = heat_analytic(size, 0.0)
    u = b.heat_solve(u0, HEAT_R, steps)
    elapsed = time.perf_counter() - t0
    exact = heat_analytic(size, t_end)
    return {
        "kernel": "heat", "backend": name, "grid": size, "steps": steps, "t_end": t_end, "elapsed_s": elapsed,
        "linf_error_vs_analytic": float(np.max(np.abs(u - exact))),
        "l2_error_vs_analytic": float(np.sqrt(np.mean((u - exact) ** 2))),
    }  # fmt: skip


def run_job(run_dir: Path | str) -> None:
    """Never raises: failures are recorded in status.json (state=failed + traceback tail)."""
    run_dir = Path(run_dir)
    try:
        spec = JobSpec.model_validate_json((run_dir / "spec.json").read_text())
        update_status(run_dir, state=JobState.running.value, started=now_iso(), message="running")
        if spec.kind == "benchmark":

            def progress(done: int, total: int, msg: str) -> None:
                update_status(run_dir, progress=round(done / total, 4), message=msg)

            steps = {spec.kernel: spec.steps} if spec.steps else None
            rows = run_benchmark(
                [spec.kernel], spec.backends, {spec.kernel: spec.sizes}, steps,
                repeats=spec.repeats, warmup=spec.warmup, seed=spec.seed, progress=progress,
            )  # fmt: skip
            write_csv(rows, run_dir / "result.csv")
            (run_dir / "result.json").write_text(json.dumps({"kind": "benchmark", "rows": rows}, indent=2))
        else:
            diag = run_simulation(spec)
            (run_dir / "result.json").write_text(json.dumps({"kind": "simulation", "diagnostics": diag}, indent=2))
            keys = list(diag)
            (run_dir / "result.csv").write_text(",".join(keys) + "\n" + ",".join(str(diag[k]) for k in keys) + "\n")
        update_status(run_dir, state=JobState.done.value, progress=1.0, finished=now_iso(), message="done")
    except Exception as exc:
        tail = "".join(traceback.format_exception(exc)[-3:])
        try:
            update_status(
                run_dir,
                state=JobState.failed.value,
                finished=now_iso(),
                error=f"{type(exc).__name__}: {exc}",
                message=tail,
            )
        except Exception:  # status file itself unreadable: nothing more we can record
            print(tail, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m numbench.runner <run_dir>", file=sys.stderr)
        return 2
    run_job(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
