"""Benchmark harness: warm-up, repeats, median/IQR timing, peak memory, accuracy.

Each case is timed end to end (host input -> device -> compute -> host output), which is
what a scientist calling the kernel actually experiences. Every row also carries the
error of the final result against the float64 NumPy reference, so a fast-but-wrong
backend cannot hide in the table.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
import tracemalloc
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from .backends import Backend, get_backend, get_info
from .problems import default_heat_ic, make_nbody_ic
from .tolerances import scaled_max_error

REFERENCE = "numpy64"
DEFAULT_SIZES = {"nbody": [256, 512, 1024, 2048], "heat": [128, 256, 512, 1024]}
DEFAULT_STEPS = {"nbody": 10, "heat": 200}
QUICK_SIZES = {"nbody": [64, 128], "heat": [32, 64]}
CSV_FIELDS = [
    "kernel", "backend", "dtype", "device", "size", "steps", "repeats",
    "median_s", "q1_s", "q3_s", "iqr_s", "min_s", "speedup_vs_numpy64",
    "munits_per_s", "peak_mem_mb", "mem_source", "rel_err_vs_ref",
]  # fmt: skip


def _make_runner(kernel: str, size: int, steps: int, seed: int) -> tuple[Callable[[Backend], np.ndarray], int]:
    """Return (fn(backend)->final field, work_units). One work unit is one pair
    interaction (nbody) or one interior-cell update (heat)."""
    if kernel == "nbody":
        ic = make_nbody_ic(size, seed)

        def run(b: Backend) -> np.ndarray:
            pos, _ = b.nbody_simulate(ic.pos, ic.vel, ic.mass, ic.softening, ic.dt, steps, ic.g)
            return pos

        return run, size * size * (steps + 1)
    if kernel == "heat":
        u0 = default_heat_ic(size)

        def run(b: Backend) -> np.ndarray:
            return b.heat_solve(u0, 0.2, steps)

        return run, (size - 2) ** 2 * steps
    raise ValueError(f"unknown kernel {kernel!r}")


def _peak_memory(backend: Backend, fn: Callable[[Backend], np.ndarray]) -> tuple[float, str]:
    """One untimed run to measure memory (tracemalloc slows execution)."""
    backend.reset_peak_memory()
    if backend.family == "numpy":
        tracemalloc.start()
        try:
            fn(backend)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak / 2**20, "tracemalloc"
    fn(backend)
    got = backend.peak_memory_mb()
    if got is None:
        return float("nan"), "unavailable"
    return got


def time_case(fn: Callable[[Backend], np.ndarray], backend: Backend, warmup: int, repeats: int):
    for _ in range(warmup):
        out = fn(backend)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(backend)
        times.append(time.perf_counter() - t0)
    return times, out


def summarize(times: list[float]) -> dict[str, float]:
    q1, _, q3 = (float(x) for x in np.percentile(times, [25, 50, 75]))
    return {
        "median_s": statistics.median(times),
        "q1_s": q1,
        "q3_s": q3,
        "iqr_s": q3 - q1,
        "min_s": min(times),
    }


def run_benchmark(
    kernels: Iterable[str],
    backends: Iterable[str],
    sizes: dict[str, list[int]] | None = None,
    steps: dict[str, int] | None = None,
    repeats: int = 7,
    warmup: int = 2,
    seed: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    """Run all (kernel, size, backend) cases. numpy64 is always run as the reference."""
    kernels = list(kernels)
    names = [REFERENCE] + [b for b in dict.fromkeys(backends) if b != REFERENCE]
    sizes = {**DEFAULT_SIZES, **(sizes or {})}
    steps = {**DEFAULT_STEPS, **(steps or {})}
    total = sum(len(sizes[k]) * len(names) for k in kernels)
    done = 0
    rows: list[dict] = []
    instances = {n: get_backend(n) for n in names}  # fails fast with the unavailable reason
    for kernel in kernels:
        for size in sizes[kernel]:
            fn, units = _make_runner(kernel, size, steps[kernel], seed)
            ref_out = ref_median = None
            for name in names:
                b = instances[name]
                times, out = time_case(fn, b, warmup, repeats)
                stats = summarize(times)
                if name == REFERENCE:
                    ref_out, ref_median = out, stats["median_s"]
                mem, src = _peak_memory(b, fn)
                info = get_info(name)
                rows.append(
                    {
                        "kernel": kernel, "backend": name, "dtype": info.dtype, "device": info.device,
                        "size": size, "steps": steps[kernel], "repeats": repeats, **stats,
                        "speedup_vs_numpy64": ref_median / stats["median_s"],
                        "munits_per_s": units / stats["median_s"] / 1e6,
                        "peak_mem_mb": mem, "mem_source": src,
                        "rel_err_vs_ref": scaled_max_error(out, ref_out),
                    }
                )  # fmt: skip
                done += 1
                if progress:
                    progress(done, total, f"{kernel} n={size} {name}")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in CSV_FIELDS})


def read_csv(path: Path) -> list[dict]:
    with Path(path).open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in CSV_FIELDS:
            if k in ("kernel", "backend", "dtype", "device", "mem_source"):
                continue
            r[k] = float(r[k]) if k not in ("size", "steps", "repeats") else int(float(r[k]))
    return rows


def plot_speedup(rows: list[dict], path: Path) -> None:
    """2x2 figure: median runtime (top) and speedup over numpy64 (bottom), per kernel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kernels = [k for k in ("nbody", "heat") if any(r["kernel"] == k for r in rows)]
    backends = list(dict.fromkeys(r["backend"] for r in rows))
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#7F7F7F", "#000000"]
    styles = {b: (palette[i % len(palette)], "osv^Dpx*"[i % 8]) for i, b in enumerate(backends)}
    titles = {"nbody": "N-body (velocity Verlet)", "heat": "2D heat equation (FTCS)"}
    xl = {"nbody": "particles N", "heat": "grid points per side"}
    fig, axes = plt.subplots(2, len(kernels), figsize=(6.2 * len(kernels), 7.4), squeeze=False)
    for col, k in enumerate(kernels):
        for b in backends:
            pts = sorted(
                (r["size"], r["median_s"], r["speedup_vs_numpy64"])
                for r in rows
                if r["kernel"] == k and r["backend"] == b
            )
            if not pts:
                continue
            xs, ts, sp = zip(*pts, strict=True)
            c, m = styles[b]
            axes[0, col].plot(xs, ts, marker=m, color=c, label=b)
            axes[1, col].plot(xs, sp, marker=m, color=c, label=b)
        axes[0, col].set_title(titles[k])
        axes[0, col].set_ylabel("median time per call (s)")
        axes[1, col].set_ylabel("speedup vs numpy64 (x)")
        axes[1, col].axhline(1.0, color="0.5", lw=0.8, ls="--")
        for row in (0, 1):
            ax = axes[row, col]
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel(xl[k])
            ax.grid(True, which="both", alpha=0.25)
        axes[0, col].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def default_backends() -> list[str]:
    from .backends import list_backends

    skip = {"torch-auto-f32"}  # alias of whichever device wins auto-detection
    return [b.name for b in list_backends() if b.available and b.name not in skip]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="numbench benchmark harness")
    ap.add_argument("--kernel", choices=["nbody", "heat", "both"], default="both")
    ap.add_argument("--backends", nargs="*", help="default: every available backend")
    ap.add_argument("--sizes", nargs="*", type=int, help="override sizes (one kernel only)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="tiny sizes for CI smoke runs")
    ap.add_argument("--torch-threads", type=int, help="torch.set_num_threads(N) for the torch CPU backends")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    if args.torch_threads:
        import torch

        torch.set_num_threads(args.torch_threads)

    kernels = ["nbody", "heat"] if args.kernel == "both" else [args.kernel]
    sizes = dict(QUICK_SIZES) if args.quick else {}
    steps = {"nbody": 3, "heat": 20} if args.quick else {}
    if args.sizes:
        if len(kernels) != 1:
            ap.error("--sizes requires --kernel nbody|heat")
        sizes[kernels[0]] = args.sizes
    if args.steps:
        steps = {**steps, **{k: args.steps for k in kernels}}
    rows = run_benchmark(
        kernels,
        args.backends or default_backends(),
        sizes,
        steps,
        repeats=3 if args.quick else args.repeats,
        warmup=1 if args.quick else args.warmup,
        progress=lambda i, n, msg: print(f"[{i}/{n}] {msg}", flush=True),
    )
    write_csv(rows, args.out / "benchmark.csv")
    plot_speedup(rows, args.out / "speedup.png")
    print(f"wrote {args.out / 'benchmark.csv'} and {args.out / 'speedup.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
