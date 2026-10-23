"""Profiling helpers: cProfile (Python-level) and torch.profiler (operator-level).

python -m numbench.profile --kernel nbody --backend numpy64 --size 1024
python -m numbench.profile --kernel heat  --backend torch-cpu-f32 --size 512 --torch
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats

from .backends import get_backend
from .bench import DEFAULT_STEPS, _make_runner


def cprofile_report(kernel: str, backend: str, size: int, steps: int | None = None, top: int = 12) -> str:
    """Top functions by internal time. For vectorised code nearly all time lands in a few
    C-level calls (ufuncs, reductions); anything Python-side shows up as its own line."""
    steps = steps or DEFAULT_STEPS[kernel]
    b = get_backend(backend)
    fn, _ = _make_runner(kernel, size, steps, seed=0)
    fn(b)  # warm-up (JIT, allocator, lazy imports)
    prof = cProfile.Profile()
    prof.enable()
    fn(b)
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).strip_dirs().sort_stats("tottime").print_stats(top)
    return buf.getvalue()


def torch_profile_report(kernel: str, backend: str, size: int, steps: int | None = None, top: int = 10) -> str:
    """Operator-level table from torch.profiler (CPU always, CUDA activities when present)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    steps = steps or DEFAULT_STEPS[kernel]
    b = get_backend(backend)
    if b.family != "torch":
        raise ValueError("torch profiler needs a torch backend")
    fn, _ = _make_runner(kernel, size, steps, seed=0)
    fn(b)
    acts = [ProfilerActivity.CPU]
    sort_key = "self_cpu_time_total"
    if torch.cuda.is_available() and b.device == "cuda":
        acts.append(ProfilerActivity.CUDA)
        sort_key = "self_cuda_time_total"
    with profile(activities=acts, profile_memory=True) as prof:
        fn(b)
    return prof.key_averages().table(sort_by=sort_key, row_limit=top, max_name_column_width=40)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="numbench profiling")
    ap.add_argument("--kernel", choices=["nbody", "heat"], required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--torch", action="store_true", help="use torch.profiler instead of cProfile")
    args = ap.parse_args(argv)
    if args.torch:
        print(torch_profile_report(args.kernel, args.backend, args.size, args.steps))
    else:
        print(cprofile_report(args.kernel, args.backend, args.size, args.steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
