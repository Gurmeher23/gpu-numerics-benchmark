"""Backend registry. Importing this package registers every backend with its
availability (and the reason when unavailable) so tests and the CLI can report skips."""

from __future__ import annotations

from .base import (
    Backend,
    BackendInfo,
    BackendUnavailable,
    backend_names,
    get_backend,
    get_info,
    list_backends,
    register,
)


def _register_all() -> None:
    import numpy as np

    from .array_backend import ArrayBackend

    register(
        BackendInfo(
            "numpy64", "numpy", "float64", "cpu", True, "", lambda: ArrayBackend(np, "float64", "numpy64", "numpy")
        )
    )
    register(
        BackendInfo(
            "numpy32", "numpy", "float32", "cpu", True, "", lambda: ArrayBackend(np, "float32", "numpy32", "numpy")
        )
    )

    # ---- PyTorch: cpu f32/f64, mps f32, cuda f32/f64, auto f32
    try:
        import torch  # noqa: F401

        from .torch_backend import TorchBackend, detect_device, device_unavailable_reason

        def torch_entry(name: str, device: str, dtype: str) -> None:
            reason = "" if device == "auto" else device_unavailable_reason(device)
            shown = detect_device() if device == "auto" else device
            register(
                BackendInfo(name, "torch", dtype, shown, not reason, reason, lambda: TorchBackend(device, dtype, name))
            )

        torch_entry("torch-cpu-f32", "cpu", "float32")
        torch_entry("torch-cpu-f64", "cpu", "float64")
        torch_entry("torch-mps-f32", "mps", "float32")
        torch_entry("torch-cuda-f32", "cuda", "float32")
        torch_entry("torch-cuda-f64", "cuda", "float64")
        torch_entry("torch-auto-f32", "auto", "float32")
    except ImportError:
        for n in (
            "torch-cpu-f32",
            "torch-cpu-f64",
            "torch-mps-f32",
            "torch-cuda-f32",
            "torch-cuda-f64",
            "torch-auto-f32",
        ):
            register(BackendInfo(n, "torch", "float32", "?", False, "PyTorch is not installed", lambda: None))

    # ---- CuPy (optional, needs an NVIDIA GPU)
    cupy_reason = ""
    cp = None
    try:
        import cupy as cp  # type: ignore[no-redef]

        if cp.cuda.runtime.getDeviceCount() < 1:
            cupy_reason = "CuPy imported but no CUDA device found"
    except Exception as exc:  # ImportError, CUDARuntimeError, driver errors...
        cupy_reason = f"CuPy unavailable: {type(exc).__name__}: {exc}".splitlines()[0]
    for dtype, short in (("float32", "f32"), ("float64", "f64")):
        nm = f"cupy-{short}"
        register(
            BackendInfo(
                nm,
                "cupy",
                dtype,
                "cuda",
                not cupy_reason,
                cupy_reason,
                (lambda d=dtype, n=nm: ArrayBackend(cp, d, n, "cupy", "cuda")),
            )
        )

    # ---- Numba (optional)
    try:
        from .numba_backend import NumbaBackend

        register(BackendInfo("numba-f64", "numba", "float64", "cpu", True, "", NumbaBackend))
    except ImportError as exc:
        register(BackendInfo("numba-f64", "numba", "float64", "cpu", False, f"Numba unavailable: {exc}", lambda: None))


_register_all()

__all__ = [
    "Backend",
    "BackendInfo",
    "BackendUnavailable",
    "backend_names",
    "get_backend",
    "get_info",
    "list_backends",
]
