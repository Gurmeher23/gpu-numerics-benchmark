"""PyTorch backend with device selection (cpu / mps / cuda) and auto-detection.

CPU and Apple-MPS results are measured on this repository's development machine.
The CUDA branch uses only standard torch.cuda calls and is NOT executed locally.
"""

from __future__ import annotations

import numpy as np

from .base import Backend, BackendUnavailable, check_softening, heat_ftcs, verlet


def detect_device() -> str:
    """cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_unavailable_reason(device: str) -> str:
    """Empty string when `device` can be used."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "PyTorch is not installed"
    if device == "cpu":
        return ""
    if device == "cuda":
        return "" if torch.cuda.is_available() else "torch.cuda.is_available() is False (no NVIDIA GPU/driver)"
    if device == "mps":
        ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        return "" if ok else "torch.backends.mps.is_available() is False (not Apple Silicon / macOS 12.3+)"
    return f"unknown device {device!r}"


class TorchBackend(Backend):
    family = "torch"

    def __init__(self, device: str, dtype: str, name: str):
        import torch

        if device == "auto":
            device = detect_device()
        reason = device_unavailable_reason(device)
        if reason:
            raise BackendUnavailable(f"torch/{device}: {reason}")
        if device == "mps" and dtype == "float64":
            raise BackendUnavailable("torch/mps: float64 is not supported by the MPS backend")
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.name = name
        self._tdt = {"float32": torch.float32, "float64": torch.float64}[dtype]
        self._npdt = np.dtype(dtype)

    def _to_dev(self, a):
        t = self.torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))
        return t.to(device=self.device, dtype=self._tdt, copy=True)

    def _to_host(self, t) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float64)  # .cpu() synchronises

    def _accel(self, pos, mass, eps2, g):
        diff = pos[None, :, :] - pos[:, None, :]
        r2 = (diff * diff).sum(dim=-1) + eps2
        w = mass[None, :] * r2.pow(-1.5)
        return g * (w.unsqueeze(-1) * diff).sum(dim=1)

    def nbody_accel(self, pos, mass, softening, g=1.0):
        check_softening(softening)
        with self.torch.no_grad():
            p, m = self._to_dev(pos), self._to_dev(mass)
            return self._to_host(self._accel(p, m, softening**2, g))

    def nbody_simulate(self, pos, vel, mass, softening, dt, steps, g=1.0):
        check_softening(softening)
        with self.torch.no_grad():
            p, v, m = self._to_dev(pos), self._to_dev(vel), self._to_dev(mass)
            eps2 = softening**2
            verlet(lambda x: self._accel(x, m, eps2, g), p, v, dt, steps)
            return self._to_host(p), self._to_host(v)

    def heat_solve(self, u0, r, steps):
        with self.torch.no_grad():
            u = self._to_dev(u0)
            heat_ftcs(u, r, steps)
            return self._to_host(u)

    def reset_peak_memory(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def peak_memory_mb(self):
        if self.device == "cuda":
            return self.torch.cuda.max_memory_allocated() / 2**20, "cuda_max_allocated"
        if self.device == "mps":
            # MPS exposes no peak counter; this is the driver's current allocation.
            return self.torch.mps.driver_allocated_memory() / 2**20, "mps_driver_allocated"
        return None
