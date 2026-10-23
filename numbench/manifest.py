"""Reproducibility manifest: inputs, code version and environment fingerprint."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata as md
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

KEY_PACKAGES = ("numpy", "torch", "numba", "cupy-cuda12x", "fastapi", "pydantic", "matplotlib")


@lru_cache(maxsize=1)
def environment_hash() -> str:
    """SHA-256 over the sorted `name==version` list of every installed distribution.

    Two runs with the same hash executed against the same package set; a differing hash
    explains numerical differences (e.g. a different BLAS or torch build).
    """
    pkgs = sorted(f"{d.metadata['Name'].lower()}=={d.version}" for d in md.distributions() if d.metadata["Name"])
    return hashlib.sha256("\n".join(pkgs).encode()).hexdigest()


def git_info(repo_dir: Path | None = None) -> dict:
    """Commit of the numbench source tree. NUMBENCH_GIT_COMMIT overrides it (useful in
    container images built without a .git directory)."""
    override = os.environ.get("NUMBENCH_GIT_COMMIT")
    if override:
        return {"commit": override, "dirty": None}
    cwd = repo_dir or Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "."],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def package_versions() -> dict[str, str]:
    out = {}
    for name in KEY_PACKAGES:
        with contextlib.suppress(md.PackageNotFoundError):
            out[name] = md.version(name)
    return out


def build_manifest(job_id: str, spec: dict) -> dict:
    return {
        "job_id": job_id,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": spec,
        "git": git_info(),
        "environment": {
            "env_hash": environment_hash(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": package_versions(),
        },
    }
