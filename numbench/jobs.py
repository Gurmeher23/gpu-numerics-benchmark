"""Job specification and on-disk job store.

Layout of a run directory (`<runs_root>/<job_id>/`):

    spec.json      validated input
    manifest.json  inputs + git commit + environment hash (reproducibility record)
    status.json    {state, progress, message, timestamps, scheduler_job_id, error}
    result.json / result.csv   outputs (state == done)
    job.sbatch / slurm-<id>.out   (SLURM executor only)

The store is plain files so the same run directory works whether the job ran in an API
thread, or on a compute node via sbatch writing to a shared filesystem.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .backends import get_info
from .manifest import build_manifest

JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
MAX_SIZE = {"nbody": 4096, "heat": 4096}


class JobState(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["benchmark", "simulation"] = "benchmark"
    kernel: Literal["nbody", "heat"]
    backends: list[str] = Field(default_factory=lambda: ["numpy64"], min_length=1, max_length=12)
    sizes: list[int] = Field(min_length=1, max_length=8)
    steps: int | None = Field(default=None, ge=1, le=100_000)
    repeats: int = Field(default=5, ge=1, le=50)
    warmup: int = Field(default=1, ge=0, le=10)
    seed: int = 0

    @field_validator("backends")
    @classmethod
    def _known_backends(cls, v: list[str]) -> list[str]:
        for name in v:
            try:
                get_info(name)
            except KeyError as exc:
                raise ValueError(str(exc)) from None
        return v

    @model_validator(mode="after")
    def _check(self) -> JobSpec:
        limit = MAX_SIZE[self.kernel]
        if any(s < 8 or s > limit for s in self.sizes):
            raise ValueError(f"sizes must be in [8, {limit}] for kernel {self.kernel!r}")
        if self.kind == "simulation" and (len(self.backends) != 1 or len(self.sizes) != 1):
            raise ValueError("a simulation job takes exactly one backend and one size")
        return self


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)  # readers never see a half-written file


def read_status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text())


def update_status(run_dir: Path, **fields: Any) -> dict:
    status = read_status(run_dir)
    status.update(fields)
    _atomic_write_json(run_dir / "status.json", status)
    return status


class JobStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.match(job_id):
            raise KeyError(job_id)
        d = self.root / job_id
        if not (d / "status.json").exists():
            raise KeyError(job_id)
        return d

    def create(self, spec: JobSpec, executor_name: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        d = self.root / job_id
        d.mkdir(parents=True)
        payload = spec.model_dump()
        _atomic_write_json(d / "spec.json", payload)
        _atomic_write_json(d / "manifest.json", build_manifest(job_id, payload))
        _atomic_write_json(
            d / "status.json",
            {
                "job_id": job_id, "state": JobState.queued.value, "progress": 0.0, "message": "queued",
                "created": now_iso(), "started": None, "finished": None, "error": None,
                "executor": executor_name, "scheduler_job_id": None,
            },
        )  # fmt: skip
        return job_id

    def status(self, job_id: str) -> dict:
        return read_status(self.run_dir(job_id))

    def list_ids(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if JOB_ID_RE.match(p.name))
