"""FastAPI job service.

    POST /jobs                 submit a JobSpec            -> 202 {job_id, ...}
    GET  /jobs                 list job ids
    GET  /jobs/{id}            status + progress
    GET  /jobs/{id}/result     ?format=json|csv (409 until the job is done)
    GET  /jobs/{id}/manifest   reproducibility manifest
    GET  /backends             availability of every backend, with reasons

Run:  NUMBENCH_EXECUTOR=local|slurm  NUMBENCH_RUNS_DIR=runs  uvicorn numbench.api:app
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

from . import __version__
from .backends import list_backends
from .executors import Executor, ExecutorUnavailable, LocalThreadExecutor, SlurmConfig, SlurmExecutor
from .jobs import JOB_ID_RE, JobSpec, JobState, JobStore

JobIdPath = PathParam(..., pattern=JOB_ID_RE.pattern, description="12-hex-character job id")


def executor_from_env() -> Executor:
    kind = os.environ.get("NUMBENCH_EXECUTOR", "local")
    if kind == "local":
        return LocalThreadExecutor()
    if kind == "slurm":
        return SlurmExecutor(
            SlurmConfig(
                partition=os.environ.get("NUMBENCH_SLURM_PARTITION"),
                gpus=int(os.environ.get("NUMBENCH_SLURM_GPUS", "0")),
                time_limit=os.environ.get("NUMBENCH_SLURM_TIME", "00:30:00"),
            )
        )
    raise ValueError(f"NUMBENCH_EXECUTOR must be 'local' or 'slurm', got {kind!r}")


def create_app(store: JobStore | None = None, executor: Executor | None = None) -> FastAPI:
    store = store or JobStore(os.environ.get("NUMBENCH_RUNS_DIR", "runs"))
    executor = executor or executor_from_env()
    app = FastAPI(title="numbench job service", version=__version__)

    def _dir(job_id: str) -> Path:
        try:
            return store.run_dir(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}") from None

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "executor": executor.name, "version": __version__}

    @app.get("/backends")
    def backends() -> list[dict]:
        return [
            {"name": b.name, "family": b.family, "dtype": b.dtype, "device": b.device,
             "available": b.available, "reason": b.reason}
            for b in list_backends()
        ]  # fmt: skip

    @app.post("/jobs", status_code=202)
    def submit(spec: JobSpec) -> dict:
        info = {b.name: b for b in list_backends()}
        unavailable = {n: info[n].reason for n in spec.backends if not info[n].available}
        if unavailable:
            raise HTTPException(status_code=422, detail={"unavailable_backends": unavailable})
        job_id = store.create(spec, executor.name)
        try:
            executor.submit(store, job_id)
        except ExecutorUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        return {"job_id": job_id, "state": JobState.queued.value, "status_url": f"/jobs/{job_id}"}

    @app.get("/jobs")
    def list_jobs() -> list[str]:
        return store.list_ids()

    @app.get("/jobs/{job_id}")
    def status(job_id: str = JobIdPath) -> dict:
        d = _dir(job_id)
        executor.refresh(store, job_id)
        return store.status(job_id) | {"run_dir": str(d)}

    @app.get("/jobs/{job_id}/result")
    def result(job_id: str = JobIdPath, format: Literal["json", "csv"] = "json") -> FileResponse:
        d = _dir(job_id)
        executor.refresh(store, job_id)
        st = store.status(job_id)
        if st["state"] != JobState.done.value:
            raise HTTPException(status_code=409, detail={"state": st["state"], "error": st.get("error")})
        media = "application/json" if format == "json" else "text/csv"
        return FileResponse(d / f"result.{format}", media_type=media, filename=f"{job_id}.{format}")

    @app.get("/jobs/{job_id}/manifest")
    def manifest(job_id: str = JobIdPath) -> dict:
        return json.loads((_dir(job_id) / "manifest.json").read_text())

    return app


def __getattr__(name: str):  # lazy `app` so importing the module has no side effects
    if name == "app":
        return create_app()
    raise AttributeError(name)
