"""Pluggable job executors.

`LocalThreadExecutor` runs jobs in-process on one worker thread (benchmarks should not
compete with each other for the same cores, so jobs are serialised).

`SlurmExecutor` writes an sbatch script into the run directory and shells out to
`sbatch`/`squeue`. The compute node runs `python -m numbench.runner <run_dir>`, which
updates `status.json` on the shared filesystem; `refresh()` only has to notice jobs that
died without doing so (cancelled, timed out, OOM-killed, node failure).
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .jobs import JobState, JobStore, now_iso, read_status, update_status
from .runner import run_job


class ExecutorUnavailable(RuntimeError):
    pass


class Executor(Protocol):
    name: str

    def submit(self, store: JobStore, job_id: str) -> None: ...

    def refresh(self, store: JobStore, job_id: str) -> None: ...


class LocalThreadExecutor:
    name = "local"

    def __init__(self, max_workers: int = 1):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="numbench-job")

    def submit(self, store: JobStore, job_id: str) -> None:
        self._pool.submit(run_job, store.run_dir(job_id))

    def refresh(self, store: JobStore, job_id: str) -> None:
        return None  # the worker thread writes status.json itself

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


@dataclass(frozen=True)
class SlurmConfig:
    partition: str | None = None
    gpus: int = 0  # >0 adds --gres=gpu:N
    cpus_per_task: int = 8
    mem: str = "16G"
    time_limit: str = "00:30:00"
    python: str = sys.executable  # on a real cluster point this at the module/venv python
    setup_lines: tuple[str, ...] = ()  # e.g. ("module load cuda/12.4", "source /opt/venv/bin/activate")


_SLURM_FAILED = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


class SlurmExecutor:
    name = "slurm"

    def __init__(self, config: SlurmConfig | None = None, sbatch: str = "sbatch", squeue: str = "squeue"):
        self.config = config or SlurmConfig()
        self.sbatch, self.squeue = sbatch, squeue

    # -- script generation (pure function of config + run_dir, easy to unit test)
    def render_script(self, job_id: str, run_dir: Path) -> str:
        c = self.config
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=numbench-{job_id}",
            f"#SBATCH --output={run_dir}/slurm-%j.out",
            f"#SBATCH --time={c.time_limit}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={c.cpus_per_task}",
            f"#SBATCH --mem={c.mem}",
        ]
        if c.partition:
            lines.append(f"#SBATCH --partition={c.partition}")
        if c.gpus > 0:
            lines.append(f"#SBATCH --gres=gpu:{c.gpus}")
        lines += [
            "",
            "set -euo pipefail",
            'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
            *c.setup_lines,
            f"{shlex.quote(c.python)} -m numbench.runner {shlex.quote(str(run_dir))}",
            "",
        ]
        return "\n".join(lines)

    def submit(self, store: JobStore, job_id: str) -> None:
        if shutil.which(self.sbatch) is None:
            raise ExecutorUnavailable(f"`{self.sbatch}` not found on PATH; this host is not a SLURM submit node")
        run_dir = store.run_dir(job_id)
        script = run_dir / "job.sbatch"
        script.write_text(self.render_script(job_id, run_dir))
        proc = subprocess.run([self.sbatch, "--parsable", str(script)], capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            update_status(
                run_dir, state=JobState.failed.value, finished=now_iso(), error=f"sbatch failed: {proc.stderr.strip()}"
            )
            raise ExecutorUnavailable(f"sbatch failed: {proc.stderr.strip()}")
        slurm_id = proc.stdout.strip().split(";")[0]  # --parsable prints "id[;cluster]"
        update_status(run_dir, scheduler_job_id=slurm_id, message=f"submitted to SLURM as {slurm_id}")

    def refresh(self, store: JobStore, job_id: str) -> None:
        run_dir = store.run_dir(job_id)
        status = read_status(run_dir)
        if status["state"] in (JobState.done.value, JobState.failed.value) or not status.get("scheduler_job_id"):
            return
        try:
            proc = subprocess.run(
                [self.squeue, "-h", "-j", status["scheduler_job_id"], "-o", "%T"],
                capture_output=True, text=True, timeout=30,
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return  # scheduler temporarily unreachable: keep the last known state
        if proc.returncode != 0:
            return
        slurm_state = proc.stdout.strip().split("\n")[0].strip()
        if slurm_state in _SLURM_FAILED:
            update_status(run_dir, state=JobState.failed.value, finished=now_iso(), error=f"SLURM state {slurm_state}")
        elif not slurm_state:
            # Left the queue. The worker may have finished a moment ago: re-read before judging.
            latest = read_status(run_dir)
            if latest["state"] not in (JobState.done.value, JobState.failed.value):
                update_status(
                    run_dir, state=JobState.failed.value, finished=now_iso(),
                    error=f"job {status['scheduler_job_id']} left the SLURM queue without a result (see slurm-*.out)",
                )  # fmt: skip
