"""SlurmExecutor against fake `sbatch`/`squeue` executables on PATH."""

import os
import stat
import sys

import pytest

from numbench.executors import ExecutorUnavailable, SlurmConfig, SlurmExecutor
from numbench.jobs import JobSpec, JobStore


def write_exe(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch_calls.log"
    state_file = tmp_path / "squeue_state"
    state_file.write_text("PENDING")
    write_exe(bin_dir / "sbatch", f'echo "$@" >> {log}\necho "424242;cluster1"\n')
    write_exe(bin_dir / "squeue", f"cat {state_file}\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return state_file, log


def make_job(tmp_path):
    store = JobStore(tmp_path / "runs")
    return store, store.create(JobSpec(kernel="nbody", sizes=[64], backends=["numpy64"]), "slurm")


def test_submit_generates_script_and_records_slurm_id(tmp_path, fake_slurm):
    _, log = fake_slurm
    store, job_id = make_job(tmp_path)
    ex = SlurmExecutor(
        SlurmConfig(partition="gpu", gpus=1, time_limit="01:00:00", setup_lines=("module load cuda/12.4",))
    )
    ex.submit(store, job_id)

    script = (store.run_dir(job_id) / "job.sbatch").read_text()
    assert "#SBATCH --partition=gpu" in script and "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --time=01:00:00" in script and "module load cuda/12.4" in script
    assert f"-m numbench.runner {store.run_dir(job_id)}" in script
    assert "--parsable" in log.read_text() and "job.sbatch" in log.read_text()
    assert store.status(job_id)["scheduler_job_id"] == "424242"  # ";cluster1" suffix stripped


def test_cpu_script_has_no_gres(tmp_path):
    store, job_id = make_job(tmp_path)
    script = SlurmExecutor(SlurmConfig()).render_script(job_id, store.run_dir(job_id))
    assert "--gres" not in script and "--partition" not in script and sys.executable in script


def test_refresh_detects_failures_and_vanished_jobs(tmp_path, fake_slurm):
    state_file, _ = fake_slurm
    store, job_id = make_job(tmp_path)
    ex = SlurmExecutor()
    ex.submit(store, job_id)

    ex.refresh(store, job_id)  # PENDING: unchanged
    assert store.status(job_id)["state"] == "queued"

    state_file.write_text("")  # job left the queue but wrote no result
    ex.refresh(store, job_id)
    st = store.status(job_id)
    assert st["state"] == "failed" and "left the SLURM queue" in st["error"]

    store2, job2 = make_job(tmp_path)
    ex.submit(store2, job2)
    state_file.write_text("TIMEOUT")
    ex.refresh(store2, job2)
    assert store2.status(job2)["error"] == "SLURM state TIMEOUT"


def test_job_that_finished_before_leaving_queue_stays_done(tmp_path, fake_slurm):
    state_file, _ = fake_slurm
    store, job_id = make_job(tmp_path)
    ex = SlurmExecutor()
    ex.submit(store, job_id)
    from numbench.runner import run_job

    run_job(store.run_dir(job_id))  # what the compute node would do
    state_file.write_text("")
    ex.refresh(store, job_id)
    assert store.status(job_id)["state"] == "done"
    assert (store.run_dir(job_id) / "result.csv").exists()


def test_missing_sbatch_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
    store, job_id = make_job(tmp_path)
    with pytest.raises(ExecutorUnavailable, match="not found"):
        SlurmExecutor().submit(store, job_id)


def test_sbatch_error_marks_job_failed(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_exe(bin_dir / "sbatch", 'echo "sbatch: error: invalid partition" >&2\nexit 1\n')
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    store, job_id = make_job(tmp_path)
    with pytest.raises(ExecutorUnavailable, match="invalid partition"):
        SlurmExecutor().submit(store, job_id)
    assert store.status(job_id)["state"] == "failed"
