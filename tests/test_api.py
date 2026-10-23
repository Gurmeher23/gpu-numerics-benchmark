import json
import time

import pytest
from fastapi.testclient import TestClient

from numbench.api import create_app
from numbench.executors import LocalThreadExecutor
from numbench.jobs import JobStore


@pytest.fixture
def client(tmp_path):
    ex = LocalThreadExecutor()
    app = create_app(JobStore(tmp_path / "runs"), ex)
    with TestClient(app) as c:
        yield c
    ex.shutdown()


def wait_done(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/jobs/{job_id}").json()
        if st["state"] in ("done", "failed"):
            return st
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_submit_poll_fetch_benchmark(client):
    spec = {"kernel": "nbody", "backends": ["numpy32"], "sizes": [16, 32], "steps": 2, "repeats": 2, "warmup": 0}
    r = client.post("/jobs", json=spec)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    st = wait_done(client, job_id)
    assert st["state"] == "done" and st["progress"] == 1.0

    rows = client.get(f"/jobs/{job_id}/result").json()["rows"]
    assert {r["backend"] for r in rows} == {"numpy64", "numpy32"} and len(rows) == 4
    csv_text = client.get(f"/jobs/{job_id}/result", params={"format": "csv"}).text
    assert csv_text.splitlines()[0].startswith("kernel,backend") and len(csv_text.splitlines()) == 5

    manifest = client.get(f"/jobs/{job_id}/manifest").json()
    assert manifest["inputs"]["kernel"] == "nbody"
    assert len(manifest["environment"]["env_hash"]) == 64 and "commit" in manifest["git"]
    assert job_id in client.get("/jobs").json()


def test_simulation_job_reports_physics(client):
    spec = {"kind": "simulation", "kernel": "heat", "backends": ["numpy64"], "sizes": [129], "steps": 82}
    job_id = client.post("/jobs", json=spec).json()["job_id"]
    assert wait_done(client, job_id)["state"] == "done"
    diag = client.get(f"/jobs/{job_id}/result").json()["diagnostics"]
    assert diag["linf_error_vs_analytic"] < 1e-3

    spec = {"kind": "simulation", "kernel": "nbody", "backends": ["torch-cpu-f32"], "sizes": [32], "steps": 20}
    job_id = client.post("/jobs", json=spec).json()["job_id"]
    assert wait_done(client, job_id)["state"] == "done"
    diag = client.get(f"/jobs/{job_id}/result").json()["diagnostics"]
    assert diag["energy_rel_drift"] < 1e-3 and diag["pos_max_err_vs_float64_ref"] < 1e-4


def test_result_before_done_is_409_and_unknown_is_404(client, tmp_path):
    store = JobStore(tmp_path / "runs")
    from numbench.jobs import JobSpec

    job_id = store.create(JobSpec(kernel="heat", sizes=[16]), "local")  # never executed
    app = create_app(store, LocalThreadExecutor())
    with TestClient(app) as c:
        assert c.get(f"/jobs/{job_id}").json()["state"] == "queued"
        assert c.get(f"/jobs/{job_id}/result").status_code == 409
        assert c.get("/jobs/0123456789ab").status_code == 404


@pytest.mark.parametrize(
    "bad",
    [
        {"kernel": "fft", "sizes": [16]},
        {"kernel": "heat", "sizes": []},
        {"kernel": "heat", "sizes": [10**6]},
        {"kernel": "heat", "sizes": [16], "backends": ["nope"]},
        {"kernel": "heat", "sizes": [16], "extra_field": 1},
        {"kind": "simulation", "kernel": "heat", "sizes": [16, 32]},
    ],
)
def test_invalid_specs_rejected(client, bad):
    assert client.post("/jobs", json=bad).status_code == 422


def test_unavailable_backend_rejected_with_reason(client):
    r = client.post("/jobs", json={"kernel": "heat", "sizes": [16], "backends": ["cupy-f32"]})
    assert r.status_code == 422
    assert "cupy-f32" in json.dumps(r.json()["detail"])


def test_failed_job_is_reported(client, monkeypatch):
    import numbench.runner as runner

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(runner, "run_benchmark", boom)
    job_id = client.post("/jobs", json={"kernel": "heat", "sizes": [16]}).json()["job_id"]
    st = wait_done(client, job_id)
    assert st["state"] == "failed" and "kaboom" in st["error"]
    assert client.get(f"/jobs/{job_id}/result").status_code == 409


def test_backends_endpoint(client):
    names = {b["name"]: b for b in client.get("/backends").json()}
    assert names["numpy64"]["available"] and not names["cupy-f32"]["available"]


def test_api_with_slurm_executor_end_to_end(tmp_path, monkeypatch):
    """Full API path through SlurmExecutor with a fake sbatch that runs the worker inline."""
    import os
    import stat
    import sys

    from numbench.executors import SlurmExecutor

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "sbatch"
    # The fake scheduler executes the generated script synchronously, like a node would.
    fake.write_text(f'#!/bin/sh\nPYTHONPATH={os.getcwd()} sh "$2" >/dev/null 2>&1\necho 777\n')
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    (bin_dir / "squeue").write_text("#!/bin/sh\n")
    (bin_dir / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    from numbench.executors import SlurmConfig

    app = create_app(JobStore(tmp_path / "runs"), SlurmExecutor(SlurmConfig(python=sys.executable)))
    with TestClient(app) as c:
        job_id = c.post("/jobs", json={"kernel": "heat", "sizes": [16], "steps": 3, "repeats": 1, "warmup": 0}).json()[
            "job_id"
        ]
        st = c.get(f"/jobs/{job_id}").json()
        assert st["state"] == "done" and st["scheduler_job_id"] == "777"
        assert c.get(f"/jobs/{job_id}/result", params={"format": "csv"}).status_code == 200
