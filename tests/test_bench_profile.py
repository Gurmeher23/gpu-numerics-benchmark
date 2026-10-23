import csv
import math

import pytest

from numbench import bench, profile
from numbench.backends import list_backends


def test_registry_reports_reasons_for_unavailable_backends():
    infos = {b.name: b for b in list_backends()}
    assert {"numpy64", "numpy32", "torch-cpu-f32", "torch-cuda-f32", "cupy-f32", "numba-f64"} <= set(infos)
    for b in infos.values():
        assert b.available or b.reason  # nothing is silently unavailable


def test_benchmark_smoke_and_csv_roundtrip(tmp_path):
    rows = bench.run_benchmark(["nbody", "heat"], ["numpy32", "torch-cpu-f32"], {"nbody": [16], "heat": [16]},
                               {"nbody": 2, "heat": 5}, repeats=3, warmup=1)  # fmt: skip
    assert len(rows) == 6  # (reference + 2 backends) x 2 kernels
    ref = [r for r in rows if r["backend"] == "numpy64"]
    assert all(r["speedup_vs_numpy64"] == 1.0 and r["rel_err_vs_ref"] == 0.0 for r in ref)
    for r in rows:
        assert r["q1_s"] <= r["median_s"] <= r["q3_s"]
        assert r["iqr_s"] >= 0 and r["median_s"] > 0
        assert r["rel_err_vs_ref"] < 1e-4  # fast but wrong would show up here
    numpy_row = next(r for r in rows if r["backend"] == "numpy32")
    assert numpy_row["mem_source"] == "tracemalloc" and numpy_row["peak_mem_mb"] > 0
    torch_row = next(r for r in rows if r["backend"] == "torch-cpu-f32")
    assert math.isnan(torch_row["peak_mem_mb"]) and torch_row["mem_source"] == "unavailable"

    out = tmp_path / "b.csv"
    bench.write_csv(rows, out)
    back = bench.read_csv(out)
    assert [r["backend"] for r in back] == [r["backend"] for r in rows]
    assert list(csv.DictReader(out.open()).fieldnames) == bench.CSV_FIELDS
    bench.plot_speedup(rows, tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 1000


def test_unavailable_backend_fails_fast():
    from numbench.backends import BackendUnavailable

    with pytest.raises(BackendUnavailable, match="cupy"):
        bench.run_benchmark(["heat"], ["cupy-f32"], {"heat": [16]}, {"heat": 2}, repeats=1, warmup=0)


def test_cli_quick(tmp_path):
    assert bench.main(["--quick", "--kernel", "heat", "--backends", "numpy32", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "benchmark.csv").exists() and (tmp_path / "speedup.png").exists()


def test_cprofile_report_names_the_hotspot():
    text = profile.cprofile_report("nbody", "numpy64", 64, steps=2, top=8)
    assert "tottime" in text and "sum" in text  # the reduction shows up as a C-level call


def test_torch_profile_report():
    text = profile.torch_profile_report("heat", "torch-cpu-f32", 32, steps=3, top=5)
    assert "aten::" in text
