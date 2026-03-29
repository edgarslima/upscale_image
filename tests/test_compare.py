"""Tests for run comparison (step 15).

Fixtures build minimal run directories with manifest.json and optional
metrics files, matching the structure written by the real pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from upscale_image.cli.main import app
from upscale_image.reports.compare import (
    ComparisonResult,
    RunDelta,
    RunSnapshot,
    _compute_delta,
    compare_runs,
    comparison_to_dict,
    load_run_snapshot,
)

_RUNNER = CliRunner()


# ---------------------------------------------------------------------------
# Helpers: build fake run artefacts
# ---------------------------------------------------------------------------

def _make_manifest(
    run_id: str,
    *,
    model_name: str = "mock",
    scale: int = 4,
    device: str = "cpu",
    precision: str = "fp32",
    total_elapsed_s: float = 2.0,
    avg_inference_ms: float | None = 100.0,
    total: int = 3,
    done: int = 3,
    failed: int = 0,
    skipped: int = 0,
    success_rate: float = 1.0,
) -> dict:
    return {
        "run_id": run_id,
        "model": {
            "name": model_name,
            "scale": scale,
            "device": device,
            "precision": precision,
        },
        "runtime": {"code_version": "0.1.0", "python_version": "3.12.0"},
        "timing": {
            "total_elapsed_s": total_elapsed_s,
            "avg_inference_ms": avg_inference_ms,
            "min_inference_ms": 80.0,
            "max_inference_ms": 120.0,
        },
        "status": {
            "total": total,
            "done": done,
            "failed": failed,
            "skipped": skipped,
            "success_rate": success_rate,
        },
        "artifacts": {
            "effective_config": "effective_config.yaml",
            "outputs_dir": "outputs",
            "metrics_dir": "metrics",
            "log": "logs.txt",
        },
    }


def _make_run_dir(
    base: Path,
    run_id: str,
    *,
    fr_summary: dict | None = None,
    nr_summary: dict | None = None,
    **manifest_kwargs,
) -> Path:
    """Create a minimal run directory with the given artefacts."""
    run_dir = base / run_id
    run_dir.mkdir(parents=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir()

    manifest = _make_manifest(run_id, **manifest_kwargs)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if fr_summary is not None:
        (metrics_dir / "summary.json").write_text(
            json.dumps(fr_summary, indent=2), encoding="utf-8"
        )

    if nr_summary is not None:
        (metrics_dir / "niqe_summary.json").write_text(
            json.dumps(nr_summary, indent=2), encoding="utf-8"
        )

    return run_dir


# ---------------------------------------------------------------------------
# load_run_snapshot
# ---------------------------------------------------------------------------

class TestLoadRunSnapshot:
    def test_loads_basic_fields(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A", model_name="mock", scale=4)
        snap = load_run_snapshot(run_dir)

        assert snap.run_id == "run_A"
        assert snap.model_name == "mock"
        assert snap.model_scale == 4
        assert snap.device == "cpu"
        assert snap.precision == "fp32"

    def test_loads_timing(self, tmp_path):
        run_dir = _make_run_dir(
            tmp_path, "run_A", total_elapsed_s=5.5, avg_inference_ms=200.0
        )
        snap = load_run_snapshot(run_dir)

        assert snap.total_elapsed_s == pytest.approx(5.5)
        assert snap.avg_inference_ms == pytest.approx(200.0)

    def test_loads_status(self, tmp_path):
        run_dir = _make_run_dir(
            tmp_path, "run_A", total=10, done=8, failed=2, skipped=1, success_rate=0.8
        )
        snap = load_run_snapshot(run_dir)

        assert snap.total_images == 10
        assert snap.done == 8
        assert snap.failed == 2
        assert snap.skipped == 1
        assert snap.success_rate == pytest.approx(0.8)

    def test_no_metrics_gives_none(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        snap = load_run_snapshot(run_dir)

        assert snap.avg_psnr is None
        assert snap.avg_ssim is None
        assert snap.avg_lpips is None
        assert snap.avg_niqe is None

    def test_loads_fr_metrics(self, tmp_path):
        fr = {"mode": "full_reference", "avg_psnr": 32.5, "avg_ssim": 0.91, "avg_lpips": 0.05}
        run_dir = _make_run_dir(tmp_path, "run_A", fr_summary=fr)
        snap = load_run_snapshot(run_dir)

        assert snap.avg_psnr == pytest.approx(32.5)
        assert snap.avg_ssim == pytest.approx(0.91)
        assert snap.avg_lpips == pytest.approx(0.05)

    def test_loads_nr_metrics(self, tmp_path):
        nr = {"mode": "no_reference", "avg_niqe": 7.3}
        run_dir = _make_run_dir(tmp_path, "run_A", nr_summary=nr)
        snap = load_run_snapshot(run_dir)

        assert snap.avg_niqe == pytest.approx(7.3)

    def test_missing_manifest_raises_file_not_found(self, tmp_path):
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="manifest.json"):
            load_run_snapshot(run_dir)

    def test_malformed_manifest_raises_value_error(self, tmp_path):
        run_dir = tmp_path / "bad_run"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text("not valid json", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid manifest"):
            load_run_snapshot(run_dir)

    def test_run_dir_stored_in_snapshot(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        snap = load_run_snapshot(run_dir)

        assert str(run_dir) in snap.run_dir


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------

class TestCompareRuns:
    def test_single_run_no_deltas(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        result = compare_runs([run_dir])

        assert len(result.snapshots) == 1
        assert result.deltas == []

    def test_two_runs_one_delta(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = _make_run_dir(tmp_path, "run_B")
        result = compare_runs([dir_a, dir_b])

        assert len(result.snapshots) == 2
        assert len(result.deltas) == 1

    def test_three_runs_two_deltas(self, tmp_path):
        dirs = [_make_run_dir(tmp_path, f"run_{c}") for c in "ABC"]
        result = compare_runs(dirs)

        assert len(result.snapshots) == 3
        assert len(result.deltas) == 2

    def test_delta_run_ids_correct(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = _make_run_dir(tmp_path, "run_B")
        result = compare_runs([dir_a, dir_b])

        delta = result.deltas[0]
        assert delta.run_id_a == "run_A"
        assert delta.run_id_b == "run_B"

    def test_delta_elapsed_correct(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A", total_elapsed_s=2.0)
        dir_b = _make_run_dir(tmp_path, "run_B", total_elapsed_s=3.5)
        result = compare_runs([dir_a, dir_b])

        assert result.deltas[0].delta_total_elapsed_s == pytest.approx(1.5)

    def test_delta_negative_when_improved(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A", total_elapsed_s=5.0)
        dir_b = _make_run_dir(tmp_path, "run_B", total_elapsed_s=3.0)
        result = compare_runs([dir_a, dir_b])

        assert result.deltas[0].delta_total_elapsed_s == pytest.approx(-2.0)

    def test_delta_metrics_none_when_missing(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A")  # no metrics
        dir_b = _make_run_dir(tmp_path, "run_B")  # no metrics
        result = compare_runs([dir_a, dir_b])

        d = result.deltas[0]
        assert d.delta_avg_psnr is None
        assert d.delta_avg_ssim is None
        assert d.delta_avg_niqe is None

    def test_delta_fr_metrics_when_available(self, tmp_path):
        fr_a = {"avg_psnr": 30.0, "avg_ssim": 0.85, "avg_lpips": 0.10}
        fr_b = {"avg_psnr": 32.0, "avg_ssim": 0.90, "avg_lpips": 0.07}
        dir_a = _make_run_dir(tmp_path, "run_A", fr_summary=fr_a)
        dir_b = _make_run_dir(tmp_path, "run_B", fr_summary=fr_b)
        result = compare_runs([dir_a, dir_b])

        d = result.deltas[0]
        assert d.delta_avg_psnr == pytest.approx(2.0)
        assert d.delta_avg_ssim == pytest.approx(0.05, abs=0.001)
        assert d.delta_avg_lpips == pytest.approx(-0.03, abs=0.001)

    def test_empty_run_dirs_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            compare_runs([])

    def test_returns_comparison_result(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        result = compare_runs([run_dir])

        assert isinstance(result, ComparisonResult)


# ---------------------------------------------------------------------------
# comparison_to_dict
# ---------------------------------------------------------------------------

class TestComparisonToDict:
    def test_top_level_keys(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A")
        result = compare_runs([dir_a])
        data = comparison_to_dict(result)

        assert "runs" in data
        assert "deltas" in data

    def test_run_entry_structure(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A")
        result = compare_runs([dir_a])
        data = comparison_to_dict(result)

        run = data["runs"][0]
        assert run["run_id"] == "run_A"
        assert "model" in run
        assert "timing" in run
        assert "status" in run
        assert "metrics" in run

    def test_dict_is_json_serialisable(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = _make_run_dir(tmp_path, "run_B")
        result = compare_runs([dir_a, dir_b])
        data = comparison_to_dict(result)

        serialised = json.dumps(data)
        assert isinstance(serialised, str)

    def test_delta_entry_structure(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = _make_run_dir(tmp_path, "run_B")
        result = compare_runs([dir_a, dir_b])
        data = comparison_to_dict(result)

        delta = data["deltas"][0]
        assert delta["from"] == "run_A"
        assert delta["to"] == "run_B"
        assert "delta_total_elapsed_s" in delta
        assert "delta_avg_psnr" in delta

    def test_metrics_reflect_artefacts(self, tmp_path):
        fr = {"avg_psnr": 35.0, "avg_ssim": 0.95, "avg_lpips": 0.03}
        nr = {"avg_niqe": 4.2}
        dir_a = _make_run_dir(tmp_path, "run_A", fr_summary=fr, nr_summary=nr)
        result = compare_runs([dir_a])
        data = comparison_to_dict(result)

        metrics = data["runs"][0]["metrics"]
        assert metrics["avg_psnr"] == pytest.approx(35.0)
        assert metrics["avg_niqe"] == pytest.approx(4.2)


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

class TestCLICompare:
    def test_compare_two_runs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = _make_run_dir(tmp_path, "run_B")

        result = _RUNNER.invoke(app, ["compare", str(dir_a), str(dir_b)])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["runs"]) == 2
        assert len(data["deltas"]) == 1

    def test_compare_saves_to_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = _make_run_dir(tmp_path, "run_B")
        out_file = tmp_path / "comparison.json"

        result = _RUNNER.invoke(app, [
            "compare", str(dir_a), str(dir_b),
            "--output", str(out_file),
        ])

        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "runs" in data

    def test_compare_single_run_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dir_a = _make_run_dir(tmp_path, "run_A")

        result = _RUNNER.invoke(app, ["compare", str(dir_a)])

        assert result.exit_code == 1

    def test_compare_missing_manifest_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = tmp_path / "run_B"
        dir_b.mkdir()  # no manifest.json

        result = _RUNNER.invoke(app, ["compare", str(dir_a), str(dir_b)])

        assert result.exit_code == 1
