"""Tests for HTML report generation (step 16).

The HTML is treated as a presentation layer: tests verify structure,
content presence and writability — not pixel-perfect rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from upscale_image.cli.main import app
from upscale_image.reports.compare import compare_runs, load_run_snapshot
from upscale_image.reports.html import (
    generate_html_report,
    render_comparison_report,
    render_run_report,
)

_RUNNER = CliRunner()


# ---------------------------------------------------------------------------
# Helpers (duplicated minimal subset to keep this file self-contained)
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
        "model": {"name": model_name, "scale": scale, "device": device, "precision": precision},
        "runtime": {"code_version": "0.1.0", "python_version": "3.12.0"},
        "timing": {
            "total_elapsed_s": total_elapsed_s,
            "avg_inference_ms": avg_inference_ms,
            "min_inference_ms": 80.0,
            "max_inference_ms": 120.0,
        },
        "status": {
            "total": total, "done": done, "failed": failed,
            "skipped": skipped, "success_rate": success_rate,
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
    run_dir = base / run_id
    run_dir.mkdir(parents=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(_make_manifest(run_id, **manifest_kwargs)), encoding="utf-8"
    )
    if fr_summary:
        (metrics_dir / "summary.json").write_text(
            json.dumps(fr_summary), encoding="utf-8"
        )
    if nr_summary:
        (metrics_dir / "niqe_summary.json").write_text(
            json.dumps(nr_summary), encoding="utf-8"
        )
    return run_dir


# ---------------------------------------------------------------------------
# render_run_report (single run)
# ---------------------------------------------------------------------------

class TestRenderRunReport:
    def test_returns_string(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        snap = load_run_snapshot(run_dir)
        html = render_run_report(snap)

        assert isinstance(html, str)

    def test_is_valid_html_structure(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        snap = load_run_snapshot(run_dir)
        html = render_run_report(snap)

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<body" in html

    def test_contains_run_id(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_20260101_120000_mock_4x")
        snap = load_run_snapshot(run_dir)
        html = render_run_report(snap)

        assert "run_20260101_120000_mock_4x" in html

    def test_contains_model_name(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A", model_name="realesrgan-x4")
        snap = load_run_snapshot(run_dir)
        html = render_run_report(snap)

        assert "realesrgan-x4" in html

    def test_contains_timing_fields(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A", total_elapsed_s=5.123)
        snap = load_run_snapshot(run_dir)
        html = render_run_report(snap)

        assert "5.123" in html

    def test_contains_status_counts(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A", total=10, done=9, failed=1)
        snap = load_run_snapshot(run_dir)
        html = render_run_report(snap)

        assert "10" in html
        assert "9" in html

    def test_contains_fr_metrics_when_present(self, tmp_path):
        fr = {"avg_psnr": 34.5, "avg_ssim": 0.93, "avg_lpips": 0.04}
        run_dir = _make_run_dir(tmp_path, "run_A", fr_summary=fr)
        snap = load_run_snapshot(run_dir)
        html = render_run_report(snap)

        assert "34.5" in html or "34.50" in html

    def test_contains_niqe_when_present(self, tmp_path):
        nr = {"avg_niqe": 6.78}
        run_dir = _make_run_dir(tmp_path, "run_A", nr_summary=nr)
        snap = load_run_snapshot(run_dir)
        html = render_run_report(snap)

        assert "6.78" in html or "6.780" in html

    def test_na_placeholder_when_metrics_absent(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        snap = load_run_snapshot(run_dir)
        html = render_run_report(snap)

        assert "—" in html  # na placeholder for missing metrics

    def test_has_inline_style(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        snap = load_run_snapshot(run_dir)
        html = render_run_report(snap)

        assert "<style>" in html


# ---------------------------------------------------------------------------
# render_comparison_report (multiple runs)
# ---------------------------------------------------------------------------

class TestRenderComparisonReport:
    def test_single_run_renders_summary(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        result = compare_runs([run_dir])
        html = render_comparison_report(result)

        assert "run_A" in html

    def test_two_runs_both_ids_present(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = _make_run_dir(tmp_path, "run_B")
        result = compare_runs([dir_a, dir_b])
        html = render_comparison_report(result)

        assert "run_A" in html
        assert "run_B" in html

    def test_comparison_section_present(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = _make_run_dir(tmp_path, "run_B")
        result = compare_runs([dir_a, dir_b])
        html = render_comparison_report(result)

        assert "comparison" in html.lower() or "delta" in html.lower()

    def test_delta_row_present(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A", total_elapsed_s=2.0)
        dir_b = _make_run_dir(tmp_path, "run_B", total_elapsed_s=3.5)
        result = compare_runs([dir_a, dir_b])
        html = render_comparison_report(result)

        assert "Delta" in html or "delta" in html.lower() or "Δ" in html

    def test_three_runs_renders_without_error(self, tmp_path):
        dirs = [_make_run_dir(tmp_path, f"run_{c}") for c in "ABC"]
        result = compare_runs(dirs)
        html = render_comparison_report(result)

        assert "run_A" in html
        assert "run_B" in html
        assert "run_C" in html

    def test_html_is_self_contained(self, tmp_path):
        """No external script/link references."""
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = _make_run_dir(tmp_path, "run_B")
        result = compare_runs([dir_a, dir_b])
        html = render_comparison_report(result)

        assert "<link " not in html
        assert "<script src=" not in html


# ---------------------------------------------------------------------------
# generate_html_report (file I/O)
# ---------------------------------------------------------------------------

class TestGenerateHtmlReport:
    def test_creates_file(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        out = tmp_path / "report.html"

        generate_html_report([run_dir], out)

        assert out.exists()

    def test_file_is_nonempty(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        out = tmp_path / "report.html"

        generate_html_report([run_dir], out)

        assert out.stat().st_size > 100

    def test_returns_path(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        out = tmp_path / "report.html"

        result = generate_html_report([run_dir], out)

        assert result == out

    def test_creates_parent_dirs(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        out = tmp_path / "nested" / "deep" / "report.html"

        generate_html_report([run_dir], out)

        assert out.exists()

    def test_file_starts_with_doctype(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_A")
        out = tmp_path / "report.html"

        generate_html_report([run_dir], out)

        content = out.read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html>")

    def test_two_runs_comparison(self, tmp_path):
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = _make_run_dir(tmp_path, "run_B")
        out = tmp_path / "report.html"

        generate_html_report([dir_a, dir_b], out)

        content = out.read_text(encoding="utf-8")
        assert "run_A" in content
        assert "run_B" in content


# ---------------------------------------------------------------------------
# CLI report command
# ---------------------------------------------------------------------------

class TestCLIReport:
    def test_report_single_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = _make_run_dir(tmp_path, "run_A")
        out = tmp_path / "report.html"

        result = _RUNNER.invoke(app, [
            "report", str(run_dir),
            "--output", str(out),
        ])

        assert result.exit_code == 0
        assert out.exists()

    def test_report_two_runs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dir_a = _make_run_dir(tmp_path, "run_A")
        dir_b = _make_run_dir(tmp_path, "run_B")
        out = tmp_path / "report.html"

        result = _RUNNER.invoke(app, [
            "report", str(dir_a), str(dir_b),
            "--output", str(out),
        ])

        assert result.exit_code == 0
        assert out.exists()

    def test_report_missing_manifest_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bad_dir = tmp_path / "no_manifest"
        bad_dir.mkdir()
        out = tmp_path / "report.html"

        result = _RUNNER.invoke(app, [
            "report", str(bad_dir),
            "--output", str(out),
        ])

        assert result.exit_code == 1

    def test_report_output_path_in_message(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = _make_run_dir(tmp_path, "run_A")
        out = tmp_path / "my_report.html"

        result = _RUNNER.invoke(app, [
            "report", str(run_dir),
            "--output", str(out),
        ])

        assert "my_report.html" in result.output
