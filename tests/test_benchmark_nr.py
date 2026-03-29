"""Tests for no-reference benchmark (step 14).

NIQE is injected via ``_niqe_metric`` to avoid heavy model loading in tests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

from upscale_image.metrics.no_reference import (
    NiqeResult,
    NrBenchmarkSummary,
    _build_nr_summary,
    _compute_niqe,
    run_no_reference_benchmark,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_image(path: Path, h: int = 64, w: int = 64) -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _mock_niqe(score: float = 5.0):
    """Return a mock NIQE metric that always returns *score*."""
    m = MagicMock()
    m.return_value = torch.tensor(score)
    return m


# ---------------------------------------------------------------------------
# Mode distinction
# ---------------------------------------------------------------------------

class TestModeDistinction:
    def test_summary_mode_is_no_reference(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()

        summary = run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        assert summary.mode == "no_reference"

    def test_summary_json_mode_field(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()

        run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        data = json.loads((met_dir / "niqe_summary.json").read_text())
        assert data["mode"] == "no_reference"

    def test_output_files_named_niqe(self, tmp_path):
        """Output files must use niqe_ prefix to distinguish from FR benchmark."""
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()

        run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        assert (met_dir / "niqe_per_image.csv").exists()
        assert (met_dir / "niqe_summary.json").exists()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestNrPersistence:
    def test_creates_niqe_csv(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        _write_image(out_dir / "img.png")

        run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        assert (met_dir / "niqe_per_image.csv").exists()

    def test_creates_niqe_summary_json(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        _write_image(out_dir / "img.png")

        run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        assert (met_dir / "niqe_summary.json").exists()

    def test_csv_columns(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        _write_image(out_dir / "img.png")

        run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        with open(met_dir / "niqe_per_image.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert set(rows[0].keys()) == {"filename", "niqe", "error"}

    def test_csv_score_value(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        _write_image(out_dir / "img.png")

        run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe(7.5))

        with open(met_dir / "niqe_per_image.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert float(rows[0]["niqe"]) == pytest.approx(7.5, abs=0.01)
        assert rows[0]["error"] == ""

    def test_csv_sorted_by_filename(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        for name in ("c.png", "a.png", "b.png"):
            _write_image(out_dir / name)

        run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        with open(met_dir / "niqe_per_image.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert [r["filename"] for r in rows] == ["a.png", "b.png", "c.png"]

    def test_metrics_dir_created_if_missing(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics" / "nested"
        out_dir.mkdir()

        run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        assert met_dir.exists()

    def test_empty_outputs_produces_valid_files(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()

        summary = run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        assert summary.total == 0
        assert (met_dir / "niqe_per_image.csv").exists()
        assert (met_dir / "niqe_summary.json").exists()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestNrAggregation:
    def test_summary_counts_all_images(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        for i in range(4):
            _write_image(out_dir / f"img{i}.png")

        summary = run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        assert summary.total == 4
        assert summary.computed == 4
        assert summary.skipped == 0

    def test_avg_niqe_correct(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        for i in range(3):
            _write_image(out_dir / f"img{i}.png")

        summary = run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe(6.0))

        assert summary.avg_niqe == pytest.approx(6.0, abs=0.01)

    def test_avg_niqe_none_when_empty(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()

        summary = run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        assert summary.avg_niqe is None

    def test_returns_nr_benchmark_summary(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()

        summary = run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe())

        assert isinstance(summary, NrBenchmarkSummary)

    def test_summary_json_has_avg_niqe(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        _write_image(out_dir / "img.png")

        run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=_mock_niqe(8.0))

        data = json.loads((met_dir / "niqe_summary.json").read_text())
        assert data["avg_niqe"] == pytest.approx(8.0, abs=0.01)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestNrErrorHandling:
    def test_compute_error_recorded_not_raised(self, tmp_path):
        """A compute failure must be recorded, not abort the benchmark."""
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        _write_image(out_dir / "img.png")

        failing_metric = MagicMock(side_effect=RuntimeError("NIQE failed"))
        summary = run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=failing_metric)

        assert summary.skipped == 1
        assert summary.computed == 0

    def test_compute_error_in_csv(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        _write_image(out_dir / "img.png")

        failing_metric = MagicMock(side_effect=RuntimeError("boom"))
        run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=failing_metric)

        with open(met_dir / "niqe_per_image.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert "compute_error" in rows[0]["error"]
        assert rows[0]["niqe"] == ""

    def test_one_error_does_not_stop_remaining(self, tmp_path):
        out_dir = tmp_path / "outputs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        for i in range(3):
            _write_image(out_dir / f"img{i}.png")

        call_count = [0]

        def partial_fail(tensor):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("second fails")
            return torch.tensor(5.0)

        metric = MagicMock(side_effect=partial_fail)
        summary = run_no_reference_benchmark(out_dir, met_dir, _niqe_metric=metric)

        assert summary.total == 3
        assert summary.computed == 2
        assert summary.skipped == 1
