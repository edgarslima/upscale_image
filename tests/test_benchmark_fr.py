"""Tests for full-reference benchmark (step 13).

LPIPS is injected via ``_lpips_metric`` to avoid network downloads in CI.
The mock metric returns a fixed float so the test focuses on structure,
not numeric precision.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from upscale_image.metrics.full_reference import (
    BenchmarkSummary,
    ImagePair,
    PairResult,
    UnpairedOutput,
    _build_summary,
    _save_per_image_csv,
    _save_summary_json,
    compute_psnr,
    compute_ssim,
    pair_outputs_with_references,
    run_full_reference_benchmark,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_image(path: Path, h: int = 16, w: int = 16, value: int = 0) -> None:
    img = np.full((h, w, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _mock_lpips(score: float = 0.1):
    """Return a mock pyiqa LPIPS metric that always returns *score*."""
    m = MagicMock()
    import torch
    m.return_value = torch.tensor(score)
    return m


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

class TestPairing:
    def test_matches_by_stem(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "img1.png")
        _write_image(ref_dir / "img1.png")

        pairs, unpaired = pair_outputs_with_references(out_dir, ref_dir)

        assert len(pairs) == 1
        assert pairs[0].filename == "img1.png"
        assert unpaired == []

    def test_reference_different_extension(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "img1.png")
        _write_image(ref_dir / "img1.jpg")

        pairs, unpaired = pair_outputs_with_references(out_dir, ref_dir)

        assert len(pairs) == 1
        assert unpaired == []

    def test_missing_reference_is_unpaired(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "img1.png")
        _write_image(out_dir / "img2.png")
        _write_image(ref_dir / "img1.png")

        pairs, unpaired = pair_outputs_with_references(out_dir, ref_dir)

        assert len(pairs) == 1
        assert len(unpaired) == 1
        assert unpaired[0].filename == "img2.png"
        assert unpaired[0].reason == "no_reference"

    def test_all_unpaired_when_refs_empty(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        out_dir.mkdir()
        ref_dir.mkdir()
        for i in range(3):
            _write_image(out_dir / f"img{i}.png")

        pairs, unpaired = pair_outputs_with_references(out_dir, ref_dir)

        assert pairs == []
        assert len(unpaired) == 3

    def test_case_insensitive_stem_matching(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "IMG001.png")
        _write_image(ref_dir / "img001.png")

        pairs, unpaired = pair_outputs_with_references(out_dir, ref_dir)

        assert len(pairs) == 1
        assert unpaired == []

    def test_pairs_are_sorted_by_filename(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        out_dir.mkdir()
        ref_dir.mkdir()
        for name in ("c.png", "a.png", "b.png"):
            _write_image(out_dir / name)
            _write_image(ref_dir / name)

        pairs, _ = pair_outputs_with_references(out_dir, ref_dir)

        assert [p.filename for p in pairs] == ["a.png", "b.png", "c.png"]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

class TestMetricComputation:
    def test_psnr_identical_images_is_inf(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        val = compute_psnr(img, img)
        assert val == float("inf") or val > 100

    def test_psnr_different_images(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        ref = np.full((16, 16, 3), 128, dtype=np.uint8)
        val = compute_psnr(img, ref)
        assert 0 < val < 100

    def test_ssim_identical_images_is_one(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        val = compute_ssim(img, img)
        assert abs(val - 1.0) < 1e-6

    def test_ssim_different_images(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        ref = np.full((16, 16, 3), 128, dtype=np.uint8)
        val = compute_ssim(img, ref)
        assert -1.0 <= val < 1.0


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

class TestRunFullReferenceBenchmark:
    def test_creates_per_image_csv(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "img.png")
        _write_image(ref_dir / "img.png")

        run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        assert (met_dir / "per_image.csv").exists()

    def test_creates_summary_json(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "img.png")
        _write_image(ref_dir / "img.png")

        run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        assert (met_dir / "summary.json").exists()

    def test_per_image_csv_has_correct_columns(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "img.png")
        _write_image(ref_dir / "img.png")

        run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        with open(met_dir / "per_image.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert set(rows[0].keys()) == {"filename", "psnr", "ssim", "lpips", "error"}

    def test_per_image_csv_values_for_matched_pair(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "img.png", value=0)
        _write_image(ref_dir / "img.png", value=0)  # identical → high PSNR, SSIM≈1

        run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips(0.05))

        with open(met_dir / "per_image.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        row = rows[0]
        assert row["filename"] == "img.png"
        assert row["error"] == ""
        assert float(row["ssim"]) > 0.99
        assert float(row["lpips"]) == pytest.approx(0.05, abs=0.01)

    def test_unpaired_output_recorded_in_csv(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "img.png")
        # No reference

        run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        with open(met_dir / "per_image.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["error"] == "no_reference"
        assert rows[0]["psnr"] == ""

    def test_summary_json_mode_field(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()

        run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        data = json.loads((met_dir / "summary.json").read_text())
        assert data["mode"] == "full_reference"

    def test_summary_counts_match_results(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()
        for i in range(3):
            _write_image(out_dir / f"img{i}.png")
            if i < 2:
                _write_image(ref_dir / f"img{i}.png")
        # img2 has no reference

        run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        data = json.loads((met_dir / "summary.json").read_text())
        assert data["total_pairs"] == 3
        assert data["computed"] == 2
        assert data["skipped"] == 1

    def test_summary_averages_populated_when_computed(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "a.png")
        _write_image(ref_dir / "a.png")

        run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        data = json.loads((met_dir / "summary.json").read_text())
        assert data["avg_psnr"] is not None
        assert data["avg_ssim"] is not None
        assert data["avg_lpips"] is not None

    def test_summary_averages_none_when_all_skipped(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "img.png")
        # no references

        summary = run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        assert summary.avg_psnr is None
        assert summary.avg_ssim is None
        assert summary.avg_lpips is None

    def test_size_mismatch_recorded_as_error(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()
        _write_image(out_dir / "img.png", h=16, w=16)
        _write_image(ref_dir / "img.png", h=32, w=32)  # different size

        run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        with open(met_dir / "per_image.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert "size_mismatch" in rows[0]["error"]

    def test_empty_run_produces_valid_files(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()

        summary = run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        assert summary.total_pairs == 0
        assert (met_dir / "per_image.csv").exists()
        assert (met_dir / "summary.json").exists()

    def test_metrics_dir_created_if_missing(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics" / "nested"
        out_dir.mkdir()
        ref_dir.mkdir()

        run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        assert met_dir.exists()
        assert (met_dir / "summary.json").exists()

    def test_multiple_pairs_all_recorded(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()
        for i in range(4):
            _write_image(out_dir / f"img{i}.png")
            _write_image(ref_dir / f"img{i}.png")

        run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        with open(met_dir / "per_image.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 4
        assert all(row["error"] == "" for row in rows)

    def test_returns_benchmark_summary(self, tmp_path):
        out_dir = tmp_path / "outputs"
        ref_dir = tmp_path / "refs"
        met_dir = tmp_path / "metrics"
        out_dir.mkdir()
        ref_dir.mkdir()

        summary = run_full_reference_benchmark(out_dir, ref_dir, met_dir, _lpips_metric=_mock_lpips())

        assert isinstance(summary, BenchmarkSummary)
