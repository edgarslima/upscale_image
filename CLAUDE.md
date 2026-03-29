# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python CLI application for batch image super-resolution using deep learning models. Applies SR models to collections of images, tracks quality metrics, and produces auditable run artifacts.

**Status**: Pre-implementation — architecture is fully documented in `docs/`, but source code has not been written yet. Follow the 18-step plan in `docs/plano_criação_passos.md` in order, validating against acceptance criteria at each step.

## Environment Setup

```bash
# Activate virtual environment (already created with deps installed)
source .venv/bin/activate

# Tiered installation
pip install -r requirements/base.txt        # Core runtime
pip install -r requirements/benchmark.txt   # Adds PSNR/SSIM/LPIPS/NIQE metrics
pip install -r requirements/dev.txt         # Adds pytest
pip install -r requirements/realesrgan-extra.txt  # Optional: Real-ESRGAN ecosystem
```

## Testing

```bash
pytest                        # Run all tests
pytest tests/test_foo.py      # Run a single test file
pytest -k "test_name"         # Run tests matching a name pattern
```

## Architecture

The pipeline is linear and deterministic:

```
Input Discovery → Config Resolution → Model Load → Batch Inference →
Per-Item Results → Aggregation → Manifest → Benchmark → Compare → Report
```

**Configuration precedence**: CLI flags > YAML file > defaults

**Run artifact structure**: Each execution creates an isolated directory `run_<timestamp>_<model>_<scale>/` containing `outputs/`, `metrics/`, `logs/`, and the effective config snapshot. This is the unit of persistence and auditability.

**Model abstraction**: Models must conform to a common contract (interface + registry). The pipeline never calls model implementations directly — always through the registry. New models are added by registering them, not modifying the pipeline.

**Error strategy**: Item-level failures (corrupt images, inference errors) are logged and skipped; the run continues. Structural failures (invalid config, model load failure) abort early.

**Quality metrics**:
- Full-reference (when ground truth available): PSNR, SSIM, LPIPS
- No-reference (blind): NIQE

## Key Documentation

All docs are written in Portuguese:

- `docs/especificação_tecnica.md` — Complete technical specification (primary reference)
- `docs/adr/` — 9 architectural decision records (frozen decisions, do not contradict them)
- `docs/parts/` — 18 numbered implementation guides with acceptance criteria
- `docs/plano_criação_passos.md` — The 18-step ordered roadmap

## Technology Stack

| Concern | Library |
|---|---|
| CLI | `typer` |
| Terminal output / logging | `rich` |
| Deep learning inference | `torch` + `torchvision` |
| Image I/O | `opencv-python`, `Pillow` |
| Full-reference metrics | `scikit-image` |
| Advanced metrics (LPIPS, NIQE) | `pyiqa` |
| Config files | `PyYAML` |
| Testing | `pytest` |
