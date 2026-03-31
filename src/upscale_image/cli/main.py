"""Main CLI entry point for upscale_image."""

from __future__ import annotations

import typer
from rich.console import Console

from upscale_image import __version__
from upscale_image.config import resolve_config
from upscale_image.metrics import run_full_reference_benchmark, run_no_reference_benchmark
from upscale_image.models import resolve_model
from upscale_image.optimize import OptimizeConfig, run_optimization
from upscale_image.pdf import (
    ComposeReadyResult,
    PagePrepConfig,
    PdfComposeResult,
    PdfExtractionConfig,
    compose_pdf_from_pages,
    extract_pdf_pages,
    prepare_pages_for_composition,
)
from upscale_image.pipeline import (
    create_run,
    patch_manifest_with_compose_ready,
    patch_manifest_with_optimization,
    patch_manifest_with_pdf_rebuilt,
    patch_manifest_with_pdf_source,
    run_batch,
    setup_run_logger,
    write_manifest,
)
from upscale_image.reports import compare_runs, comparison_to_dict, generate_html_report

app = typer.Typer(
    name="upscale-image",
    help="Batch image super-resolution with measurable, auditable runs.",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"upscale-image {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """upscale-image: batch image super-resolution CLI."""


@app.command()
def upscale(
    input_dir: str = typer.Argument(
        None,
        help="Directory with input images. Omit when using --pdf-file.",
    ),
    output_dir: str = typer.Option(..., "--output", "-o", help="Base output directory."),
    pdf_file: str = typer.Option(
        None,
        "--pdf-file",
        help="Path to a PDF file. When provided, pages are extracted and upscaled; a rebuilt PDF is generated.",
    ),
    dpi: int = typer.Option(
        150,
        "--dpi",
        help="DPI for PDF page rendering (only used with --pdf-file).",
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model name to use."),
    scale: int = typer.Option(None, "--scale", "-s", help="Upscale factor."),
    config: str = typer.Option(None, "--config", "-c", help="Path to YAML config file."),
    device: str = typer.Option(None, "--device", "-d", help="Compute device: cpu or cuda."),
    tile_size: int = typer.Option(
        None,
        "--tile-size",
        "-t",
        help="Tile size for tiled inference (0 = disabled). Use e.g. 512 to reduce VRAM usage.",
    ),
    optimize: bool = typer.Option(
        False,
        "--optimize",
        help="Run optimization on outputs after inference (generates optimized/ derivatives).",
    ),
    opt_webp_quality: int = typer.Option(
        80,
        "--opt-webp-quality",
        help="WebP quality for chained optimization (0-100).",
    ),
    opt_jpeg_quality: int = typer.Option(
        85,
        "--opt-jpeg-quality",
        help="JPEG quality for chained optimization (0-100).",
    ),
    opt_formats: list[str] = typer.Option(
        None,
        "--opt-format",
        help="Output formats for chained optimization. Repeatable. Defaults to webp and jpeg.",
    ),
    pdf_budget_ratio: float = typer.Option(
        2.0,
        "--pdf-budget-ratio",
        help="Maximum allowed size ratio of rebuilt PDF vs original (default 2.0 = 2×).",
    ),
    async_io: bool = typer.Option(
        False,
        "--async-io",
        help="Overlap disk I/O and GPU inference (producer-consumer pipeline).",
    ),
    prefetch: int = typer.Option(
        4,
        "--prefetch",
        help="Number of images to prefetch from disk (only used with --async-io).",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Images per GPU forward pass. 0 = auto-detect by VRAM (requires --async-io).",
    ),
    multi_gpu: bool = typer.Option(
        False,
        "--multi-gpu",
        help="Distribute tasks across all available CUDA GPUs (requires 2+ GPUs).",
    ),
    reference_dir: str = typer.Option(
        None,
        "--reference-dir",
        "-r",
        help="Directory with ground truth images for full-reference benchmark.",
    ),
    benchmark_nr: bool = typer.Option(
        False,
        "--benchmark-nr",
        help="Run no-reference benchmark (NIQE) on outputs.",
    ),
) -> None:
    """Run super-resolution on all images in INPUT_DIR or on pages from a PDF.

    Provide either INPUT_DIR (image mode) or --pdf-file (PDF mode), not both.
    """
    from pathlib import Path as _Path

    # --- Validate input mode ---
    if pdf_file and input_dir:
        console.print("[red]Error:[/red] Provide either INPUT_DIR or --pdf-file, not both.")
        raise typer.Exit(1)
    if not pdf_file and not input_dir:
        console.print("[red]Error:[/red] Provide either INPUT_DIR or --pdf-file.")
        raise typer.Exit(1)

    pdf_path = _Path(pdf_file) if pdf_file else None

    # --- Configuration ---
    # Use pdf parent dir as placeholder when in PDF mode (overridden after extraction)
    effective_input_dir = input_dir if input_dir else str(pdf_path.parent)
    try:
        cfg = resolve_config(
            input_dir=effective_input_dir,
            output_dir=output_dir,
            model=model,
            scale=scale,
            device=device,
            config_file=config,
            tile_size=tile_size,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1) from exc

    # --- Multi-GPU validation ---
    if multi_gpu:
        try:
            import torch as _torch
            _n_gpus = _torch.cuda.device_count()
        except ImportError:
            _n_gpus = 0
        if _n_gpus < 2:
            console.print(
                f"[yellow]Warning:[/yellow] --multi-gpu requested but only "
                f"{_n_gpus} CUDA GPU(s) detected — continuing in single-GPU mode."
            )
        else:
            cfg.runtime.multi_gpu = True

    # --- Run setup ---
    try:
        ctx = create_run(cfg)
    except Exception as exc:
        console.print(f"[red]Failed to create run:[/red] {exc}")
        raise typer.Exit(1) from exc

    logger = setup_run_logger(ctx)

    # --- PDF extraction (when --pdf-file is provided) ---
    pdf_extraction_result = None
    if pdf_path is not None:
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            logger.close()
            console.print(f"[red]Error:[/red] PDF file not found: {pdf_path}")
            raise typer.Exit(1)
        logger.info(f"PDF mode — extracting pages from: {pdf_path.name}")
        try:
            pdf_cfg = PdfExtractionConfig(dpi=dpi)
            pdf_extraction_result = extract_pdf_pages(pdf_path, ctx.run_dir, pdf_cfg)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(f"PDF extraction failed: {exc}")
            logger.close()
            console.print(f"[red]PDF error:[/red] {exc}")
            raise typer.Exit(1) from exc

        if pdf_extraction_result.extracted == 0:
            logger.error("No pages extracted — aborting.")
            logger.close()
            console.print("[red]PDF error:[/red] No pages could be extracted.")
            raise typer.Exit(1)

        logger.info(
            f"  extracted {pdf_extraction_result.extracted}/{pdf_extraction_result.total_pages} pages"
            + (f"  ({pdf_extraction_result.failed} failed)" if pdf_extraction_result.failed else "")
        )
        # Override input_dir so the batch runs on the extracted pages
        cfg.input_dir = str(pdf_extraction_result.extracted_dir)

    # --- Model ---
    try:
        sr_model = resolve_model(cfg)
        sr_model.load()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        logger.error(f"Model load failed: {exc}")
        logger.close()
        console.print(f"[red]Model error:[/red] {exc}")
        raise typer.Exit(1) from exc

    # --- Batch ---
    batch = None
    try:
        batch = run_batch(
            cfg, ctx, sr_model, logger,
            async_io=async_io,
            prefetch_size=prefetch,
            batch_size=batch_size,
        )
    except ValueError as exc:
        logger.error(f"Fatal error during run: {exc}")
        logger.close()
        console.print(f"[red]Run error:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        sr_model.unload()

    # --- Manifest ---
    write_manifest(ctx, cfg, batch)

    # --- Patch manifest with PDF source (when in PDF mode) ---
    if pdf_extraction_result is not None:
        patch_manifest_with_pdf_source(ctx.run_dir, pdf_extraction_result)

    # --- Full-reference benchmark (optional) ---
    if reference_dir:
        try:
            run_full_reference_benchmark(
                outputs_dir=ctx.outputs_dir,
                reference_dir=reference_dir,
                metrics_dir=ctx.metrics_dir,
                device=cfg.runtime.device,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Benchmark warning:[/yellow] {exc}")

    # --- No-reference benchmark (optional) ---
    if benchmark_nr:
        try:
            run_no_reference_benchmark(
                outputs_dir=ctx.outputs_dir,
                metrics_dir=ctx.metrics_dir,
                device=cfg.runtime.device,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]NR benchmark warning:[/yellow] {exc}")

    # --- Chained optimization (optional) ---
    opt_failed = 0
    if optimize:
        opt_cfg = OptimizeConfig(
            formats=opt_formats or ["webp", "jpeg"],
            webp_quality=opt_webp_quality,
            jpeg_quality=opt_jpeg_quality,
        )
        logger.info(
            f"Optimization started — formats={opt_cfg.formats}  "
            f"webp_quality={opt_cfg.webp_quality}  jpeg_quality={opt_cfg.jpeg_quality}"
        )
        try:
            opt_summary = run_optimization(ctx.run_dir, opt_cfg)
            patch_manifest_with_optimization(ctx.run_dir, opt_summary)
            logger.info(
                f"Optimization done — optimized={opt_summary.optimized}  "
                f"failed={opt_summary.failed}  "
                f"saved={opt_summary.bytes_saved_total:,} bytes "
                f"({opt_summary.saving_ratio_total * 100:.1f}%)"
            )
            opt_failed = opt_summary.failed
            if opt_failed:
                logger.warning(
                    f"Optimization partial failures: {opt_failed} item(s) — "
                    "see optimized/per_image.csv"
                )
                console.print(
                    f"[yellow]Optimization:[/yellow] {opt_failed} item(s) failed — "
                    "see optimized/per_image.csv"
                )
            else:
                console.print(
                    f"Optimization: saved {opt_summary.bytes_saved_total:,} bytes "
                    f"({opt_summary.saving_ratio_total * 100:.1f}%)"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Optimization failed: {exc}")
            console.print(f"[yellow]Optimization warning:[/yellow] {exc}")

    # --- PDF recomposition (when in PDF mode) ---
    if pdf_extraction_result is not None:
        source_pdf_size = pdf_extraction_result.source_copy.stat().st_size

        # Step 1: prepare pages (progressive JPEG compression)
        logger.info(
            f"PDF page preparation started — source={source_pdf_size:,} bytes  "
            f"budget={pdf_budget_ratio}× ({int(source_pdf_size * pdf_budget_ratio):,} bytes)"
        )
        compose_ready: ComposeReadyResult | None = None
        try:
            prep_cfg = PagePrepConfig(budget_ratio=pdf_budget_ratio)
            compose_ready = prepare_pages_for_composition(
                ctx.outputs_dir,
                ctx.run_dir,
                source_pdf_size,
                prep_cfg,
            )
            patch_manifest_with_compose_ready(ctx.run_dir, compose_ready)
            status_label = "within budget" if compose_ready.within_budget else "OVER BUDGET"
            logger.info(
                f"Page preparation done — quality={compose_ready.preset_quality}  "
                f"estimated={compose_ready.estimated_bytes:,} bytes  "
                f"ratio={compose_ready.ratio:.2f}  [{status_label}]"
            )
            if not compose_ready.within_budget:
                logger.warning(
                    f"PDF size budget not met: estimated ratio {compose_ready.ratio:.2f} "
                    f"> target {pdf_budget_ratio:.2f} — "
                    "rebuilt PDF may be impractically large"
                )
                console.print(
                    f"[yellow]PDF budget warning:[/yellow] estimated ratio "
                    f"{compose_ready.ratio:.2f} exceeds target {pdf_budget_ratio:.2f}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"PDF page preparation error: {exc}")
            console.print(f"[yellow]PDF preparation warning:[/yellow] {exc}")

        # Step 2: compose PDF from prepared pages (or fall back to outputs/)
        pages_source = compose_ready.pages_dir if compose_ready is not None else ctx.outputs_dir
        logger.info(f"PDF recomposition started — source: {pages_source.relative_to(ctx.run_dir)}")
        try:
            compose_result = compose_pdf_from_pages(
                pages_source,
                ctx.run_dir,
                output_stem=pdf_extraction_result.source_pdf.stem,
            )
            patch_manifest_with_pdf_rebuilt(ctx.run_dir, compose_result)
            if compose_result.status == "ok":
                logger.info(
                    f"PDF rebuilt — {compose_result.pages_included} pages → "
                    f"{compose_result.output_pdf.name}"
                )
                console.print(
                    f"PDF rebuilt: {compose_result.pages_included} pages → "
                    f"pdf/rebuilt/{compose_result.output_pdf.name}"
                )
            else:
                logger.warning(f"PDF recomposition failed: {compose_result.error}")
                console.print(f"[yellow]PDF rebuild warning:[/yellow] {compose_result.error}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"PDF recomposition error: {exc}")
            console.print(f"[yellow]PDF rebuild warning:[/yellow] {exc}")

    logger.close()

    raise typer.Exit(0 if batch.failed == 0 else 2)


@app.command()
def report(
    run_dirs: list[str] = typer.Argument(..., help="Run directories to include in the report."),
    output: str = typer.Option(
        "report.html",
        "--output",
        "-o",
        help="Output HTML file path.",
    ),
) -> None:
    """Generate an HTML report for one or more completed runs."""
    if not run_dirs:
        console.print("[red]Error:[/red] At least one run directory is required.")
        raise typer.Exit(1)

    try:
        out_path = generate_html_report(run_dirs, output)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Report error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"Report written to [bold]{out_path}[/bold]")


@app.command()
def compare(
    run_dirs: list[str] = typer.Argument(..., help="Run directories to compare."),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Save comparison JSON to this file path.",
    ),
) -> None:
    """Compare two or more completed runs side by side."""
    if len(run_dirs) < 2:
        console.print("[red]Error:[/red] At least two run directories are required.")
        raise typer.Exit(1)

    try:
        result = compare_runs(run_dirs)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Comparison error:[/red] {exc}")
        raise typer.Exit(1) from exc

    data = comparison_to_dict(result)

    if output:
        import json as _json
        from pathlib import Path as _Path
        _Path(output).write_text(
            _json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        console.print(f"Comparison saved to [bold]{output}[/bold]")
    else:
        import json as _json
        console.print(_json.dumps(data, indent=2, ensure_ascii=False))


@app.command()
def pdf(
    pdf_file: str = typer.Argument(..., help="Path to the PDF file to upscale."),
    output_dir: str = typer.Option(..., "--output", "-o", help="Base output directory."),
    model: str = typer.Option(None, "--model", "-m", help="Model name to use."),
    scale: int = typer.Option(None, "--scale", "-s", help="Upscale factor."),
    config: str = typer.Option(None, "--config", "-c", help="Path to YAML config file."),
    device: str = typer.Option(None, "--device", "-d", help="Compute device: cpu or cuda."),
    tile_size: int = typer.Option(
        None,
        "--tile-size",
        "-t",
        help="Tile size for tiled inference (0 = disabled).",
    ),
    dpi: int = typer.Option(150, "--dpi", help="DPI for PDF page rendering."),
    optimize: bool = typer.Option(
        False,
        "--optimize",
        help="Run optimization on outputs after inference.",
    ),
    opt_webp_quality: int = typer.Option(80, "--opt-webp-quality"),
    opt_jpeg_quality: int = typer.Option(85, "--opt-jpeg-quality"),
    opt_formats: list[str] = typer.Option(None, "--opt-format"),
) -> None:
    """Extract pages from a PDF and run super-resolution on each page."""
    from pathlib import Path as _Path

    pdf_path = _Path(pdf_file)

    # --- Validate PDF path early ---
    if not pdf_path.exists():
        console.print(f"[red]Error:[/red] PDF file not found: {pdf_path}")
        raise typer.Exit(1)

    # --- Configuration (uses a placeholder input_dir; replaced after extraction) ---
    try:
        cfg = resolve_config(
            input_dir=str(pdf_path.parent),  # placeholder, overridden below
            output_dir=output_dir,
            model=model,
            scale=scale,
            device=device,
            config_file=config,
            tile_size=tile_size,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1) from exc

    # --- Run setup ---
    try:
        ctx = create_run(cfg)
    except Exception as exc:
        console.print(f"[red]Failed to create run:[/red] {exc}")
        raise typer.Exit(1) from exc

    logger = setup_run_logger(ctx)

    # --- PDF extraction ---
    logger.info(f"PDF mode — extracting pages from: {pdf_path.name}")
    try:
        pdf_cfg = PdfExtractionConfig(dpi=dpi)
        pdf_result = extract_pdf_pages(pdf_path, ctx.run_dir, pdf_cfg)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"PDF extraction failed: {exc}")
        logger.close()
        console.print(f"[red]PDF error:[/red] {exc}")
        raise typer.Exit(1) from exc

    logger.info(
        f"  extracted {pdf_result.extracted}/{pdf_result.total_pages} pages"
        + (f"  ({pdf_result.failed} failed)" if pdf_result.failed else "")
    )

    if pdf_result.extracted == 0:
        logger.error("No pages extracted — aborting.")
        logger.close()
        console.print("[red]PDF error:[/red] No pages could be extracted.")
        raise typer.Exit(1)

    # Override input_dir with the extracted pages directory
    cfg.input_dir = str(pdf_result.extracted_dir)

    # --- Model ---
    try:
        sr_model = resolve_model(cfg)
        sr_model.load()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        logger.error(f"Model load failed: {exc}")
        logger.close()
        console.print(f"[red]Model error:[/red] {exc}")
        raise typer.Exit(1) from exc

    # --- Batch ---
    batch = None
    try:
        batch = run_batch(cfg, ctx, sr_model, logger)
    except ValueError as exc:
        logger.error(f"Fatal error during run: {exc}")
        logger.close()
        console.print(f"[red]Run error:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        sr_model.unload()

    # --- Manifest (base) ---
    write_manifest(ctx, cfg, batch)

    # --- Patch manifest with PDF origin ---
    patch_manifest_with_pdf_source(ctx.run_dir, pdf_result)

    # --- Chained optimization (optional) ---
    if optimize:
        opt_cfg = OptimizeConfig(
            formats=opt_formats or ["webp", "jpeg"],
            webp_quality=opt_webp_quality,
            jpeg_quality=opt_jpeg_quality,
        )
        logger.info(f"Optimization started — formats={opt_cfg.formats}")
        try:
            opt_summary = run_optimization(ctx.run_dir, opt_cfg)
            patch_manifest_with_optimization(ctx.run_dir, opt_summary)
            logger.info(
                f"Optimization done — saved {opt_summary.bytes_saved_total:,} bytes "
                f"({opt_summary.saving_ratio_total * 100:.1f}%)"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Optimization failed: {exc}")
            console.print(f"[yellow]Optimization warning:[/yellow] {exc}")

    logger.close()

    raise typer.Exit(0 if batch.failed == 0 else 2)


@app.command()
def optimize(
    run_dir: str = typer.Argument(..., help="Path to a completed run directory."),
    webp_quality: int = typer.Option(
        80, "--webp-quality", help="WebP compression quality (0-100)."
    ),
    jpeg_quality: int = typer.Option(
        85, "--jpeg-quality", help="JPEG compression quality (0-100)."
    ),
    formats: list[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Output formats to generate. Repeatable. Defaults to webp and jpeg.",
    ),
) -> None:
    """Generate compressed derivatives from a completed run's canonical PNG outputs."""
    from pathlib import Path as _Path

    cfg = OptimizeConfig(
        formats=formats or ["webp", "jpeg"],
        webp_quality=webp_quality,
        jpeg_quality=jpeg_quality,
    )

    try:
        summary = run_optimization(run_dir, cfg)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Optimization error:[/red] {exc}")
        raise typer.Exit(1) from exc

    patch_manifest_with_optimization(_Path(run_dir), summary)

    console.print(
        f"Optimized {summary.optimized} file(s) — "
        f"saved {summary.bytes_saved_total:,} bytes "
        f"({summary.saving_ratio_total * 100:.1f}%)"
    )
    if summary.failed:
        console.print(f"[yellow]{summary.failed} file(s) failed — see optimized/per_image.csv[/yellow]")

    raise typer.Exit(0 if summary.failed == 0 else 2)


def run() -> None:
    app()
