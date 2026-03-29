"""Main CLI entry point for upscale_image."""

from __future__ import annotations

import typer
from rich.console import Console

from upscale_image import __version__
from upscale_image.config import resolve_config
from upscale_image.metrics import run_full_reference_benchmark, run_no_reference_benchmark
from upscale_image.models import resolve_model
from upscale_image.pipeline import create_run, run_batch, setup_run_logger, write_manifest
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
    input_dir: str = typer.Argument(..., help="Directory with input images."),
    output_dir: str = typer.Option(..., "--output", "-o", help="Base output directory."),
    model: str = typer.Option(None, "--model", "-m", help="Model name to use."),
    scale: int = typer.Option(None, "--scale", "-s", help="Upscale factor."),
    config: str = typer.Option(None, "--config", "-c", help="Path to YAML config file."),
    device: str = typer.Option(None, "--device", "-d", help="Compute device: cpu or cuda."),
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
    """Run super-resolution on all images in INPUT_DIR."""
    # --- Configuration ---
    try:
        cfg = resolve_config(
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            scale=scale,
            device=device,
            config_file=config,
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
        console.print(f"[red]Run error:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        sr_model.unload()
        logger.close()

    # --- Manifest ---
    write_manifest(ctx, cfg, batch)

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


def run() -> None:
    app()
