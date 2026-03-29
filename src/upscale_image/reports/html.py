"""HTML report generator: human-readable presentation of run artefacts.

HTML is the presentation layer only. All data is read from manifest.json
and metrics/ files — no metrics are recalculated here.

ADR 0009: HTML depends on manifest and metrics, does not replace them.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

from upscale_image.reports.compare import (
    ComparisonResult,
    RunSnapshot,
    compare_runs,
    comparison_to_dict,
)

# ---------------------------------------------------------------------------
# CSS (inline, no external dependencies)
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: monospace; margin: 2rem; color: #222; background: #fafafa; }
h1 { color: #333; border-bottom: 2px solid #ccc; padding-bottom: 0.3rem; }
h2 { color: #555; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
th { background: #ddd; text-align: left; padding: 0.4rem 0.8rem; }
td { padding: 0.35rem 0.8rem; border-bottom: 1px solid #e0e0e0; }
tr:nth-child(even) td { background: #f5f5f5; }
.delta-pos { color: #c0392b; }
.delta-neg { color: #27ae60; }
.delta-zero { color: #888; }
.na { color: #aaa; font-style: italic; }
.tag { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px;
       background: #e8e8e8; font-size: 0.85em; }
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_run_report(snapshot: RunSnapshot) -> str:
    """Return a standalone HTML string for a single run."""
    body = _section_run(snapshot)
    return _wrap_html(f"Run Report — {_e(snapshot.run_id)}", body)


def render_comparison_report(result: ComparisonResult) -> str:
    """Return a standalone HTML string for a multi-run comparison."""
    parts: list[str] = []

    if len(result.snapshots) == 1:
        parts.append(_section_run(result.snapshots[0]))
    else:
        parts.append(_section_comparison_table(result))
        parts.append(_section_delta_table(result))

    title = "Comparison Report — " + ", ".join(
        _e(s.run_id) for s in result.snapshots
    )
    return _wrap_html(title, "\n".join(parts))


def generate_html_report(
    run_dirs: list[str | Path],
    output_path: str | Path,
) -> Path:
    """Load runs, render HTML and write to *output_path*.

    Args:
        run_dirs:    One or more run directory paths.
        output_path: Destination .html file.

    Returns:
        Resolved path of the written file.
    """
    result = compare_runs(run_dirs)
    html = render_comparison_report(result)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Internal rendering helpers
# ---------------------------------------------------------------------------

def _e(text: str) -> str:
    """HTML-escape a string."""
    return _html.escape(str(text))


def _fmt(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return '<span class="na">—</span>'
    return f"{value:.{decimals}f}"


def _fmt_delta(value: float | None, decimals: int = 3, lower_is_better: bool = False) -> str:
    """Format a delta value with colour coding."""
    if value is None:
        return '<span class="na">—</span>'
    if abs(value) < 1e-9:
        return f'<span class="delta-zero">±0</span>'
    positive_is_good = not lower_is_better
    css = "delta-neg" if (value > 0) == positive_is_good else "delta-pos"
    sign = "+" if value > 0 else ""
    return f'<span class="{css}">{sign}{value:.{decimals}f}</span>'


def _wrap_html(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        f"  <meta charset='utf-8'>\n"
        f"  <title>{title}</title>\n"
        f"  <style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{title}</h1>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def _section_run(s: RunSnapshot) -> str:
    rows = [
        ("Run ID", _e(s.run_id)),
        ("Model", f'<span class="tag">{_e(s.model_name)}</span>'),
        ("Scale", f"{s.model_scale}×"),
        ("Device", _e(s.device)),
        ("Precision", _e(s.precision)),
        ("Total elapsed (s)", _fmt(s.total_elapsed_s)),
        ("Avg inference (ms)", _fmt(s.avg_inference_ms)),
        ("Images: total / done / failed / skipped",
         f"{s.total_images} / {s.done} / {s.failed} / {s.skipped}"),
        ("Success rate", _fmt(s.success_rate, 4)),
        ("PSNR (avg)", _fmt(s.avg_psnr, 2)),
        ("SSIM (avg)", _fmt(s.avg_ssim, 4)),
        ("LPIPS (avg)", _fmt(s.avg_lpips, 4)),
        ("NIQE (avg)", _fmt(s.avg_niqe, 3)),
    ]
    html_rows = "".join(
        f"<tr><th>{label}</th><td>{value}</td></tr>\n"
        for label, value in rows
    )
    return f"<h2>Run summary</h2>\n<table>\n{html_rows}</table>\n"


def _section_comparison_table(result: ComparisonResult) -> str:
    """One column per run, one row per metric."""
    snapshots = result.snapshots
    header_cells = "<th>Metric</th>" + "".join(
        f"<th>{_e(s.run_id)}</th>" for s in snapshots
    )

    def row(label: str, getter, decimals: int = 3) -> str:
        cells = "".join(
            f"<td>{_fmt(getter(s), decimals)}</td>" for s in snapshots
        )
        return f"<tr><th>{label}</th>{cells}</tr>\n"

    rows = (
        row("Model", lambda s: None)  # placeholder replaced below
        + row("Scale", lambda s: s.model_scale, 0)
        + row("Device", lambda s: None)
        + row("Precision", lambda s: None)
        + row("Total elapsed (s)", lambda s: s.total_elapsed_s)
        + row("Avg inference (ms)", lambda s: s.avg_inference_ms)
        + row("Success rate", lambda s: s.success_rate, 4)
        + row("PSNR (avg)", lambda s: s.avg_psnr, 2)
        + row("SSIM (avg)", lambda s: s.avg_ssim, 4)
        + row("LPIPS (avg)", lambda s: s.avg_lpips, 4)
        + row("NIQE (avg)", lambda s: s.avg_niqe, 3)
    )

    # Override text-only rows
    def text_row(label: str, getter) -> str:
        cells = "".join(f"<td>{_e(str(getter(s)))}</td>" for s in snapshots)
        return f"<tr><th>{label}</th>{cells}</tr>\n"

    rows_final = (
        text_row("Model", lambda s: s.model_name)
        + row("Scale", lambda s: s.model_scale, 0)
        + text_row("Device", lambda s: s.device)
        + text_row("Precision", lambda s: s.precision)
        + row("Total elapsed (s)", lambda s: s.total_elapsed_s)
        + row("Avg inference (ms)", lambda s: s.avg_inference_ms)
        + row("Success rate", lambda s: s.success_rate, 4)
        + row("PSNR (avg)", lambda s: s.avg_psnr, 2)
        + row("SSIM (avg)", lambda s: s.avg_ssim, 4)
        + row("LPIPS (avg)", lambda s: s.avg_lpips, 4)
        + row("NIQE (avg)", lambda s: s.avg_niqe, 3)
    )

    return (
        "<h2>Run comparison</h2>\n"
        f"<table>\n<tr>{header_cells}</tr>\n{rows_final}</table>\n"
    )


def _section_delta_table(result: ComparisonResult) -> str:
    """One row per delta (sequential: run[i] vs run[i-1])."""
    if not result.deltas:
        return ""

    header = (
        "<tr>"
        "<th>Metric</th>"
        + "".join(
            f"<th>{_e(d.run_id_a)} → {_e(d.run_id_b)}</th>"
            for d in result.deltas
        )
        + "</tr>\n"
    )

    def delta_row(label: str, getter, lower_is_better: bool = False) -> str:
        cells = "".join(
            f"<td>{_fmt_delta(getter(d), lower_is_better=lower_is_better)}</td>"
            for d in result.deltas
        )
        return f"<tr><th>{label}</th>{cells}</tr>\n"

    rows = (
        delta_row("Δ Total elapsed (s)", lambda d: d.delta_total_elapsed_s, lower_is_better=True)
        + delta_row("Δ Avg inference (ms)", lambda d: d.delta_avg_inference_ms, lower_is_better=True)
        + delta_row("Δ Success rate", lambda d: d.delta_success_rate)
        + delta_row("Δ PSNR", lambda d: d.delta_avg_psnr)
        + delta_row("Δ SSIM", lambda d: d.delta_avg_ssim)
        + delta_row("Δ LPIPS", lambda d: d.delta_avg_lpips, lower_is_better=True)
        + delta_row("Δ NIQE", lambda d: d.delta_avg_niqe, lower_is_better=True)
    )

    return f"<h2>Deltas (B − A)</h2>\n<table>\n{header}{rows}</table>\n"
