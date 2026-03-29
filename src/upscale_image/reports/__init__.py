"""Run comparison and HTML report generation."""

from upscale_image.reports.compare import (
    ComparisonResult,
    RunDelta,
    RunSnapshot,
    compare_runs,
    comparison_to_dict,
    load_run_snapshot,
)
from upscale_image.reports.html import (
    generate_html_report,
    render_comparison_report,
    render_run_report,
)

__all__ = [
    "ComparisonResult",
    "RunDelta",
    "RunSnapshot",
    "compare_runs",
    "comparison_to_dict",
    "generate_html_report",
    "load_run_snapshot",
    "render_comparison_report",
    "render_run_report",
]
