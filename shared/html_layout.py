"""shared/html_layout.py — Layout helpers, output path, Plotly base config.

This module owns:
- VIEWPORT_META — single source of truth for the viewport meta tag
- get_output_path() — input-file-first output path resolution
- plotly_layout_base() — base Plotly layout dict (no height)
- PLOTLY_CFG_JS — standard Plotly config as inline JS string
- plotly_config() — standard Plotly config as Python dict
"""

from __future__ import annotations

from pathlib import Path

from shared.file_utils import get_default_output_dir

# ---------------------------------------------------------------------------
# Viewport meta — single source of truth, import everywhere else
# ---------------------------------------------------------------------------

VIEWPORT_META = '<meta name="viewport" content="width=device-width,initial-scale=1">'

# ---------------------------------------------------------------------------
# Standard Plotly config
# ---------------------------------------------------------------------------

PLOTLY_CFG_JS = '{"responsive":true,"displayModeBar":true,"scrollZoom":true,"plotGlPixelRatio":0}'


def plotly_config() -> dict:
    """Return standard Plotly config dict for fig.show() or pio.to_html()."""
    return {
        "responsive": True,
        "displayModeBar": True,
        "scrollZoom": True,
        "plotGlPixelRatio": 0,
    }


# ---------------------------------------------------------------------------
# Base Plotly layout — no height (CSS controls chart height)
# ---------------------------------------------------------------------------


def plotly_layout_base(
    plot_bg: str,
    font_color: str,
    margin: dict | None = None,
) -> dict:
    """Return a base Plotly layout dict. Never includes height — CSS controls that."""
    return {
        "paper_bgcolor": plot_bg,
        "plot_bgcolor": plot_bg,
        "font": {"color": font_color},
        "margin": margin or {"l": 50, "r": 20, "t": 20, "b": 40},
        "autosize": True,
    }


# ---------------------------------------------------------------------------
# Output path — input-file-first
# ---------------------------------------------------------------------------


def get_output_path(
    output_path: str,
    input_path: Path | None,
    stem_suffix: str,
    ext: str = "html",
) -> Path:
    """Resolve output path.

    Priority:
      1. Explicit output_path argument if given
      2. Same directory as input file (when input_path is provided)
      3. ~/Downloads/<stem>_<suffix>.<ext>  (pure generation, no input file)
    """
    if output_path:
        return Path(output_path).resolve()
    base_dir = get_default_output_dir(str(input_path) if input_path is not None else None)
    stem = input_path.stem if input_path is not None else stem_suffix
    return base_dir / f"{stem}_{stem_suffix}.{ext}"
