"""shared/html_theme.py — Plotly theme + save_chart + _open_file helpers."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme configuration
# ---------------------------------------------------------------------------

THEMES: dict[str, dict] = {
    "light": {
        "plotly_template": "plotly",
        "bg_color": "#ffffff",
        "paper_color": "#f8f9fa",
        "text_color": "#212529",
        "grid_color": "#dee2e6",
        "accent": "#0d6efd",
        "success": "#198754",
        "warning": "#ffc107",
        "danger": "#dc3545",
        "card_bg": "#ffffff",
        "sidebar_bg": "#f8f9fa",
        "sidebar_text": "#212529",
        "border_color": "#dee2e6",
    },
    "dark": {
        "plotly_template": "plotly_dark",
        "bg_color": "#1a1a2e",
        "paper_color": "#16213e",
        "text_color": "#e0e0e0",
        "grid_color": "#2d3748",
        "accent": "#4dabf7",
        "success": "#51cf66",
        "warning": "#ffd43b",
        "danger": "#ff6b6b",
        "card_bg": "#16213e",
        "sidebar_bg": "#0f3460",
        "sidebar_text": "#e0e0e0",
        "border_color": "#2d3748",
    },
}


def get_theme(theme: str = "light") -> dict:
    """Return theme config dict. Falls back to light for unknown themes."""
    return THEMES.get(theme, THEMES["light"])


# ---------------------------------------------------------------------------
# Browser auto-open (cross-platform)
# ---------------------------------------------------------------------------


def _open_file(path: str | Path) -> None:
    """Open file in default browser/app. Best-effort, never raises."""
    p = Path(path).resolve()
    try:
        if sys.platform == "win32":
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception as exc:
        logger.debug("_open_file failed for %s: %s", p, exc)


# ---------------------------------------------------------------------------
# Plotly chart → HTML file
# ---------------------------------------------------------------------------


def save_chart(
    fig: object,
    output_path: str | Path,
    theme: str = "light",
    open_browser: bool = True,
    title: str = "",
) -> tuple[str, str]:
    """Save a Plotly figure as a standalone HTML file.

    Returns (absolute_path_str, filename).
    """
    import plotly.io as pio  # lazy

    t = get_theme(theme)
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Apply theme to figure
    fig.update_layout(  # type: ignore[attr-defined]
        template=t["plotly_template"],
        paper_bgcolor=t["paper_color"],
        plot_bgcolor=t["bg_color"],
        font_color=t["text_color"],
    )

    html_str = pio.to_html(
        fig,
        full_html=True,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": True, "scrollZoom": False},
    )

    # Inject viewport meta for mobile responsiveness
    viewport = '<meta name="viewport" content="width=device-width, initial-scale=1">'
    if title:
        html_str = html_str.replace(
            "<head>", f"<head>\n  {viewport}\n  <title>{title}</title>"
        )
    else:
        html_str = html_str.replace("<head>", f"<head>\n  {viewport}")

    out.write_text(html_str, encoding="utf-8")

    if open_browser:
        _open_file(out)

    return str(out), out.name


# ---------------------------------------------------------------------------
# Multi-section HTML report builder
# ---------------------------------------------------------------------------

def _build_css(t: dict) -> str:
    """Build CSS string with theme values substituted (avoids .format() brace conflicts)."""
    rules = [
        "* { box-sizing: border-box; margin: 0; padding: 0; }",
        f"body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: {t['bg_color']}; color: {t['text_color']}; }}",
        f".layout {{ display: flex; min-height: 100vh; }}",
        f".sidebar {{ width: 220px; min-height: 100vh; background: {t['sidebar_bg']}; color: {t['sidebar_text']}; padding: 1.5rem 1rem; position: sticky; top: 0; flex-shrink: 0; }}",
        f".sidebar h2 {{ font-size: 0.9rem; text-transform: uppercase; letter-spacing: .05em; color: {t['accent']}; margin-bottom: 1rem; }}",
        f".sidebar a {{ display: block; padding: 0.4rem 0.6rem; border-radius: 4px; color: {t['sidebar_text']}; text-decoration: none; font-size: 0.85rem; margin-bottom: 0.2rem; }}",
        f".sidebar a:hover {{ background: {t['accent']}; color: #fff; }}",
        ".content { flex: 1; padding: 2rem; max-width: 1200px; }",
        f"h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; color: {t['text_color']}; }}",
        f".subtitle {{ font-size: 0.9rem; color: {t['grid_color']}; margin-bottom: 2rem; }}",
        ".section { margin-bottom: 2.5rem; }",
        f".section h2 {{ font-size: 1.1rem; border-bottom: 2px solid {t['accent']}; padding-bottom: 0.4rem; margin-bottom: 1rem; color: {t['text_color']}; }}",
        ".cards { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem; }",
        f".card {{ background: {t['card_bg']}; border: 1px solid {t['border_color']}; border-radius: 8px; padding: 1rem 1.25rem; min-width: 140px; flex: 1; }}",
        f".card .label {{ font-size: 0.75rem; text-transform: uppercase; color: {t['grid_color']}; letter-spacing: .05em; }}",
        f".card .value {{ font-size: 1.5rem; font-weight: 700; color: {t['accent']}; }}",
        f".card .sub {{ font-size: 0.8rem; color: {t['grid_color']}; margin-top: 0.2rem; }}",
        "table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem; }",
        f"th {{ background: {t['paper_color']}; color: {t['text_color']}; padding: 0.5rem 0.75rem; text-align: left; border-bottom: 2px solid {t['border_color']}; }}",
        f"td {{ padding: 0.4rem 0.75rem; border-bottom: 1px solid {t['border_color']}; color: {t['text_color']}; }}",
        f"tr:hover td {{ background: {t['paper_color']}; }}",
        f".chart-container {{ width: 100%; overflow: hidden; border-radius: 8px; border: 1px solid {t['border_color']}; margin-top: 0.5rem; }}",
        ".alert { padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.5rem; font-size: 0.85rem; }",
        f".alert-warning {{ background: {t['warning']}22; border-left: 3px solid {t['warning']}; }}",
        f".alert-success {{ background: {t['success']}22; border-left: 3px solid {t['success']}; }}",
        f".alert-danger  {{ background: {t['danger']}22;  border-left: 3px solid {t['danger']};  }}",
        "@media (max-width: 768px) { .sidebar { display: none; } .cards { flex-direction: column; } }",
    ]
    return "\n".join(rules)


def build_html_report(
    title: str,
    subtitle: str,
    sections: list[dict],
    theme: str = "light",
    open_browser: bool = True,
    output_path: str | Path = "",
) -> str:
    """Build a full multi-section HTML report.

    Each section dict:
        {"id": str, "heading": str, "html": str}

    Returns rendered HTML string. Writes to output_path if provided and auto-opens.
    """
    t = get_theme(theme)
    css = _build_css(t)

    nav_links = "\n".join(
        f'<a href="#{s["id"]}">{s["heading"]}</a>' for s in sections
    )
    sections_html = "\n".join(
        f'<div class="section" id="{s["id"]}"><h2>{s["heading"]}</h2>{s["html"]}</div>'
        for s in sections
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
<div class="layout">
  <nav class="sidebar">
    <h2>Contents</h2>
    {nav_links}
  </nav>
  <main class="content">
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>
    {sections_html}
  </main>
</div>
</body>
</html>"""

    if output_path:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        if open_browser:
            _open_file(out)

    return html


# ---------------------------------------------------------------------------
# Small helpers for report sections
# ---------------------------------------------------------------------------


def metrics_cards_html(metrics: dict) -> str:
    """Render a dict of metrics as card HTML."""
    cards = []
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            val = f"{v:.4f}" if abs(v) < 10 else f"{v:,.2f}"
        else:
            val = str(v)
        cards.append(
            f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value">{val}</div></div>'
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def data_table_html(rows: list[dict], max_rows: int = 50) -> str:
    """Render a list of dicts as an HTML table."""
    if not rows:
        return "<p>No data.</p>"
    headers = list(rows[0].keys())
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for row in rows[:max_rows]:
        tds = "".join(f"<td>{row.get(h, '')}</td>" for h in headers)
        trs += f"<tr>{tds}</tr>"
    return f"<div style='overflow-x:auto'><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>"


def plotly_div(fig: object, height: int = 450) -> str:
    """Embed a Plotly figure as an inline div (no full HTML wrapper)."""
    import plotly.io as pio  # lazy
    return f'<div class="chart-container" style="height:{height}px">' + \
           pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"responsive": True, "displayModeBar": True}) + \
           "</div>"
