"""shared/html_theme.py — Plotly theme + HTML report utilities.

Matches the MCP_Data_Analyst layout pattern exactly:
- CSS custom properties for light/dark/device themes
- Device-mode JS for auto system-preference detection
- Plotly CDN loaded in <head> (version-pinned)
- Fixed sidebar, responsive grid cards, styled tables
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plotly template mapping
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE: dict[str, str] = {
    "dark": "plotly_dark",
    "light": "plotly_white",
    "device": "plotly_white",  # device starts light, JS switches it
}


def plotly_template(theme: str) -> str:
    """Return the Plotly template string for a given theme."""
    return PLOTLY_TEMPLATE.get(theme, "plotly_dark")


# ---------------------------------------------------------------------------
# Viewport meta tag
# ---------------------------------------------------------------------------

VIEWPORT_META = '<meta name="viewport" content="width=device-width,initial-scale=1">'

# ---------------------------------------------------------------------------
# Plotly CDN (version-pinned)
# ---------------------------------------------------------------------------

PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

# ---------------------------------------------------------------------------
# CSS custom property blocks
# ---------------------------------------------------------------------------

_DARK_VARS = (
    "--bg:#0d1117;--surface:#161b22;--border:#21262d;--text:#c9d1d9;"
    "--text-muted:#8b949e;--accent:#58a6ff;--green:#3fb950;"
    "--orange:#f0883e;--red:#f85149;"
)
_LIGHT_VARS = (
    "--bg:#ffffff;--surface:#f6f8fa;--border:#d0d7de;--text:#1f2328;"
    "--text-muted:#636c76;--accent:#0969da;--green:#1a7f37;"
    "--orange:#9a6700;--red:#cf222e;"
)


def css_vars(theme: str) -> str:
    """Return a CSS :root{} block (and optional media query) for the theme."""
    if theme == "light":
        return f":root{{{_LIGHT_VARS}}}"
    elif theme == "device":
        return f":root{{{_LIGHT_VARS}}}@media(prefers-color-scheme:dark){{:root{{{_DARK_VARS}}}}}"
    else:  # dark (default)
        return f":root{{{_DARK_VARS}}}"


# ---------------------------------------------------------------------------
# Device-mode JS (auto-switches Plotly template + body bg on system pref)
# ---------------------------------------------------------------------------

_DEVICE_JS = """<script>
(function(){
  var DARK_BG='#0d1117',LIGHT_BG='#ffffff';
  function applyTheme(){
    var dark=window.matchMedia('(prefers-color-scheme:dark)').matches;
    document.body.style.background=dark?DARK_BG:LIGHT_BG;
    document.documentElement.setAttribute('data-theme',dark?'dark':'light');
    document.querySelectorAll('.plotly-graph-div').forEach(function(d){
      try{Plotly.relayout(d,{template:dark?'plotly_dark':'plotly_white'});}catch(e){}
    });
  }
  if(typeof Plotly!=='undefined'){applyTheme();}
  else{window.addEventListener('load',applyTheme);}
  window.matchMedia('(prefers-color-scheme:dark)').addEventListener('change',applyTheme);
})();
</script>"""


def device_mode_js() -> str:
    """Return device-mode theme switching JS snippet."""
    return _DEVICE_JS


# ---------------------------------------------------------------------------
# Theme colors for Plotly charts (used when embedding chart data as JSON)
# ---------------------------------------------------------------------------

THEMES: dict[str, dict] = {
    "light": {
        "plotly_template": "plotly_white",
        "bg_color": "#ffffff",
        "paper_color": "#f6f8fa",
        "text_color": "#1f2328",
        "grid_color": "#d0d7de",
        "accent": "#0969da",
        "success": "#1a7f37",
        "warning": "#9a6700",
        "danger": "#cf222e",
        "card_bg": "#ffffff",
        "sidebar_bg": "#f6f8fa",
        "sidebar_text": "#1f2328",
        "border_color": "#d0d7de",
    },
    "dark": {
        "plotly_template": "plotly_dark",
        "bg_color": "#0d1117",
        "paper_color": "#161b22",
        "text_color": "#c9d1d9",
        "grid_color": "#21262d",
        "accent": "#58a6ff",
        "success": "#3fb950",
        "warning": "#f0883e",
        "danger": "#f85149",
        "card_bg": "#161b22",
        "sidebar_bg": "#161b22",
        "sidebar_text": "#c9d1d9",
        "border_color": "#21262d",
    },
}
THEMES["device"] = THEMES["light"]


def get_theme(theme: str = "light") -> dict:
    """Return theme config dict. Falls back to light for unknown themes."""
    return THEMES.get(theme, THEMES["light"])


def theme_plot_colors(theme: str) -> tuple[str, str, str]:
    """Return (plot_bg, font_color, accent_color) for inline Plotly scripts."""
    if theme == "dark":
        return "#161b22", "#c9d1d9", "#58a6ff"
    return "#f6f8fa", "#1f2328", "#0969da"


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
# save_chart — standalone Plotly figure → HTML file
# ---------------------------------------------------------------------------


def save_chart(
    fig: object,
    output_path: str | Path,
    theme: str = "light",
    open_browser: bool = True,
    title: str = "",
) -> tuple[str, str]:
    """Save a Plotly figure as a standalone responsive HTML file.

    Returns (absolute_path_str, filename).
    """
    tmpl = plotly_template(theme)
    fig.update_layout(template=tmpl, autosize=True)  # type: ignore[attr-defined]

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    html = fig.to_html(  # type: ignore[attr-defined]
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displayModeBar": True, "scrollZoom": True},
    )

    # Inject viewport meta
    html = html.replace("<head>", f"<head>\n{VIEWPORT_META}", 1)
    if title:
        html = html.replace("<head>", f"<head>\n<title>{title}</title>", 1)

    # Inject device-mode JS and CSS media query
    if theme == "device":
        style_block = (
            "<style>"
            "@media(prefers-color-scheme:dark){html,body{background:#0d1117!important;}}"
            "@media(prefers-color-scheme:light){html,body{background:#ffffff!important;}}"
            "</style>"
        )
        html = html.replace("</head>", f"{style_block}\n</head>", 1)
        html = html.replace("</body>", f"{device_mode_js()}\n</body>", 1)
    else:
        bg = "#0d1117" if theme == "dark" else "#ffffff"
        html = html.replace(
            "</head>",
            f"<style>html,body{{background:{bg}!important;}}</style>\n</head>",
            1,
        )

    out.write_text(html, encoding="utf-8")
    if open_browser:
        _open_file(out)

    return str(out), out.name


# ---------------------------------------------------------------------------
# Report CSS — matches MCP_Data_Analyst _eda_css() pattern exactly
# ---------------------------------------------------------------------------


def report_css(vars_block: str) -> str:
    """Return the full CSS block for multi-section HTML reports."""
    return f"""{vars_block}
*{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;transition:background 0.2s,color 0.2s}}
::-webkit-scrollbar{{width:6px}}::-webkit-scrollbar-track{{background:var(--bg)}}::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px}}
.sidebar{{width:260px;background:var(--surface);border-right:1px solid var(--border);position:fixed;top:0;left:0;bottom:0;overflow-y:auto;z-index:100}}
.sidebar-hdr{{padding:20px;border-bottom:1px solid var(--border)}}
.sidebar-hdr h2{{color:var(--accent);font-size:16px;margin-bottom:4px}}
.sidebar-hdr .meta{{color:var(--text-muted);font-size:12px}}
.nav{{padding:8px 0}}
.nav a{{display:block;padding:7px 20px;color:var(--text-muted);text-decoration:none;font-size:13px;border-left:3px solid transparent;transition:all 0.15s}}
.nav a:hover,.nav a.active{{color:var(--accent);background:rgba(88,166,255,0.06);border-left-color:var(--accent)}}
.nav .st{{padding:14px 20px 4px;color:var(--border);font-size:10px;text-transform:uppercase;letter-spacing:1px;font-weight:600}}
.main{{margin-left:260px;padding:32px;min-height:100vh;overflow-x:auto}}
.section{{margin-bottom:48px}}
.section>h2{{color:var(--accent);font-size:20px;margin-bottom:20px;padding-bottom:10px;border-bottom:2px solid var(--border);font-weight:600}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:24px}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center;transition:transform 0.15s,border-color 0.15s}}
.card:hover{{transform:translateY(-2px);border-color:var(--accent)}}
.card .num{{font-size:28px;font-weight:700;color:var(--accent);line-height:1.2}}
.card .lbl{{font-size:11px;color:var(--text-muted);margin-top:4px;text-transform:uppercase;letter-spacing:0.8px}}
.card.good .num{{color:var(--green)}}.card.warn .num{{color:var(--orange)}}.card.bad .num{{color:var(--red)}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px;background:var(--surface);border-radius:8px;overflow:hidden}}
th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid var(--border)}}
th{{background:rgba(88,166,255,0.08);color:var(--accent);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:0.5px}}
tr:hover{{background:rgba(88,166,255,0.03)}}
.good{{color:var(--green)}}.warn{{color:var(--orange)}}.bad{{color:var(--red)}}
.badge{{font-size:11px;padding:2px 8px;border-radius:10px;background:var(--border);color:var(--text-muted);font-weight:500}}
.stats-cell{{font-size:12px;color:var(--text-muted);font-family:monospace}}
.insights{{list-style:none;padding:0}}
.insights li{{padding:10px 14px;margin:6px 0;background:var(--surface);border-radius:8px;border-left:4px solid var(--accent);font-size:13px;line-height:1.5}}
.insights li.warn{{border-left-color:var(--orange)}}.insights li.bad{{border-left-color:var(--red)}}.insights li.good{{border-left-color:var(--green)}}
.mbar{{height:24px;background:var(--border);border-radius:6px;overflow:hidden;margin:4px 0}}
.mbar-fill{{height:100%;background:linear-gradient(90deg,var(--orange),var(--red));border-radius:6px}}
.chart-container{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;margin:16px 0;min-height:420px;overflow:hidden;max-width:100%}}
pre{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;overflow-x:auto;font-size:13px;line-height:1.5;color:var(--text)}}
.alert-panel{{border-radius:10px;overflow:hidden;margin-bottom:20px}}
.alert-item{{padding:10px 14px;margin:3px 0;font-size:13px;border-radius:8px;display:flex;align-items:flex-start;gap:10px;background:var(--surface);border:1px solid var(--border)}}
.alert-item.error{{border-left:4px solid var(--red)}}.alert-item.warning{{border-left:4px solid var(--orange)}}.alert-item.info{{border-left:4px solid var(--green)}}
.alert-badge{{font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;white-space:nowrap;flex-shrink:0;margin-top:1px}}
.alert-badge.error{{background:var(--red);color:#fff}}.alert-badge.warning{{background:var(--orange);color:#fff}}.alert-badge.info{{background:var(--green);color:#fff}}
@media(max-width:1100px){{.sidebar{{width:220px}}.main{{margin-left:220px}}}}
@media(max-width:768px){{.sidebar{{display:none}}.main{{margin-left:0;padding:16px}}.cards{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:480px){{.cards{{grid-template-columns:1fr}}th,td{{padding:8px 10px;font-size:12px}}}}"""  # noqa: E501


# ---------------------------------------------------------------------------
# Multi-section HTML report builder
# ---------------------------------------------------------------------------


def build_html_report(
    title: str,
    subtitle: str,
    sections: list[dict],
    theme: str = "light",
    open_browser: bool = True,
    output_path: str | Path = "",
    sidebar_title: str = "",
    sidebar_meta: str = "",
) -> str:
    """Build a full multi-section HTML report with sidebar navigation.

    Each section dict: {"id": str, "heading": str, "html": str}

    Supports themes: "light", "dark", "device" (auto-detects system pref).
    Returns rendered HTML string. Writes to output_path if provided.
    """
    vars_css = css_vars(theme)
    css_block = report_css(vars_css)
    dev_js = device_mode_js() if theme == "device" else ""

    nav_links = "\n    ".join(f'<a href="#{s["id"]}">{s["heading"]}</a>' for s in sections)
    sections_html = "\n".join(
        f'<div id="{s["id"]}" class="section">\n  <h2>{s["heading"]}</h2>\n  {s["html"]}\n</div>' for s in sections
    )

    sb_title = sidebar_title or title
    sb_meta = f'<p class="meta">{sidebar_meta}</p>' if sidebar_meta else ""

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
{VIEWPORT_META}
<title>{title}</title>
{PLOTLY_CDN}
<style>
{css_block}
</style></head><body>
<div class="sidebar">
  <div class="sidebar-hdr">
    <h2>{sb_title}</h2>
    {sb_meta}
  </div>
  <div class="nav">
    <div class="st">Sections</div>
    {nav_links}
  </div>
</div>
<div class="main">
  {sections_html}
</div>
{dev_js}
</body></html>"""

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


def metrics_cards_html(metrics: dict, styles: dict[str, str] | None = None) -> str:
    """Render a dict of metrics as card HTML.

    Optional styles dict maps key -> CSS class (good/warn/bad).
    """
    if styles is None:
        styles = {}
    cards = []
    for k, v in metrics.items():
        cls = styles.get(k, "")
        cls_attr = f' class="card {cls}"' if cls else ' class="card"'
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            val = f"{v:.4f}" if abs(v) < 10 else f"{v:,.2f}"
        else:
            val = str(v)
        cards.append(f'<div{cls_attr}><div class="num">{val}</div><div class="lbl">{label}</div></div>')
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
    if len(rows) > max_rows:
        remaining = len(rows) - max_rows
        trs += (
            f'<tr><td colspan="{len(headers)}" '
            f'style="text-align:center;color:var(--text-muted);font-style:italic">'
            f"&hellip; {remaining} more rows</td></tr>"
        )
    return f'<div style="overflow-x:auto"><table><tr>{th}</tr>{trs}</table></div>'


def plotly_div(fig: object, height: int = 450) -> str:
    """Embed a Plotly figure as an inline div.

    Uses Plotly CDN loaded from build_html_report <head>.
    """
    import plotly.io as pio  # lazy

    return (
        f'<div class="chart-container" style="min-height:{height}px">'
        + pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=False,
            config={
                "responsive": True,
                "displayModeBar": True,
                "scrollZoom": True,
            },
        )
        + "</div>"
    )
