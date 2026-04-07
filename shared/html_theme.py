"""shared/html_theme.py — HTML report builder with light/dark/device themes,
responsive layout, and Plotly integration."""

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
    "light": "plotly_white",
    "dark": "plotly_dark",
    "device": "plotly_white",  # device starts light; JS switches at runtime
}


def plotly_template(theme: str) -> str:
    """Return the Plotly template name for a given theme."""
    return PLOTLY_TEMPLATE.get(theme, "plotly_white")


# ---------------------------------------------------------------------------
# CSS custom properties (design tokens)
# ---------------------------------------------------------------------------

_LIGHT_VARS = (
    "--bg:#ffffff;"
    "--surface:#f6f8fa;"
    "--surface-hover:#eaeef2;"
    "--border:#d0d7de;"
    "--border-subtle:#e8ebef;"
    "--text:#1f2328;"
    "--text-secondary:#656d76;"
    "--text-muted:#8b949e;"
    "--accent:#0969da;"
    "--accent-hover:#0550ae;"
    "--accent-subtle:rgba(9,105,218,0.08);"
    "--green:#1a7f37;"
    "--green-subtle:rgba(26,127,55,0.1);"
    "--orange:#9a6700;"
    "--orange-subtle:rgba(154,103,0,0.1);"
    "--red:#cf222e;"
    "--red-subtle:rgba(207,34,46,0.1);"
    "--sidebar-bg:#f6f8fa;"
    "--sidebar-text:#1f2328;"
    "--sidebar-active:rgba(9,105,218,0.12);"
    "--card-shadow:0 1px 3px rgba(0,0,0,0.06),0 1px 2px rgba(0,0,0,0.04);"
    "--card-shadow-hover:0 4px 12px rgba(0,0,0,0.08),0 2px 4px rgba(0,0,0,0.04);"
    "--overlay-bg:rgba(0,0,0,0.3);"
    "--radius:8px;"
    "--radius-sm:6px;"
    "--transition:0.2s cubic-bezier(0.4,0,0.2,1);"
)

_DARK_VARS = (
    "--bg:#0d1117;"
    "--surface:#161b22;"
    "--surface-hover:#1c2129;"
    "--border:#30363d;"
    "--border-subtle:#21262d;"
    "--text:#e6edf3;"
    "--text-secondary:#9da5b0;"
    "--text-muted:#6e7681;"
    "--accent:#58a6ff;"
    "--accent-hover:#79c0ff;"
    "--accent-subtle:rgba(88,166,255,0.1);"
    "--green:#3fb950;"
    "--green-subtle:rgba(63,185,80,0.1);"
    "--orange:#f0883e;"
    "--orange-subtle:rgba(240,136,62,0.1);"
    "--red:#f85149;"
    "--red-subtle:rgba(248,81,73,0.1);"
    "--sidebar-bg:#161b22;"
    "--sidebar-text:#e6edf3;"
    "--sidebar-active:rgba(88,166,255,0.15);"
    "--card-shadow:0 1px 3px rgba(0,0,0,0.3),0 1px 2px rgba(0,0,0,0.2);"
    "--card-shadow-hover:0 4px 12px rgba(0,0,0,0.4),0 2px 4px rgba(0,0,0,0.2);"
    "--overlay-bg:rgba(0,0,0,0.6);"
    "--radius:8px;"
    "--radius-sm:6px;"
    "--transition:0.2s cubic-bezier(0.4,0,0.2,1);"
)


def css_vars(theme: str) -> str:
    """Return CSS :root{} block (with optional media query for device mode)."""
    if theme == "light":
        return f":root{{{_LIGHT_VARS}}}"
    if theme == "device":
        return f":root{{{_LIGHT_VARS}}}@media(prefers-color-scheme:dark){{:root{{{_DARK_VARS}}}}}"
    return f":root{{{_DARK_VARS}}}"


# ---------------------------------------------------------------------------
# Legacy theme dict (backward-compatible for callers using get_theme())
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
        "text_color": "#e6edf3",
        "grid_color": "#30363d",
        "accent": "#58a6ff",
        "success": "#3fb950",
        "warning": "#f0883e",
        "danger": "#f85149",
        "card_bg": "#161b22",
        "sidebar_bg": "#161b22",
        "sidebar_text": "#e6edf3",
        "border_color": "#30363d",
    },
}
# Device mode starts as light — JS handles runtime switching.
THEMES["device"] = THEMES["light"]


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
# Device-mode JS — auto-detects system preference and live-switches theme
# ---------------------------------------------------------------------------

_DEVICE_JS = """<script>
(function(){
  var DARK='dark',LIGHT='light';
  function getPreferred(){return window.matchMedia('(prefers-color-scheme:dark)').matches?DARK:LIGHT;}
  function apply(mode){
    var d=document.documentElement;
    d.setAttribute('data-theme',mode);
    /* Switch Plotly chart templates */
    document.querySelectorAll('.plotly-graph-div').forEach(function(el){
      try{Plotly.relayout(el,{template:mode===DARK?'plotly_dark':'plotly_white'});}catch(e){}
    });
    /* Update toggle icon */
    var btn=document.getElementById('theme-toggle');
    if(btn){btn.textContent=mode===DARK?'\\u2600':'\\u263E';}
  }
  /* Apply on load */
  var saved=localStorage.getItem('mcp-theme');
  var mode=saved||(getPreferred());
  apply(mode);
  /* Listen for system changes (only if no manual override) */
  window.matchMedia('(prefers-color-scheme:dark)').addEventListener('change',function(e){
    if(!localStorage.getItem('mcp-theme')){apply(e.matches?DARK:LIGHT);}
  });
  /* Theme toggle button handler */
  document.addEventListener('click',function(e){
    if(e.target&&e.target.id==='theme-toggle'){
      var cur=document.documentElement.getAttribute('data-theme')||getPreferred();
      var next=cur===DARK?LIGHT:DARK;
      localStorage.setItem('mcp-theme',next);
      apply(next);
    }
  });
  /* Mobile sidebar toggle */
  document.addEventListener('click',function(e){
    var sidebar=document.getElementById('sidebar');
    var overlay=document.getElementById('sidebar-overlay');
    if(!sidebar)return;
    if(e.target&&e.target.id==='menu-toggle'){
      sidebar.classList.toggle('open');
      if(overlay)overlay.classList.toggle('visible');
    }
    if(e.target&&e.target.id==='sidebar-overlay'){
      sidebar.classList.remove('open');
      overlay.classList.remove('visible');
    }
    /* Close sidebar on nav link click (mobile) */
    if(e.target&&e.target.closest&&e.target.closest('.sidebar a')){
      if(window.innerWidth<=768){
        sidebar.classList.remove('open');
        if(overlay)overlay.classList.remove('visible');
      }
    }
  });
})();
</script>"""


def device_mode_js() -> str:
    """Return the device-mode theme switching JavaScript."""
    return _DEVICE_JS


# ---------------------------------------------------------------------------
# Core CSS — uses custom properties for theming
# ---------------------------------------------------------------------------

_CORE_CSS = """
/* Reset */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}

/* Base */
html{scroll-behavior:smooth;-webkit-text-size-adjust:100%;}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif;
  font-size:15px;line-height:1.6;
  background:var(--bg);color:var(--text);
  transition:background var(--transition),color var(--transition);
}

/* Layout */
.layout{display:flex;min-height:100vh;position:relative;}

/* Sidebar overlay (mobile) */
.sidebar-overlay{
  display:none;position:fixed;inset:0;background:var(--overlay-bg);
  z-index:90;opacity:0;transition:opacity var(--transition);
}
.sidebar-overlay.visible{display:block;opacity:1;}

/* Sidebar */
.sidebar{
  width:240px;min-height:100vh;
  background:var(--sidebar-bg);color:var(--sidebar-text);
  padding:1.25rem 0.75rem;position:sticky;top:0;align-self:flex-start;
  flex-shrink:0;border-right:1px solid var(--border-subtle);
  transition:background var(--transition),border-color var(--transition);
  overflow-y:auto;max-height:100vh;z-index:100;
}
.sidebar-header{
  display:flex;align-items:center;justify-content:space-between;
  padding:0 0.5rem;margin-bottom:1rem;
}
.sidebar-header h2{
  font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;
  color:var(--text-muted);font-weight:600;margin:0;
}
.sidebar a{
  display:block;padding:0.4rem 0.75rem;border-radius:var(--radius-sm);
  color:var(--sidebar-text);text-decoration:none;font-size:0.84rem;
  margin-bottom:2px;transition:all var(--transition);white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;
}
.sidebar a:hover{background:var(--sidebar-active);color:var(--accent);}

/* Theme toggle button */
.theme-toggle{
  width:32px;height:32px;border-radius:50%;border:1px solid var(--border);
  background:var(--surface);color:var(--text-secondary);cursor:pointer;
  font-size:1rem;display:flex;align-items:center;justify-content:center;
  transition:all var(--transition);flex-shrink:0;
}
.theme-toggle:hover{background:var(--accent-subtle);color:var(--accent);border-color:var(--accent);}

/* Mobile hamburger */
.menu-toggle{
  display:none;position:fixed;top:0.75rem;left:0.75rem;z-index:110;
  width:36px;height:36px;border-radius:var(--radius-sm);border:1px solid var(--border);
  background:var(--surface);color:var(--text);cursor:pointer;font-size:1.2rem;
  align-items:center;justify-content:center;
  box-shadow:var(--card-shadow);transition:all var(--transition);
}

/* Content */
.content{
  flex:1;padding:2.5rem 3rem;max-width:1100px;min-width:0;
}
.content h1{
  font-size:1.5rem;font-weight:700;margin-bottom:0.25rem;
  color:var(--text);line-height:1.3;
}
.subtitle{
  font-size:0.875rem;color:var(--text-muted);margin-bottom:2.5rem;
  line-height:1.5;
}

/* Sections */
.section{margin-bottom:3rem;}
.section h2{
  font-size:1.05rem;font-weight:600;color:var(--text);
  padding-bottom:0.5rem;margin-bottom:1.25rem;
  border-bottom:2px solid var(--accent);
}

/* Metric cards */
.cards{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));
  gap:0.875rem;margin-bottom:1.25rem;
}
.card{
  background:var(--surface);border:1px solid var(--border-subtle);
  border-radius:var(--radius);padding:1rem 1.15rem;
  transition:all var(--transition);box-shadow:var(--card-shadow);
}
.card:hover{box-shadow:var(--card-shadow-hover);border-color:var(--border);}
.card .label{
  font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;
  color:var(--text-muted);font-weight:600;margin-bottom:0.25rem;
}
.card .value{
  font-size:1.4rem;font-weight:700;color:var(--accent);
  line-height:1.2;word-break:break-word;
}
.card .sub{font-size:0.78rem;color:var(--text-muted);margin-top:0.2rem;}

/* Tables */
.table-wrap{overflow-x:auto;border-radius:var(--radius);border:1px solid var(--border-subtle);margin-top:0.5rem;}
table{width:100%;border-collapse:collapse;font-size:0.84rem;}
th{
  background:var(--surface);color:var(--text-secondary);
  padding:0.6rem 0.85rem;text-align:left;font-weight:600;font-size:0.78rem;
  text-transform:uppercase;letter-spacing:0.04em;
  border-bottom:2px solid var(--border);
  position:sticky;top:0;z-index:1;
  transition:background var(--transition);
}
td{
  padding:0.5rem 0.85rem;border-bottom:1px solid var(--border-subtle);
  color:var(--text);transition:background var(--transition);
}
tr:hover td{background:var(--surface-hover);}

/* Chart containers */
.chart-container{
  width:100%;overflow:hidden;border-radius:var(--radius);
  border:1px solid var(--border-subtle);margin-top:0.75rem;
  background:var(--surface);box-shadow:var(--card-shadow);
  transition:all var(--transition);
}

/* Alerts */
.alert{
  padding:0.8rem 1rem;border-radius:var(--radius-sm);
  margin-bottom:0.625rem;font-size:0.84rem;line-height:1.5;
  transition:all var(--transition);
}
.alert em{display:block;margin-top:0.3rem;opacity:0.85;font-size:0.8rem;}
.alert-warning{background:var(--orange-subtle);border-left:3px solid var(--orange);color:var(--text);}
.alert-success{background:var(--green-subtle);border-left:3px solid var(--green);color:var(--text);}
.alert-danger{background:var(--red-subtle);border-left:3px solid var(--red);color:var(--text);}

/* Code / pre blocks */
pre{
  background:var(--surface);border:1px solid var(--border-subtle);
  border-radius:var(--radius-sm);padding:1rem;overflow-x:auto;
  font-size:0.84rem;line-height:1.6;color:var(--text);
  transition:background var(--transition);
}

/* Print styles */
@media print{
  .sidebar,.sidebar-overlay,.menu-toggle,.theme-toggle{display:none!important;}
  .content{padding:1rem!important;max-width:100%!important;}
  .card{break-inside:avoid;}
  body{background:#fff!important;color:#000!important;}
}

/* Responsive: Tablet */
@media(max-width:1024px){
  .content{padding:2rem 1.5rem;}
  .cards{grid-template-columns:repeat(auto-fill,minmax(140px,1fr));}
}

/* Responsive: Mobile */
@media(max-width:768px){
  .menu-toggle{display:flex;}
  .sidebar{
    position:fixed;left:-260px;top:0;width:260px;height:100vh;
    transition:left var(--transition);border-right:1px solid var(--border);
    box-shadow:none;
  }
  .sidebar.open{left:0;box-shadow:4px 0 24px rgba(0,0,0,0.15);}
  .content{padding:3.5rem 1rem 2rem 1rem;}
  .content h1{font-size:1.25rem;}
  .cards{grid-template-columns:1fr 1fr;gap:0.625rem;}
  .card{padding:0.75rem 0.9rem;}
  .card .value{font-size:1.15rem;}
  table{font-size:0.78rem;}
  th,td{padding:0.4rem 0.6rem;}
  .section{margin-bottom:2rem;}
  .section h2{font-size:0.95rem;}
}

/* Responsive: Small phone */
@media(max-width:420px){
  .cards{grid-template-columns:1fr;}
  .content{padding:3rem 0.75rem 1.5rem 0.75rem;}
}
"""


# ---------------------------------------------------------------------------
# Plotly chart → standalone HTML file
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
    import plotly.io as pio  # lazy

    tmpl = plotly_template(theme)
    fig.update_layout(template=tmpl, autosize=True)  # type: ignore[attr-defined]

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    html_str = pio.to_html(
        fig,
        full_html=True,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": True, "scrollZoom": False},
    )

    # Inject viewport meta + title
    viewport = '<meta name="viewport" content="width=device-width, initial-scale=1">'
    inject_head = f"<head>\n  {viewport}"
    if title:
        inject_head += f"\n  <title>{title}</title>"

    html_str = html_str.replace("<head>", inject_head, 1)

    # Theme-aware background
    if theme == "device":
        style_block = (
            "<style>"
            f"{css_vars('device')}"
            "html,body{background:var(--bg)!important;transition:background 0.2s;}"
            "</style>"
        )
        html_str = html_str.replace("</head>", f"{style_block}\n</head>", 1)
        html_str = html_str.replace("</body>", f"{device_mode_js()}\n</body>", 1)
    else:
        bg = "#0d1117" if theme == "dark" else "#ffffff"
        fg = "#e6edf3" if theme == "dark" else "#1f2328"
        html_str = html_str.replace(
            "</head>",
            f"<style>html,body{{background:{bg}!important;color:{fg}!important;}}</style>\n</head>",
            1,
        )

    out.write_text(html_str, encoding="utf-8")

    if open_browser:
        _open_file(out)

    return str(out), out.name


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
) -> str:
    """Build a full multi-section HTML report with sidebar navigation.

    Each section dict: {"id": str, "heading": str, "html": str}

    Supports themes: "light", "dark", "device" (auto-detects system preference).
    Returns rendered HTML string. Writes to output_path if provided and auto-opens.
    """
    vars_css = css_vars(theme)
    initial_icon = "&#9790;" if theme == "light" else ("&#9728;" if theme == "dark" else "&#9790;")

    nav_links = "\n    ".join(f'<a href="#{s["id"]}">{s["heading"]}</a>' for s in sections)
    sections_html = "\n".join(
        f'<div class="section" id="{s["id"]}">\n  <h2>{s["heading"]}</h2>\n  {s["html"]}\n</div>' for s in sections
    )

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="{theme if theme != "device" else "light"}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{vars_css}{_CORE_CSS}</style>
</head>
<body>
  <div id="sidebar-overlay" class="sidebar-overlay"></div>
  <button id="menu-toggle" class="menu-toggle" aria-label="Toggle navigation">&#9776;</button>
  <div class="layout">
    <nav id="sidebar" class="sidebar">
      <div class="sidebar-header">
        <h2>Contents</h2>
        <button id="theme-toggle" class="theme-toggle" aria-label="Toggle theme">{initial_icon}</button>
      </div>
      {nav_links}
    </nav>
    <main class="content">
      <h1>{title}</h1>
      <p class="subtitle">{subtitle}</p>
      {sections_html}
    </main>
  </div>
{_DEVICE_JS}
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
        cards.append(f'<div class="card"><div class="label">{label}</div><div class="value">{val}</div></div>')
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
            f'<tr><td colspan="{len(headers)}" style="text-align:center;'
            f'color:var(--text-muted);font-style:italic;">'
            f"&hellip; {remaining} more rows</td></tr>"
        )
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def plotly_div(fig: object, height: int = 450) -> str:
    """Embed a Plotly figure as an inline div (no full HTML wrapper)."""
    import plotly.io as pio  # lazy

    return (
        f'<div class="chart-container" style="height:{height}px">'
        + pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True, "displayModeBar": True},
        )
        + "</div>"
    )
