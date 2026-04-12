"""shared/html_theme.py — Responsive Plotly theme + HTML report engine.

Layout philosophy
-----------------
- rem / em everywhere — raw px only where browser APIs require integers
- CSS custom properties for every spatial dimension and color token
- clamp() for fluid type scale; viewport-unit chart heights cap at 80 vh
- Plotly autosize + responsive:true fills the CSS container
- Mobile-first sidebar: JS-toggle hamburger below 48 em breakpoint
- Chart paper_bgcolor matches card surface — no background mismatch
- calc_chart_height() replaces magic-number heights throughout
- Downloads folder as default output; any absolute path accepted
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import plotly.io as pio  # lazy-safe: imported once at module level

from shared.file_utils import atomic_write_text
from shared.html_layout import VIEWPORT_META, get_output_path  # noqa: F401

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
    return PLOTLY_TEMPLATE.get(theme, "plotly_white")


# ---------------------------------------------------------------------------
# Viewport meta + Plotly CDN
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Inline Plotly JS — cached at module level so it's only read from disk once
# ---------------------------------------------------------------------------

_PLOTLYJS_SCRIPT: str | None = None


def get_plotlyjs_script() -> str:
    """Return a <script> tag containing the full Plotly.js bundle (inline).

    Cached at module level — disk read happens only on the first call.
    Prefer this over a CDN tag so reports work completely offline.
    """
    global _PLOTLYJS_SCRIPT
    if _PLOTLYJS_SCRIPT is None:
        from plotly.offline import get_plotlyjs

        _PLOTLYJS_SCRIPT = f"<script>{get_plotlyjs()}</script>"
    return _PLOTLYJS_SCRIPT


# ---------------------------------------------------------------------------
# Theme color tokens
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


def get_theme(theme: str = "dark") -> dict:
    """Return theme config dict. Falls back to dark for unknown themes."""
    return THEMES.get(theme, THEMES["dark"])


def theme_plot_colors(theme: str) -> tuple[str, str, str]:
    """Return (plot_bg, font_color, accent_color) for inline Plotly scripts."""
    if theme == "dark":
        return "#161b22", "#c9d1d9", "#58a6ff"
    return "#f6f8fa", "#1f2328", "#0969da"


# ---------------------------------------------------------------------------
# CSS custom property blocks (colors only — dimensions live in LAYOUT_CSS)
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
# Spatial + typographic tokens — rem-based, no px
_LAYOUT_VARS = (
    "--sidebar-w:16.25rem;"  # 260 px equivalent
    "--sidebar-w-md:13.75rem;"  # 220 px at 68.75 em breakpoint
    "--main-pad:2rem;"
    "--main-pad-sm:1rem;"
    "--section-gap:3rem;"
    "--card-gap:0.75rem;"
    "--card-min:8rem;"
    "--card-pad:1rem;"
    "--radius-sm:0.375rem;"
    "--radius-md:0.625rem;"
    "--radius-lg:0.75rem;"
    "--font-xs:0.6875rem;"  # ~11 px
    "--font-sm:0.8125rem;"  # ~13 px
    "--font-base:1rem;"
    "--font-lg:1.125rem;"
    "--font-xl:1.25rem;"
    "--font-2xl:clamp(1.125rem,2vw,1.5rem);"
    "--chart-radius:0.75rem;"
)


def css_vars(theme: str) -> str:
    """Return a CSS :root{} block (and optional media query) for the theme."""
    layout = _LAYOUT_VARS
    if theme == "light":
        return f":root{{{_LIGHT_VARS}{layout}}}"
    elif theme == "device":
        return f":root{{{_LIGHT_VARS}{layout}}}@media(prefers-color-scheme:dark){{:root{{{_DARK_VARS}}}}}"
    else:  # dark
        return f":root{{{_DARK_VARS}{layout}}}"


# ---------------------------------------------------------------------------
# Device-mode JS (auto-switches Plotly template + body bg on system pref)
# ---------------------------------------------------------------------------

_DEVICE_JS = """<script>
(function(){
  const DARK_BG='#0d1117',LIGHT_BG='#ffffff';
  function applyTheme(){
    const dark=window.matchMedia('(prefers-color-scheme:dark)').matches;
    document.body.style.background=dark?DARK_BG:LIGHT_BG;
    document.documentElement.setAttribute('data-theme',dark?'dark':'light');
    document.querySelectorAll('.plotly-graph-div').forEach(function(d){
      try{Plotly.relayout(d,{
        template:dark?'plotly_dark':'plotly_white',
        paper_bgcolor:dark?'#161b22':'#f6f8fa',
        plot_bgcolor:dark?'#161b22':'#f6f8fa'
      });}catch(e){}
    });
  }
  if(typeof Plotly!=='undefined'){applyTheme();}
  else{window.addEventListener('load',applyTheme);}
  window.matchMedia('(prefers-color-scheme:dark)').addEventListener('change',applyTheme);
})();
</script>"""


def device_mode_js() -> str:
    """Return device-mode theme-switching JS snippet."""
    return _DEVICE_JS


# ---------------------------------------------------------------------------
# Sidebar toggle JS (mobile hamburger)
# ---------------------------------------------------------------------------

_SIDEBAR_JS = """<script>
(function(){
  const btn=document.getElementById('sb-toggle');
  const sb=document.querySelector('.sidebar');
  const overlay=document.getElementById('sb-overlay');
  function close(){sb.classList.remove('open');overlay.classList.remove('show');}
  if(btn&&sb&&overlay){
    btn.addEventListener('click',function(){
      const open=sb.classList.toggle('open');
      overlay.classList.toggle('show',open);
    });
    overlay.addEventListener('click',close);
  }
})();
</script>"""

# ---------------------------------------------------------------------------
# Scroll spy — sidebar active link follows viewport
# ---------------------------------------------------------------------------

_SCROLL_SPY_JS = """<script>
(function(){
  'use strict';
  const links=document.querySelectorAll('.nav a[href^="#"]');
  const sections=Array.from(document.querySelectorAll('.section[id]'));
  if(!sections.length||!links.length)return;
  const obs=new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if(e.isIntersecting){
        links.forEach(function(l){l.classList.remove('active');});
        const m=document.querySelector('.nav a[href="#'+e.target.id+'"]');
        if(m)m.classList.add('active');
      }
    });
  },{rootMargin:'-20% 0px -70% 0px'});
  sections.forEach(function(s){obs.observe(s);});
})();
</script>"""

# ---------------------------------------------------------------------------
# Sortable tables — click <th data-sort> to sort ascending/descending
# ---------------------------------------------------------------------------

_SORTABLE_TABLES_JS = """<script>
(function(){
  'use strict';
  document.querySelectorAll('th[data-sort]').forEach(function(th){
    th.style.cursor='pointer';
    let dir=1;
    th.addEventListener('click',function(){
      const idx=th.cellIndex;
      const tbody=th.closest('table').querySelector('tbody');
      if(!tbody)return;
      const rows=Array.from(tbody.rows);
      rows.sort(function(a,b){
        const av=a.cells[idx]?a.cells[idx].textContent.trim():'';
        const bv=b.cells[idx]?b.cells[idx].textContent.trim():'';
        const n=av-bv;
        return dir*(isNaN(n)?av.localeCompare(bv):n);
      });
      rows.forEach(function(r){tbody.appendChild(r);});
      th.closest('table').querySelectorAll('th').forEach(function(t){
        t.textContent=t.textContent.replace(/ [▲▼]$/,'');
      });
      th.textContent+=dir>0?' ▲':' ▼';
      dir*=-1;
    });
  });
})();
</script>"""

# ---------------------------------------------------------------------------
# Collapsible sections — click h2 to toggle, state in sessionStorage
# ---------------------------------------------------------------------------

_COLLAPSIBLE_SECTIONS_JS = """<script>
(function(){
  'use strict';
  document.querySelectorAll('.section>h2').forEach(function(h){
    const section=h.closest('.section');
    if(!section)return;
    const id=section.id;
    const body=h.nextElementSibling;
    if(!body)return;
    h.style.cursor='pointer';
    const key='sec-collapsed-'+id;
    if(sessionStorage.getItem(key)==='1'){
      body.style.display='none';
      h.textContent='▶ '+h.textContent;
    }else{
      h.textContent='▼ '+h.textContent;
    }
    h.addEventListener('click',function(){
      const hidden=body.style.display==='none';
      body.style.display=hidden?'':'none';
      sessionStorage.setItem(key,hidden?'0':'1');
      h.textContent=(hidden?'▼ ':'▶ ')+h.textContent.slice(2);
    });
  });
})();
</script>"""

# ---------------------------------------------------------------------------
# Animated KPI counters — requestAnimationFrame count-up on load
# ---------------------------------------------------------------------------

_KPI_COUNTER_JS = """<script>
(function(){
  'use strict';
  if(window.matchMedia('(prefers-reduced-motion:reduce)').matches)return;
  document.querySelectorAll('.card .num[data-val]').forEach(function(el){
    const target=parseFloat(el.dataset.val);
    if(isNaN(target))return;
    const fmt=el.dataset.fmt||'int';
    const start=performance.now();
    const dur=600;
    function step(now){
      const t=Math.min((now-start)/dur,1);
      const ease=1-Math.pow(1-t,3);
      const v=target*ease;
      el.textContent=fmt==='float2'?v.toFixed(2)
                    :fmt==='pct'?v.toFixed(1)+'%'
                    :Math.round(v).toLocaleString();
      if(t<1)requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });
})();
</script>"""

# ---------------------------------------------------------------------------
# Copy to clipboard — data-copy attribute triggers copy + feedback
# ---------------------------------------------------------------------------

_COPY_CLIPBOARD_JS = """<script>
(function(){
  'use strict';
  document.querySelectorAll('[data-copy]').forEach(function(btn){
    btn.addEventListener('click',function(){
      const text=btn.dataset.copy;
      navigator.clipboard.writeText(text).then(function(){
        const orig=btn.textContent;
        btn.textContent='Copied!';
        setTimeout(function(){btn.textContent=orig;},1500);
      }).catch(function(){});
    });
  });
})();
</script>"""

_PRINT_BTN_JS = """<script>
(function(){
  'use strict';
  const btn=document.querySelector('.btn-print');
  if(btn){btn.addEventListener('click',function(){window.print();});}
})();
</script>"""

# ---------------------------------------------------------------------------
# Back-to-top — button appears after 300 px scroll
# ---------------------------------------------------------------------------

_BACK_TO_TOP_HTML = (
    '<button id="back-top" aria-label="Back to top" style="display:none;position:fixed;'
    "bottom:1.5rem;right:1.5rem;z-index:300;background:var(--surface);"
    "border:1px solid var(--border);border-radius:var(--radius-md);"
    "padding:.5rem .75rem;cursor:pointer;color:var(--accent);font-size:1rem;"
    'box-shadow:0 2px 8px rgba(0,0,0,.2)" title="Back to top">▲</button>'
)

_BACK_TO_TOP_JS = """<script>
(function(){
  'use strict';
  const btn=document.getElementById('back-top');
  if(!btn)return;
  window.addEventListener('scroll',function(){
    btn.style.display=window.scrollY>300?'':'none';
  },{passive:true});
  btn.addEventListener('click',function(){
    window.scrollTo({top:0,behavior:'smooth'});
  });
})();
</script>"""

# ---------------------------------------------------------------------------
# Chart height calculator — no magic numbers in calling code
# ---------------------------------------------------------------------------

# Per-item heights used as named constants
_PX_PER_ROW_SUBPLOT = 220  # height per subplot row in stacked charts
_PX_PER_ROW_BAR = 28  # height per bar/feature in horizontal bar charts
_PX_HEATMAP_PER_ITEM = 28  # height per matrix row in heatmaps
_PX_CHART_BASE = 80  # base overhead (title, margins, axis labels)
_PX_CHART_MIN = 280  # minimum chart height
_PX_CHART_MAX = 1800  # maximum chart height (prevents infinite scroll)


def calc_chart_height(
    n: int = 1,
    mode: str = "subplot",
    extra_base: int = 0,
) -> int:
    """Return chart height in px for Plotly update_layout(height=...).

    mode  : "subplot"  — stacked subplot rows (distributions, CV)
            "bar"      — horizontal bar chart (feature importance, etc.)
            "heatmap"  — correlation / confusion matrix
            "fixed"    — return n directly (caller provides exact px)
    extra_base : additional px to add to the base overhead
    """
    base = _PX_CHART_BASE + extra_base
    if mode == "subplot":
        raw = _PX_PER_ROW_SUBPLOT * n + base
    elif mode == "bar":
        raw = _PX_PER_ROW_BAR * n + base
    elif mode == "heatmap":
        raw = _PX_HEATMAP_PER_ITEM * n + base
    else:  # fixed
        raw = n
    return max(_PX_CHART_MIN, min(_PX_CHART_MAX, raw))


# ---------------------------------------------------------------------------
# Apply consistent Plotly theme colors to any figure
# ---------------------------------------------------------------------------


def apply_fig_theme(fig: object, theme: str) -> None:
    """Set paper_bgcolor, plot_bgcolor, font color to match CSS surface token.

    Call this before plotly_div() or save_chart() to eliminate background
    mismatch between chart area and surrounding card.
    """
    t = get_theme(theme)
    fig.update_layout(  # type: ignore[attr-defined]
        paper_bgcolor=t["paper_color"],
        plot_bgcolor=t["paper_color"],
        font=dict(color=t["text_color"]),
        template=plotly_template(theme),
        autosize=True,
    )


# ---------------------------------------------------------------------------
# Browser auto-open (cross-platform)
# ---------------------------------------------------------------------------


def _open_file(path: str | Path) -> None:
    """Open file in default browser/app. Best-effort, never raises."""
    p = Path(path).resolve()
    try:
        import webbrowser

        webbrowser.open(f"file://{p}")
    except Exception:
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
# Standalone chart page CSS (save_chart output)
# ---------------------------------------------------------------------------

_STANDALONE_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100vh}
html{scroll-behavior:smooth}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);
  min-height:100vh;padding:clamp(1rem,4vw,2.5rem);display:flex;
  align-items:flex-start;justify-content:center}
.page{width:100%;max-width:75rem}
.chart-title{color:var(--accent);font-size:var(--font-2xl);font-weight:600;
  margin-bottom:1.5rem;overflow-wrap:break-word;word-break:break-word;
  padding-bottom:0.75rem;border-bottom:2px solid var(--border)}
.chart-wrap{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--chart-radius);padding:var(--card-pad);overflow:hidden;
  min-height:clamp(18rem,55vh,50rem)}
.chart-wrap .js-plotly-plot,.chart-wrap .plotly-graph-div{width:100%!important;
  min-height:inherit}
::-webkit-scrollbar{width:0.375rem}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:var(--radius-sm)}
@media(max-width:30em){body{padding:0.75rem}.chart-title{font-size:var(--font-xl)}}
@media print{body{background:#fff;color:#000;padding:0}.chart-wrap{border:none;padding:0}}
"""


# ---------------------------------------------------------------------------
# save_chart — standalone Plotly figure → responsive HTML file
# ---------------------------------------------------------------------------


def save_chart(
    fig: object,
    output_path: str,
    stem_suffix: str,
    input_path: Path,
    theme: str = "dark",
    open_after: bool = True,
    open_func=None,
) -> tuple[str, str]:
    """Save a Plotly figure as a full, responsive standalone HTML page.

    output_path : explicit path (str) or "" to auto-derive from input_path
    stem_suffix : suffix for the auto-derived filename, e.g. "roc_curve"
    input_path  : source data file — used to derive the default output name
    open_func   : callable(Path) to open the file; defaults to _open_file

    Returns (absolute_path_str, filename).
    """
    apply_fig_theme(fig, theme)

    out = get_output_path(output_path, input_path, stem_suffix, "html")
    out.parent.mkdir(parents=True, exist_ok=True)

    chart_html = pio.to_html(
        fig,  # type: ignore[arg-type]
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displayModeBar": True,
            "scrollZoom": True,
            "plotGlPixelRatio": 0,
        },
    )

    dev_js = device_mode_js() if theme == "device" else ""
    vars_css = css_vars(theme)

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
{VIEWPORT_META}
<title>{stem_suffix.replace("_", " ").title()}</title>
<style>
{vars_css}
{_STANDALONE_CSS}
</style></head><body>
<div class="page">
  <div class="chart-wrap">{chart_html}</div>
</div>
{dev_js}
</body></html>"""

    atomic_write_text(out, html)
    if open_after:
        opener = open_func if open_func is not None else _open_file
        opener(out)

    return str(out), out.name


# ---------------------------------------------------------------------------
# Report CSS — full multi-section HTML report (build_html_report)
# ---------------------------------------------------------------------------


def report_css(vars_block: str) -> str:
    """Return full CSS block for multi-section HTML reports.

    All spatial properties use CSS custom properties (rem-based).
    clamp() used for fluid card numbers and headings.
    Mobile sidebar: JS toggle, overlay backdrop.
    """
    return f"""{vars_block}
*{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);
  min-height:100vh;transition:background 0.2s,color 0.2s;line-height:1.6;
  overflow-wrap:break-word;word-break:break-word;font-size:var(--font-base)}}
::-webkit-scrollbar{{width:0.375rem}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:var(--radius-sm)}}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
.sidebar{{width:var(--sidebar-w);background:var(--surface);
  border-right:1px solid var(--border);position:fixed;top:0;left:0;bottom:0;
  overflow-y:auto;z-index:100;transition:transform 0.22s ease}}
.sidebar-hdr{{padding:1.25rem var(--main-pad-sm);border-bottom:1px solid var(--border);
  overflow:hidden}}
.sidebar-hdr h2{{color:var(--accent);font-size:var(--font-lg);margin-bottom:0.25rem;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.sidebar-hdr .meta{{color:var(--text-muted);font-size:var(--font-xs);line-height:1.5;
  overflow-wrap:break-word;word-break:break-word}}
.nav{{padding:0.5rem 0}}
.nav a{{display:block;padding:0.4375rem var(--main-pad-sm);color:var(--text-muted);
  text-decoration:none;font-size:var(--font-sm);border-left:3px solid transparent;
  transition:all 0.15s;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.nav a:hover,.nav a.active{{color:var(--accent);background:rgba(88,166,255,0.06);
  border-left-color:var(--accent)}}
.nav .st{{padding:0.875rem var(--main-pad-sm) 0.25rem;color:var(--border);
  font-size:0.625rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:600}}

/* ── Hamburger toggle (mobile) ────────────────────────────────────────── */
#sb-toggle{{display:none;position:fixed;top:0.75rem;left:0.75rem;z-index:200;
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:0.375rem 0.625rem;cursor:pointer;
  color:var(--accent);font-size:var(--font-xl);line-height:1;
  box-shadow:0 2px 8px rgba(0,0,0,.15)}}
#sb-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);
  z-index:90;transition:opacity 0.2s}}
#sb-overlay.show{{display:block}}

/* ── Main content ─────────────────────────────────────────────────────── */
.main{{margin-left:var(--sidebar-w);padding:var(--main-pad);min-height:100vh;
  overflow-x:hidden;max-width:100%}}
.section{{margin-bottom:var(--section-gap)}}
.section>h2{{color:var(--accent);font-size:var(--font-2xl);margin-bottom:1.25rem;
  padding-bottom:0.625rem;border-bottom:2px solid var(--border);font-weight:600;
  overflow-wrap:break-word;word-break:break-word;
  position:sticky;top:0;z-index:10;background:var(--bg);padding-top:0.5rem}}
.btn-print{{background:none;border:1px solid var(--border);border-radius:var(--radius-sm);
  padding:0.25rem 0.625rem;cursor:pointer;color:var(--text-muted);font-size:var(--font-xs);
  margin-top:0.375rem;transition:color 0.15s,border-color 0.15s}}
.btn-print:hover{{color:var(--accent);border-color:var(--accent)}}

/* ── Metric cards ─────────────────────────────────────────────────────── */
.cards{{display:grid;
  grid-template-columns:repeat(auto-fit,minmax(var(--card-min),1fr));
  gap:var(--card-gap);margin-bottom:1.5rem}}
.card{{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius-md);padding:var(--card-pad);text-align:center;
  transition:transform 0.15s,border-color 0.15s;overflow:hidden;min-width:0}}
.card:hover{{transform:translateY(-2px);border-color:var(--accent)}}
.card .num{{font-size:clamp(1.25rem,2.5vw,1.75rem);font-weight:700;
  color:var(--accent);line-height:1.25;overflow-wrap:break-word;
  word-break:break-word;hyphens:auto}}
.card .lbl{{font-size:var(--font-xs);color:var(--text-muted);margin-top:0.25rem;
  text-transform:uppercase;letter-spacing:0.05em;overflow-wrap:break-word;
  word-break:break-word}}
.card.good .num{{color:var(--green)}}
.card.warn .num{{color:var(--orange)}}
.card.bad  .num{{color:var(--red)}}

/* ── Tables ───────────────────────────────────────────────────────────── */
.table-wrap{{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:0.75rem 0}}
table{{width:100%;border-collapse:collapse;font-size:var(--font-sm);
  background:var(--surface);border-radius:var(--radius-md);overflow:hidden;
  table-layout:auto;min-width:30rem}}
th,td{{padding:0.625rem 0.875rem;text-align:left;border-bottom:1px solid var(--border);
  overflow-wrap:break-word;word-break:break-word;max-width:20rem;min-width:0}}
th{{background:rgba(88,166,255,0.08);color:var(--accent);font-weight:600;
  font-size:var(--font-xs);text-transform:uppercase;letter-spacing:0.03em;
  white-space:nowrap}}
td{{line-height:1.45}}
tr:hover{{background:rgba(88,166,255,0.03)}}

/* ── Status helpers ───────────────────────────────────────────────────── */
.good{{color:var(--green)}}.warn{{color:var(--orange)}}.bad{{color:var(--red)}}
.badge{{font-size:var(--font-xs);padding:0.125rem 0.5rem;border-radius:0.625rem;
  background:var(--border);color:var(--text-muted);font-weight:500}}
.stats-cell{{font-size:var(--font-xs);color:var(--text-muted);font-family:monospace}}

/* ── Insights list ────────────────────────────────────────────────────── */
.insights{{list-style:none;padding:0}}
.insights li{{padding:0.625rem 0.875rem;margin:0.375rem 0;background:var(--surface);
  border-radius:var(--radius-md);border-left:4px solid var(--accent);
  font-size:var(--font-sm);line-height:1.55;overflow-wrap:break-word;
  word-break:break-word}}
.insights li.warn{{border-left-color:var(--orange)}}
.insights li.bad {{border-left-color:var(--red)}}
.insights li.good{{border-left-color:var(--green)}}

/* ── Progress bars ────────────────────────────────────────────────────── */
.mbar{{height:1.5rem;background:var(--border);border-radius:var(--radius-sm);
  overflow:hidden;margin:0.25rem 0}}
.mbar-fill{{height:100%;background:linear-gradient(90deg,var(--orange),var(--red));
  border-radius:var(--radius-sm)}}

/* ── Chart containers ─────────────────────────────────────────────────── */
.chart-container{{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--chart-radius);padding:var(--card-pad);margin:1rem 0;
  overflow:hidden;width:100%}}
.chart-container .js-plotly-plot,
.chart-container .plotly-graph-div{{width:100%!important;max-width:100%}}

/* ── Code blocks ──────────────────────────────────────────────────────── */
pre{{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius-md);padding:0.75rem;overflow-x:auto;
  font-size:var(--font-sm);line-height:1.5;color:var(--text);
  white-space:pre-wrap;word-break:break-all}}

/* ── Alert panels ─────────────────────────────────────────────────────── */
.alert-panel{{border-radius:var(--radius-md);overflow:hidden;margin-bottom:1.25rem}}
.alert-item{{padding:0.625rem 0.875rem;margin:0.1875rem 0;font-size:var(--font-sm);
  border-radius:var(--radius-md);display:flex;align-items:flex-start;gap:0.625rem;
  background:var(--surface);border:1px solid var(--border);line-height:1.55;
  overflow-wrap:break-word;word-break:break-word}}
.alert-item.error  {{border-left:4px solid var(--red)}}
.alert-item.warning{{border-left:4px solid var(--orange)}}
.alert-item.info   {{border-left:4px solid var(--green)}}
.alert-badge{{font-size:0.625rem;font-weight:700;padding:0.125rem 0.5rem;
  border-radius:0.625rem;white-space:nowrap;flex-shrink:0;margin-top:0.125rem}}
.alert-badge.error  {{background:var(--red);color:#fff}}
.alert-badge.warning{{background:var(--orange);color:#fff}}
.alert-badge.info   {{background:var(--green);color:#fff}}
.alert-item span:last-child{{min-width:0}}

/* ── Responsive — tablet ──────────────────────────────────────────────── */
@media(max-width:68.75em){{
  .sidebar{{width:var(--sidebar-w-md)}}
  .main{{margin-left:var(--sidebar-w-md)}}
}}

/* ── Responsive — mobile ──────────────────────────────────────────────── */
@media(max-width:48em){{
  #sb-toggle{{display:flex;align-items:center;justify-content:center}}
  .sidebar{{transform:translateX(-100%)}}
  .sidebar.open{{transform:translateX(0)}}
  .main{{margin-left:0;padding:var(--main-pad-sm);padding-top:3.5rem}}
  .cards{{grid-template-columns:repeat(2,1fr)}}
  table{{min-width:20rem}}
  th,td{{padding:0.5rem 0.625rem;font-size:var(--font-xs);max-width:12.5rem}}
}}

/* ── Responsive — small mobile ────────────────────────────────────────── */
@media(max-width:30em){{
  .cards{{grid-template-columns:1fr}}
  .card .num{{font-size:1.375rem}}
  th,td{{padding:0.375rem 0.5rem;max-width:9.375rem}}
}}

/* ── Print ────────────────────────────────────────────────────────────── */
@media print{{
  .sidebar,#sb-toggle,#sb-overlay{{display:none!important}}
  .main{{margin-left:0!important;padding:0!important}}
  .chart-container{{break-inside:avoid;border:1px solid #ccc}}
  .section{{break-inside:avoid}}
}}"""


# ---------------------------------------------------------------------------
# Multi-section HTML report builder
# ---------------------------------------------------------------------------


def build_html_report(
    title: str,
    subtitle: str,
    sections: list[dict],
    theme: str = "dark",
    open_after: bool = True,
    output_path: str | Path = "",
    sidebar_title: str = "",
    sidebar_meta: str = "",
) -> str:
    """Build a full multi-section HTML report with responsive sidebar.

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
    sb_meta_html = f'<p class="meta">{sidebar_meta}</p>' if sidebar_meta else ""

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
{VIEWPORT_META}
<title>{title}</title>
{get_plotlyjs_script()}
<style>
{css_block}
</style></head><body>

<button id="sb-toggle" aria-label="Open navigation">&#9776;</button>
<div id="sb-overlay"></div>
{_BACK_TO_TOP_HTML}

<div class="sidebar">
  <div class="sidebar-hdr">
    <h2 title="{sb_title}">{sb_title}</h2>
    {sb_meta_html}
    <button class="btn-print">&#x2399; Print</button>
  </div>
  <div class="nav">
    <div class="st">Sections</div>
    {nav_links}
  </div>
</div>

<div class="main">
  {sections_html}
</div>

{_SIDEBAR_JS}
{_SCROLL_SPY_JS}
{_SORTABLE_TABLES_JS}
{_COLLAPSIBLE_SECTIONS_JS}
{_KPI_COUNTER_JS}
{_COPY_CLIPBOARD_JS}
{_PRINT_BTN_JS}
{_BACK_TO_TOP_JS}
{dev_js}
</body></html>"""

    if output_path:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out, html)
        if open_after:
            _open_file(out)

    return html


# ---------------------------------------------------------------------------
# Small helpers for report sections
# ---------------------------------------------------------------------------


def metrics_cards_html(metrics: dict, styles: dict[str, str] | None = None) -> str:
    """Render a dict of metrics as card HTML.

    Optional styles dict maps key → CSS class (good/warn/bad).
    """
    import html as _html

    if styles is None:
        styles = {}
    cards = []
    for k, v in metrics.items():
        cls = styles.get(k, "")
        cls_attr = f' class="card {cls}"' if cls else ' class="card"'
        label = _html.escape(k.replace("_", " ").title())
        if isinstance(v, float):
            val = f"{v:.4f}" if abs(v) < 10 else f"{v:,.2f}"
        else:
            val = _html.escape(str(v))
        cards.append(f'<div{cls_attr}><div class="num">{val}</div><div class="lbl">{label}</div></div>')
    return f'<div class="cards">{"".join(cards)}</div>'


def data_table_html(rows: list[dict], max_rows: int = 50) -> str:
    """Render a list of dicts as a scrollable HTML table with sortable headers."""
    import html as _html

    if not rows:
        return "<p>No data.</p>"
    headers = list(rows[0].keys())
    th = "".join(f"<th data-sort>{_html.escape(str(h).replace('_', ' '))}</th>" for h in headers)
    trs = ""
    for row in rows[:max_rows]:
        tds = ""
        for h in headers:
            val = row.get(h, "")
            tds += f"<td>{_html.escape(str(val) if val is not None else '')}</td>"
        trs += f"<tr>{tds}</tr>"
    if len(rows) > max_rows:
        remaining = len(rows) - max_rows
        trs += (
            f'<tr><td colspan="{len(headers)}" '
            f'style="text-align:center;color:var(--text-muted);font-style:italic">'
            f"&hellip; {remaining:,} more rows</td></tr>"
        )
    return f'<div class="table-wrap"><table><tr>{th}</tr><tbody>{trs}</tbody></table></div>'


def plotly_div(fig: object, height: int = 450, theme: str = "dark") -> str:
    """Embed a Plotly figure as an inline div, with theme-matched background.

    Applies paper_bgcolor / plot_bgcolor matching the CSS surface token so
    the chart background is consistent with surrounding cards.
    height : integer px passed to both Plotly layout and CSS min-height.
    theme  : used to set matching background colors.
    """
    apply_fig_theme(fig, theme)

    # Cap rendered height to 80 vh via inline CSS so tall charts scroll
    # instead of forcing the viewport to grow.
    inner = pio.to_html(
        fig,  # type: ignore[arg-type]
        full_html=False,
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displayModeBar": True,
            "scrollZoom": True,
            "plotGlPixelRatio": 0,
        },
    )
    return f'<div class="chart-container" style="height:min({height}px,80vh);overflow:hidden auto">' + inner + "</div>"
