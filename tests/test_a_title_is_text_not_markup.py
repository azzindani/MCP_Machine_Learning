"""A report heading built from a column or file name is text, never markup.

build_html_report put its title (which carries the file name) and each section
heading ("Target Column: <name>") into the page as they were, and the chart
page its figure's title: a column or file named `<img src=x onerror=...>`
became an element that ran when the report was opened (sweep F22).
"""

from __future__ import annotations

from shared.chart_page import chart_page_html
from shared.html_theme import build_html_report

EVIL = "<img src=x onerror=alert(1)>"
SAFE = "&lt;img src=x onerror=alert(1)&gt;"


def test_a_report_title_and_its_headings(tmp_path):
    html = build_html_report(
        title=f"EDA Report — {EVIL}.csv",
        subtitle="",
        sections=[{"id": "target", "heading": f"Target Column: {EVIL}", "html": "<p>body</p>"}],
        theme="light",
        open_after=False,
        output_path=str(tmp_path / "r.html"),
    )
    assert EVIL not in html
    assert f"<title>EDA Report — {SAFE}.csv</title>" in html
    assert f"<h2>Target Column: {SAFE}</h2>" in html
    assert "<p>body</p>" in html, "a section's own html is the builder's, and stays markup"


def test_a_chart_page_title():
    html = chart_page_html("<div>chart</div>", EVIL, "", "")
    assert EVIL not in html and f'<h1 class="chart-title">{SAFE}</h1>' in html
