"""run_profiling_report was the last tool still hardcoded to dark.

Every other chart and report tool in this repo takes `theme: str = "device"` so
the artifact follows whatever the viewer has their system set to. This one had
no `theme` parameter at all and pinned three values:

    plot_bg, font_color, _accent = theme_plot_colors("dark")
    template = "plotly_dark"
    build_html_report(..., theme="dark", ...)

It showed up in a coverage sweep as one report rendering on a dark background
while every other report from the same run rendered light -- same machine, same
browser, same session.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from servers.ml_advanced.engine import run_profiling_report


@pytest.fixture()
def csv(tmp_path: Path) -> Path:
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0], "g": ["x", "y", "x", "y"]}).to_csv(
        p, index=False
    )
    return p


def _page(csv: Path, out: Path, **kwargs) -> str:
    result = run_profiling_report(str(csv), output_path=str(out), open_after=False, **kwargs)
    assert result["success"] is True, result.get("error")
    return out.read_text(encoding="utf-8")


class TestItFollowsTheViewer:
    def test_the_default_is_not_pinned_to_dark(self, csv: Path, tmp_path: Path):
        """The dark palette may appear, but only inside the media query -- the
        base :root must be the light one."""
        page = _page(csv, tmp_path / "p.html")
        assert page[page.index(":root{--bg:") :].startswith(":root{--bg:#ffffff")

    def test_the_default_ships_light_and_flips_on_the_system_setting(self, csv: Path, tmp_path: Path):
        page = _page(csv, tmp_path / "p.html")
        assert ":root{--bg:#ffffff" in page
        assert "prefers-color-scheme:dark" in page

    def test_the_figures_are_not_pinned_dark_either(self, csv: Path, tmp_path: Path):
        """The CSS following the viewer while the figures stay dark is the same
        bug one layer down. Asserted on the colour baked into the figures --
        'plotly_dark' as a string is expected, in the device script's ternary."""
        page = _page(csv, tmp_path / "p.html")
        assert '"paper_bgcolor":"#161b22"' not in page
        assert '"paper_bgcolor":"#f6f8fa"' in page


class TestExplicitThemesStillWork:
    def test_dark_is_still_available_on_request(self, csv: Path, tmp_path: Path):
        page = _page(csv, tmp_path / "p.html", theme="dark")
        assert ":root{--bg:#0d1117" in page

    def test_light_is_pinned_with_no_media_query(self, csv: Path, tmp_path: Path):
        page = _page(csv, tmp_path / "p.html", theme="light")
        assert ":root{--bg:#ffffff" in page
        assert "prefers-color-scheme:dark" not in page

    def test_an_unknown_theme_falls_back_to_following_the_viewer(self, csv: Path, tmp_path: Path):
        page = _page(csv, tmp_path / "p.html", theme="chartreuse")
        assert "prefers-color-scheme:dark" in page


class TestTheToolStillWorks:
    def test_the_report_has_content(self, csv: Path, tmp_path: Path):
        page = _page(csv, tmp_path / "p.html")
        assert "Profile Report" in page
        assert len(page) > 1000
