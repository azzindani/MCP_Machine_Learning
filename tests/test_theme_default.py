"""The default theme has to follow the viewer, not the machine that built it.

Training reports, cluster reports and profiling pages all defaulted to
`theme="dark"`, so every artifact came out dark regardless of who opened it.
These get sent to colleagues; the default is now "device" — the light palette
plus a `prefers-color-scheme: dark` override and a script that re-themes the
Plotly figures live when the viewer's setting changes.
"""

from __future__ import annotations

from pathlib import Path

from shared.html_theme import css_vars, device_mode_js, get_theme, plotly_template


class TestThemeResolution:
    def test_default_is_device(self):
        assert get_theme() == get_theme("device")

    def test_unknown_name_falls_back_to_the_default_not_dark(self):
        assert get_theme("Device") == get_theme("device")
        assert plotly_template("nonsense") == plotly_template("device")

    def test_device_ships_light_with_a_dark_override(self):
        css = css_vars("device")
        assert "prefers-color-scheme:dark" in css
        assert "#ffffff" in css
        assert "#0d1117" in css

    def test_explicit_choices_are_still_absolute(self):
        assert "prefers-color-scheme" not in css_vars("light")
        assert "prefers-color-scheme" not in css_vars("dark")

    def test_the_switch_script_retemplates_the_figures(self):
        """Without this the page would swap its chrome and leave every chart on
        the light template against a dark background."""
        js = device_mode_js()
        assert "Plotly.relayout" in js
        assert "plotly_dark" in js and "plotly_white" in js


class TestToolSignatures:
    def test_no_tool_still_defaults_to_dark(self):
        offenders = []
        for path in list(Path("servers").rglob("*.py")) + list(Path("shared").rglob("*.py")):
            if 'theme: str = "dark"' in path.read_text(encoding="utf-8"):
                offenders.append(str(path))
        assert offenders == [], f"still defaulting to dark: {offenders}"
