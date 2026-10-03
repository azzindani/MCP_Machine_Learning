"""A dropdown, a scrollbar and a picker follow the page's theme, in light and in dark.

The "Try the model" form showed a dark page's dropdown as pale text on a white list: the report CSS never declared
`color-scheme`, so native controls stayed light while the text colour followed the theme. The same form used two
tokens (`--card`, `--muted`) that no theme defines, so its cards and labels took no theme colour at all.
"""

from __future__ import annotations

import re

import pytest

from shared.html_theme import css_vars
from shared.model_js import _PANEL_CSS


class TestNativeControlsFollowTheTheme:
    def test_a_dark_page_declares_a_dark_color_scheme(self):
        assert "color-scheme:dark" in css_vars("dark") and "color-scheme:light" not in css_vars("dark")

    def test_a_light_page_declares_a_light_one(self):
        assert "color-scheme:light" in css_vars("light") and "color-scheme:dark" not in css_vars("light")

    def test_a_device_page_declares_each_with_the_palette_it_goes_with(self):
        css = css_vars("device")
        light, dark = css.split("@media", 1)
        assert "color-scheme:light" in light and "color-scheme:dark" in dark


class TestThePanelUsesOnlyTokensTheThemeDefines:
    @pytest.mark.parametrize("theme", ["light", "dark", "device"])
    def test_every_token_it_reads_is_defined(self, theme):
        defined = set(re.findall(r"(--[a-z-]+):", css_vars(theme)))
        used = set(re.findall(r"var\((--[a-z-]+)", _PANEL_CSS))
        assert used and used <= defined, used - defined

    def test_a_dropdown_list_is_drawn_in_the_surface_and_text_colours(self):
        assert re.search(r"select option\{background:var\(--surface\);color:var\(--text\)\}", _PANEL_CSS)
