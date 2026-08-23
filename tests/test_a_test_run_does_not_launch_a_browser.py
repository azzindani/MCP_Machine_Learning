"""A test run must never ask the desktop shell to open a chart.

_open_file() tries webbrowser first and falls back to os.startfile() on
Windows, in-process. In the sibling Office repo the equivalent call reached the
COM layer on the CI runner and killed the interpreter part-way through the
suite:

    Windows fatal exception: code 0x80010108        (RPC_E_DISCONNECTED)
    Windows fatal exception: access violation

with no failing test named and no traceback, because an access violation is not
an exception -- the `except` around it had never been able to catch it. The job
reported exit code 1 after ~30% of the tests and passed on ubuntu and macos,
where the same call is a subprocess that cannot touch the parent.

It now returns immediately when PYTEST_CURRENT_TEST is set, and spawns a child
on Windows rather than calling os.startfile() in-process.
"""

from __future__ import annotations

import os
from pathlib import Path

from shared import html_theme


class TestOpenFileIsInertUnderPytest:
    def test_it_launches_nothing_while_a_test_is_running(self, tmp_path, monkeypatch):
        opened: list = []
        monkeypatch.setattr(html_theme.subprocess, "Popen", lambda *a, **k: opened.append(a))
        import webbrowser

        monkeypatch.setattr(webbrowser, "open", lambda *a, **k: opened.append(a))
        # pytest sets this for the duration of every test; assert it rather than
        # trust it, since the guard is worthless if the name ever changes.
        assert os.environ.get("PYTEST_CURRENT_TEST")
        html_theme._open_file(tmp_path / "chart.html")
        assert opened == [], "a test run tried to launch the desktop handler"

    def test_outside_a_test_run_it_still_opens(self, tmp_path, monkeypatch):
        opened: list = []
        import webbrowser

        monkeypatch.setattr(webbrowser, "open", lambda *a, **k: opened.append(a))
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        html_theme._open_file(tmp_path / "chart.html")
        assert len(opened) == 1, "the guard must not disable the feature itself"

    def test_startfile_is_not_called_in_this_process(self):
        # os.startfile exists only on Windows, so assert on the source: the
        # in-process call is what can take the interpreter down, and no amount
        # of exception handling around it helps.
        body = Path(html_theme.__file__).read_text(encoding="utf-8").split("def _open_file(")[1]
        code = [ln for ln in body.splitlines() if not ln.lstrip().startswith("#")]
        assert "startfile" not in "\n".join(code), "still calls os.startfile in-process"
