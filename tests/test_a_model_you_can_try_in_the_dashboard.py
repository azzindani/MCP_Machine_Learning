"""The model dashboard carries the model: change a value, watch the answer move, with nothing running behind the page.

The training report has always had a "Try the model" form that runs the fitted model in the page, and the model
dashboard -- the page a model is read against labelled rows on -- had none, so the most persuasive thing a model
can do in front of someone, answer, was missing from it. A model too large to ship in a page says so and why
instead of leaving an empty gap. The last class opens the page in a real browser, when this machine has one, and
checks its answer against the model's own.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from servers.ml_advanced.engine import generate_model_dashboard
from servers.ml_basic._basic_predict import predict_single
from servers.ml_basic._basic_train import train_classifier, train_regressor
from shared import model_js


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n),
            "charges": rng.normal(70, 20, n).round(2),
            "plan": rng.choice(["basic", "pro"], n),
            "noise": rng.normal(0, 1, n).round(3),
        }
    )
    logit = -1 + 0.06 * (df["charges"] - 70) - 0.08 * df["tenure"] + (df["plan"] == "basic") * 1.5
    df["churned"] = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    df["price"] = (3 * df["tenure"] + 2 * df["charges"] + rng.normal(0, 8, n)).round(2)
    return df


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_DATA_ROOT", str(tmp_path))
    _frame(1200, 1).to_csv(tmp_path / "train.csv", index=False)
    _frame(500, 2).to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


def _train(home, kind, target, name):
    trainer = train_regressor if kind in ("lir", "rfr", "dtr") else train_classifier
    cols = ["tenure", "charges", "plan", "noise"]
    r = trainer(str(home / "train.csv"), target, kind, feature_columns=cols, output_path=str(home / f"{name}.pkl"))
    assert r["success"] is True, r
    return home / f"{name}.pkl"


def _dash(home, model, target, file="test.csv", **kw) -> dict:
    r = generate_model_dashboard(
        str(model), str(home / file), target, output_path=str(home / "d.html"), open_after=False, **kw
    )
    assert r["success"] is True, r.get("error")
    return r


class TestTheDashboardHasTheForm:
    @pytest.mark.parametrize(("kind", "target"), [("lr", "churned"), ("lir", "price"), ("dtc", "churned")])
    def test_a_model_that_can_travel_in_a_page_is_in_it(self, home, kind, target):
        r = _dash(home, _train(home, kind, target, kind), target)
        html = (home / "d.html").read_text(encoding="utf-8")
        assert r["interactive_prediction"] is True and "not_embeddable" not in r
        assert html.count("window.__MODEL__=") == 1 and 'id="mdl-form"' in html and 'id="mdl-out"' in html
        form = re.search(r'<form id="mdl-form".*?</form>', html, flags=re.S).group(0)
        assert len(re.findall(r"<(?:input|select)\b", form)) == 4, "one field for each of the model's four inputs"

    def test_it_sits_right_after_the_answer_and_the_warning_about_training_rows(self, home):
        model = _train(home, "lr", "churned", "lr")
        assert _dash(home, model, "churned")["sections_generated"][:2] == ["answer", "predict"]
        flattered = _dash(home, model, "churned", file="train.csv")["sections_generated"]
        assert flattered[:3] == ["answer", "overlap", "predict"]

    def test_the_rows_table_and_the_form_live_on_one_page(self, home):
        _dash(home, _train(home, "lr", "churned", "lr"), "churned")
        html = (home / "d.html").read_text(encoding="utf-8")
        assert 'id="pred-table"' in html and 'id="mdl-form"' in html


class TestAModelTooLargeSaysSo:
    def test_a_forest_over_the_cap_is_named_with_its_size_and_the_way_out(self, home, monkeypatch):
        model = _train(home, "rf", "churned", "rf")
        monkeypatch.setattr(model_js, "_MAX_TREE_NODES", 50)
        r = _dash(home, model, "churned")
        html = (home / "d.html").read_text(encoding="utf-8")
        assert r["interactive_prediction"] is False and re.search(r"forest has [\d,]+ nodes", r["not_embeddable"])
        assert "too large to run inside the page" in html and "max_depth" in html
        assert "window.__MODEL__=" not in html and "predict" in r["sections_generated"]

    def test_everything_else_is_still_on_the_page(self, home, monkeypatch):
        model = _train(home, "rf", "churned", "rf")
        monkeypatch.setattr(model_js, "_MAX_TREE_NODES", 50)
        sections = _dash(home, model, "churned")["sections_generated"]
        assert {"answer", "performance", "predictions", "variables", "importance"} <= set(sections)

    def test_a_model_with_no_exact_page_form_says_so_too(self, home):
        r = _dash(home, _train(home, "knn", "churned", "knn"), "churned")
        assert r["interactive_prediction"] is False and r["not_embeddable"]
        assert "too large to run inside the page" in (home / "d.html").read_text(encoding="utf-8")


def _system_python() -> str | None:
    for candidate in ("/usr/bin/python3", "/usr/local/bin/python3", shutil.which("python3")):
        if not candidate or not Path(candidate).exists() or Path(candidate).resolve() == Path(sys.executable).resolve():
            continue
        done = subprocess.run(
            [candidate, "-c", "from playwright.sync_api import sync_playwright as s\nwith s() as p: p.chromium.launch().close()"],
            capture_output=True, timeout=120,
        )  # fmt: skip
        if done.returncode == 0:
            return candidate
    return None


SYSTEM_PYTHON = _system_python()

_ASK = r"""
import json, sys
from playwright.sync_api import sync_playwright
url, changes = sys.argv[1], json.loads(sys.argv[2])
out = {"errors": []}
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.on("pageerror", lambda e: out["errors"].append(str(e)[:120]))
    pg.goto(url, wait_until="load"); pg.wait_for_timeout(1500)
    feats = pg.evaluate("window.__MODEL__.features.map(f=>f.name||f)")
    fields = pg.evaluate("[...document.querySelectorAll('#mdl-form input,#mdl-form select')].map(e=>e.value)")
    out["features"], out["defaults"], out["at_defaults"] = feats, fields, pg.inner_text("#mdl-out")
    answers = []
    for name, value in changes:
        el = pg.locator("#mdl-form input, #mdl-form select").nth(feats.index(name))
        if el.evaluate("e=>e.tagName") == "SELECT": el.select_option(value)
        else: el.fill(str(value)); el.dispatch_event("input")
        pg.wait_for_timeout(250); answers.append(pg.inner_text("#mdl-out"))
    out["after_changes"] = answers
    b.close()
print(json.dumps(out))
"""


@pytest.mark.skipif(SYSTEM_PYTHON is None, reason="no browser with playwright on this machine")
class TestThePageAnswersAsTheModelDoes:
    @pytest.mark.parametrize(("kind", "target"), [("lr", "churned"), ("lir", "price")])
    def test_changing_a_value_gives_the_models_own_answer(self, home, kind, target):
        model = _train(home, kind, target, kind)
        _dash(home, model, target)
        changes = [["tenure", 3], ["charges", 120.5], ["plan", "pro"]]
        done = subprocess.run(
            [SYSTEM_PYTHON, "-c", _ASK, (home / "d.html").as_uri(), json.dumps(changes)],
            capture_output=True, encoding="utf-8", timeout=180,
        )  # fmt: skip
        assert done.returncode == 0, done.stderr[-1500:]
        page = json.loads(done.stdout)
        assert page["errors"] == []
        record = dict(zip(page["features"], page["defaults"], strict=True))
        record = {k: (float(v) if re.fullmatch(r"-?[\d.]+", v) else v) for k, v in record.items()}
        asked = []
        for name, value in changes:
            record[name] = value
            asked.append(predict_single(str(model), json.dumps(record))["prediction"])
        assert len(set(page["after_changes"])) > 1, "the answer did not move when the values did"
        for said, true in zip(page["after_changes"], asked, strict=True):
            if kind == "lir":
                assert float(said.replace(",", "")) == pytest.approx(true, rel=1e-3), (said, true)
            else:
                assert said == str(true), (said, true)
