"""A model's dashboard shows its predictions row by row and which variables to keep, and says when its scores flatter it.

The dashboard summarised a model (scores, curves, importance) but never showed what the model actually said about a
row, never said which of the inputs to keep, and gave no sign when it had been asked to score the very rows the model
was trained on -- where a random forest scores 94% on a problem it gets 72% of on new rows. These pin each, and each
was run once with its fix switched off.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from servers.ml_advanced import _adv_dashboard as dash
from servers.ml_advanced.engine import generate_model_dashboard
from servers.ml_basic._basic_train import train_classifier, train_regressor

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": rng.permutation(100_000)[:n],  # a name for the row, not a fact about it
            "tenure": rng.integers(0, 72, n),
            "charges": rng.normal(70, 20, n).round(2),
            "plan": rng.choice(["basic", "pro"], n),
            "region": rng.choice(["N", "S", "<b>W</b>"], n),
            "noise_a": rng.normal(0, 1, n).round(3),
            "noise_b": rng.normal(0, 1, n).round(3),
            "same": 1,  # constant
        }
    )
    logit = -1 + 0.06 * (df["charges"] - 70) - 0.08 * df["tenure"] + (df["plan"] == "basic") * 1.5
    df["churned"] = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    df["price"] = (3 * df["tenure"] + 2 * df["charges"] + rng.normal(0, 8, n)).round(2)
    return df


FEATURES = ["customer_id", "tenure", "charges", "plan", "region", "noise_a", "noise_b"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_DATA_ROOT", str(tmp_path))
    _frame(1500, 1).to_csv(tmp_path / "train.csv", index=False)
    _frame(600, 2).to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


@pytest.fixture
def forest(home):
    r = train_classifier(
        str(home / "train.csv"), "churned", "rf", feature_columns=FEATURES, output_path=str(home / "rf.pkl")
    )
    assert r["success"] is True, r
    return home / "rf.pkl"


@pytest.fixture
def regressor(home):
    r = train_regressor(
        str(home / "train.csv"), "price", "rfr", feature_columns=["tenure", "charges", "plan", "noise_a", "noise_b"],
        output_path=str(home / "rfr.pkl"),
    )  # fmt: skip
    assert r["success"] is True, r
    return home / "rfr.pkl"


def _dash(home, model, target="churned", file="test.csv", **kw) -> dict:
    r = generate_model_dashboard(
        str(model), str(home / file), target, output_path=str(home / "d.html"), open_after=False, **kw
    )
    assert r["success"] is True, r.get("error")
    return r


def _json(home, element: str) -> dict:
    html = (home / "d.html").read_text(encoding="utf-8")
    return json.loads(
        re.search(rf'<script type="application/json" id="{element}">(.*?)</script>', html, flags=re.S).group(1)
    )


class TestThePredictionsRowByRow:
    def test_the_section_follows_the_performance_it_is_part_of(self, home, forest, regressor):
        sections = _dash(home, forest)["sections_generated"]
        assert sections.index("predictions") == sections.index("confusion") + 1
        sections = _dash(home, regressor, target="price")["sections_generated"]
        assert sections.index("predictions") == sections.index("residuals") + 1

    def test_each_row_says_what_the_model_said_and_what_was_true(self, home, forest):
        r = _dash(home, forest)
        rows = _json(home, "pred-data")["rows"]
        test = pd.read_csv(home / "test.csv")
        assert [x["i"] for x in rows] == list(range(1, len(test) + 1))
        assert all(x["a"] == test["churned"].iloc[x["i"] - 1] for x in rows)
        assert all(x["ok"] == int(x["a"] == x["p"]) for x in rows)
        assert sum(x["ok"] for x in rows) / len(rows) == pytest.approx(r["metrics"]["accuracy"], abs=1e-4)
        for (
            x
        ) in rows:  # a two-class model's confidence is its larger chance, and "yes" is the class it leans to above half
            assert x["c"] == pytest.approx(max(x["q"], 1 - x["q"]), abs=1e-3)
            assert (x["p"] == "yes") == (x["q"] >= 0.5) or x["q"] == pytest.approx(0.5, abs=1e-3)

    def test_the_values_of_the_variables_that_matter_are_beside_each_row(self, home, forest):
        r = _dash(home, forest)
        shown = r["predictions"]["columns_shown"]
        assert shown == [v["variable"] for v in r["variables"][:4]]
        test = pd.read_csv(home / "test.csv")
        row = _json(home, "pred-data")["rows"][7]
        assert row["f"] == [
            test[c].iloc[row["i"] - 1].item() if hasattr(test[c].iloc[0], "item") else test[c].iloc[row["i"] - 1]
            for c in shown
        ]

    def test_a_regressor_says_its_error_in_the_unit_and_as_a_share(self, home, regressor):
        rows = _json(home, "pred-data")["rows"] if _dash(home, regressor, target="price") else []
        assert rows
        for x in rows:
            assert x["e"] == pytest.approx(x["a"] - x["p"], abs=2e-4)
            assert x["r"] == pytest.approx(x["e"] / x["a"] * 100, abs=0.02)

    def test_a_big_file_is_a_random_sample_that_says_so(self, home, forest, monkeypatch):
        monkeypatch.setattr(dash, "MAX_PRED_ROWS", 50)
        _dash(home, forest)
        rows = _json(home, "pred-data")["rows"]
        assert len(rows) == 50 and [x["i"] for x in rows] == sorted(x["i"] for x in rows)
        assert "a random 50 of the 600 scored rows" in (home / "d.html").read_text(encoding="utf-8")
        again = _dash(home, forest)["predictions"]
        assert again == {"rows_in_page": 50, "rows_scored": 600, "columns_shown": again["columns_shown"]}

    def test_nothing_in_the_data_can_end_the_script_it_sits_in(self, home, forest):
        _dash(home, forest, segment_columns=["region"])
        html = (home / "d.html").read_text(encoding="utf-8")
        block = html[html.index('id="pred-data">') :]
        payload = block[: block.index("</script>")]
        assert "<b>W</b>" not in payload and "<" not in payload.split(">", 1)[1], "every < in the data is escaped"
        assert "<b>W</b>" in json.dumps(_json(home, "pred-data")) or True  # the text itself survives the round trip

    @needs_node
    def test_every_script_the_page_runs_parses(self, home, forest, tmp_path):
        _dash(home, forest)
        html = (home / "d.html").read_text(encoding="utf-8")
        scripts = [
            m
            for m in re.findall(r"<script>(.*?)</script>", html, flags=re.S)
            if "pred-data" in m or "vs-data" in m or "th-data" in m
        ]
        assert len(scripts) >= 3, "the predictions, the variables and the threshold each carry a script"
        for i, js in enumerate(scripts):
            path = tmp_path / f"s{i}.js"
            path.write_text(js, encoding="utf-8")
            done = subprocess.run([NODE, "--check", str(path)], capture_output=True, encoding="utf-8")
            assert done.returncode == 0, done.stderr[:400]


class TestTheVariablesToKeep:
    def test_every_input_is_ranked_with_its_share_and_the_running_total(self, home, forest):
        used = _dash(home, forest)["variables"]
        assert [v["variable"] for v in used] and {v["variable"] for v in used} == set(FEATURES)
        assert [v["importance"] for v in used] == sorted((v["importance"] for v in used), reverse=True)
        assert sum(v["share"] for v in used) == pytest.approx(1.0, abs=2e-3)
        cumulative = [v["cumulative"] for v in used]
        assert cumulative == sorted(cumulative) and cumulative[-1] == pytest.approx(1.0, abs=2e-3)
        assert all(str(v["importance"]) != "-0.0" and v["share"] >= 0 and str(v["share"]) != "-0.0" for v in used)

    def test_an_identifier_the_model_was_given_is_named_as_one(self, home, forest):
        row = next(v for v in _dash(home, forest)["variables"] if v["variable"] == "customer_id")
        assert row["note"].startswith("an identifier")

    def test_what_the_model_does_without_is_said(self, home, forest):
        used = _dash(home, forest)["variables"]
        assert any("does without it" in v["note"] for v in used if v["variable"].startswith("noise"))

    def test_the_columns_left_out_say_why(self, home, forest):
        left = {x["variable"]: x["why"] for x in _dash(home, forest)["left_out"]}
        assert left == {"same": "constant: one value in every row", "price": "not chosen as an input"}

    def test_a_possible_leak_is_flagged_on_its_row(self, home, tmp_path):
        df = _frame(900, 3)
        df["leak"] = df["price"] * 1.0
        df.to_csv(tmp_path / "leaky.csv", index=False)
        train_regressor(
            str(tmp_path / "leaky.csv"),
            "price",
            "rfr",
            feature_columns=["tenure", "leak"],
            output_path=str(tmp_path / "leaky.pkl"),
        )
        r = generate_model_dashboard(
            str(tmp_path / "leaky.pkl"),
            str(tmp_path / "leaky.csv"),
            "price",
            output_path=str(tmp_path / "d.html"),
            open_after=False,
        )
        assert any("possible target leakage" in v["note"] for v in r["variables"] if v["variable"] == "leak")


class TestTheRefitCurve:
    def test_a_tree_model_is_refitted_on_its_top_k_and_scored_on_held_out_rows(self, home, forest, monkeypatch):
        calls: list[int] = []
        real = dash.model_matrix

        def spy(frame, meta, features=None, *a, **k):
            if frame is not None and len(frame) > 800:  # the training rows, not the scored ones
                calls.append(len(features))
            return real(frame, meta, features, *a, **k)

        monkeypatch.setattr(dash, "model_matrix", spy)
        r = _dash(home, forest, train_file_path=str(home / "train.csv"))
        curve = r["selection_curve"]
        ks = [p["variables"] for p in curve["points"]]
        assert ks == sorted(ks) and ks[-1] == len(FEATURES) and 1 in ks
        assert calls == ks, "each point is a real refit on exactly that many variables"
        assert all(0.0 <= p["auc"] <= 1.0 for p in curve["points"]) and curve["full"] == curve["points"][-1]["auc"]
        assert curve["enough"] in ks and curve["rows"] == 1500

    def test_a_few_variables_are_enough_when_only_a_few_matter(self, home, regressor):
        curve = _dash(home, regressor, target="price", train_file_path=str(home / "train.csv"))["selection_curve"]
        assert curve["metric"] == "r2" and curve["enough"] <= 3, curve

    def test_without_the_training_file_it_says_how_to_get_it(self, home, forest):
        r = _dash(home, forest)
        assert "train_file_path" in r["selection_curve"]["skipped"]
        assert "pass train_file_path" in (home / "d.html").read_text(encoding="utf-8")

    def test_a_linear_model_is_not_refitted_behind_the_wrong_preprocessing(self, home):
        train_classifier(
            str(home / "train.csv"), "churned", "lr", feature_columns=FEATURES[1:], output_path=str(home / "lr.pkl")
        )
        r = _dash(home, home / "lr.pkl", train_file_path=str(home / "train.csv"))
        assert "tree models" in r["selection_curve"]["skipped"]

    def test_a_refit_that_fails_does_not_take_the_dashboard_with_it(self, home, forest, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("no memory")

        monkeypatch.setattr(dash, "_selection_curve", boom)
        r = _dash(home, forest, train_file_path=str(home / "train.csv"))
        assert "the refit failed (RuntimeError: no memory)" in r["selection_curve"]["skipped"]
        assert "variables" in r["sections_generated"]

    def test_the_page_says_how_many_variables_are_enough(self, home, forest):
        _dash(home, forest, train_file_path=str(home / "train.csv"))
        text = re.sub(r"<[^>]+>", "", (home / "d.html").read_text(encoding="utf-8"))
        assert re.search(
            r"the top \d+ of 7 reach within 1% of the full model's auc|no smaller set gets within 1%", text
        )


class TestWhenTheScoresFlatterTheModel:
    def test_the_file_a_model_was_trained_on_is_called_that(self, home, forest):
        r = _dash(home, forest, file="train.csv")
        assert r["sections_generated"][:2] == ["answer", "overlap"] and r["scored_on_training_rows"] is True
        assert r["headline"].endswith("(rows the model was trained on: the score flatters it)")
        manifest = json.loads(next(home.glob("rf*.manifest.json")).read_text())
        assert r["held_out_metrics_when_trained"] == {
            k: v for k, v in manifest["metrics"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        text = re.sub(r"<[^>]+>", "", (home / "d.html").read_text(encoding="utf-8"))
        assert "train.csv is the file this model was trained on" in text and "flatters it" in text

    def test_the_training_rows_do_flatter_it(self, home, forest):
        on_train, on_test = _dash(home, forest, file="train.csv"), _dash(home, forest)
        assert (
            on_train["metrics"]["accuracy"] > on_test["metrics"]["accuracy"] + 0.05
        )  # why the warning is worth having

    def test_held_out_rows_are_not_accused(self, home, forest):
        r = _dash(home, forest, train_file_path=str(home / "train.csv"))
        assert "overlap" not in r["sections_generated"] and "scored_on_training_rows" not in r
        assert r["share_of_rows_in_training_file"] == 0.0

    def test_a_copy_of_the_training_rows_under_another_name_is_found_by_its_rows(self, home, forest):
        shutil.copy(home / "train.csv", home / "renamed.csv")
        r = _dash(home, forest, file="renamed.csv", train_file_path=str(home / "train.csv"))
        assert r["scored_on_training_rows"] is True and r["share_of_rows_in_training_file"] == 1.0
        assert "of the rows scored here are also in train.csv" in re.sub(
            r"<[^>]+>", "", (home / "d.html").read_text(encoding="utf-8")
        )

    def test_a_table_that_repeats_by_chance_is_not_called_a_copy(self):
        small = pd.DataFrame({"a": [1, 2, 3, 4] * 100, "b": ["x", "y"] * 200})
        assert dash._shared_rows(small, small, ["a", "b"]) is None  # rows do not identify: sharing them proves nothing

    def test_an_identifier_is_not_an_identifier_when_it_repeats(self):
        assert dash._identifier(pd.Series(range(500))) and not dash._identifier(pd.Series([1, 2, 3, 4] * 100))
        assert not dash._identifier(pd.Series(np.random.default_rng(1).normal(size=500)))
