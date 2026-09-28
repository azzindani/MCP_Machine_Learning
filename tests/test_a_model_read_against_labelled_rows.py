"""A model read against labelled rows: every number on its dashboard is the model's own, recomputed.

The training report printed what a model scored when it was trained, and no
more: nothing about where the threshold should sit when a miss costs five
false alarms, which features move its predictions, which segments it fails,
or whether new rows still look like the ones it learned from.
generate_model_dashboard scores a labelled file and draws each of those; these
tests check each number against a brute-force or sklearn computation.
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
from sklearn.metrics import r2_score, roc_auc_score

from servers.ml_advanced._adv_dashboard import _num
from servers.ml_advanced._adv_viz import generate_cluster_report
from servers.ml_advanced.engine import generate_model_dashboard
from servers.ml_basic._basic_train import train_classifier, train_regressor


def _churn(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n),
            "charges": rng.normal(70, 20, n).round(2),
            "plan": rng.choice(["basic", "pro"], n),
            "region": rng.choice(["N", "S", "<b>W</b>"], n),
            "noise": rng.normal(0, 1, n),
        }
    )
    logit = -1 + 0.06 * (df["charges"] - 70) - 0.08 * df["tenure"] + (df["plan"] == "basic") * 1.5
    df["churned"] = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    return df


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_DATA_ROOT", str(tmp_path))
    _churn(1200, 1).to_csv(tmp_path / "train.csv", index=False)
    _churn(500, 2).to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


@pytest.fixture
def model(home):
    r = train_classifier(str(home / "train.csv"), "churned", "lr", output_path=str(home / "lr.pkl"))
    assert r["success"] is True, r
    return home / "lr.pkl"


def _dash(home, model, **kw) -> dict:
    r = generate_model_dashboard(
        str(model), str(home / "test.csv"), output_path=str(home / "d.html"), open_after=False, **kw
    )
    assert r["success"] is True, r.get("error")
    return r


def _scores(home, model) -> tuple[np.ndarray, np.ndarray]:
    """The model's positive-class scores on test.csv, and the truth, the way the dashboard computes them."""
    from servers.ml_advanced._adv_dashboard import _Scorer
    from servers.ml_advanced._adv_helpers import _load_model

    scorer = _Scorer(*_load_model(str(model)))
    df = pd.read_csv(home / "test.csv")
    _, proba = scorer.scores(df)
    return proba[:, 1], scorer.truth(df["churned"])


def test_it_is_read_against_the_majority_baseline(home, model):
    r = _dash(home, model)
    y = pd.read_csv(home / "test.csv")["churned"]
    assert r["baseline"]["accuracy"] == pytest.approx(round(float((y == y.mode()[0]).mean()), 4))
    assert r["baseline"]["rule"] == f"always predicting {y.mode()[0]!r}, the most common class"
    pos, truth = _scores(home, model)
    assert r["metrics"]["auc"] == pytest.approx(round(roc_auc_score(truth, pos), 4))
    assert r["headline"].startswith(f"{r['metrics']['accuracy']:.1%} accurate against")


def test_the_cheapest_threshold_is_the_cheapest(home, model):
    r = _dash(home, model, cost_fp=1, cost_fn=5)
    pos, y = _scores(home, model)

    def cost(t: float) -> float:
        hit = pos >= t
        return float(np.sum(hit & (y == 0)) * 1 + np.sum(~hit & (y == 1)) * 5)

    best = r["threshold"]["cheapest"]
    assert best["cost"] == pytest.approx(min(cost(t) for t in np.round(np.arange(0.01, 1.0, 0.01), 2)))
    assert best["cost"] == pytest.approx(cost(best["threshold"]))
    assert r["threshold"]["default"]["cost"] == pytest.approx(cost(0.5))


def test_the_page_opens_with_no_network(home, model):
    _dash(home, model)
    html = (home / "d.html").read_text(encoding="utf-8")
    assert not re.search(r"<(script|link|img|iframe)\b[^>]*\b(src|href)=[\"']https?://", html, flags=re.I)


def test_the_threshold_slider_recounts_in_the_page(home, model):
    _dash(home, model)
    html = (home / "d.html").read_text(encoding="utf-8")
    assert '<input id="th-range" type="range"' in html and 'id="th-data"' in html
    if shutil.which("node"):
        script = re.findall(r"<script>(\s*\(function\(\)\{\s*var d=JSON\.parse.*?)</script>", html, flags=re.S)
        assert script, "the slider's script is on the page"
        src = home / "slider.js"
        src.write_text(script[0], encoding="utf-8")
        assert subprocess.run(["node", "--check", str(src)], capture_output=True).returncode == 0


def test_lift_climbs_to_every_positive(home, model):
    deciles = _dash(home, model)["lift"]
    shares = [d["share_of_positives"] for d in deciles]
    assert shares == sorted(shares) and shares[-1] == 1.0
    pos, y = _scores(home, model)
    top = np.argsort(-pos, kind="stable")[: round(len(y) * 0.2)]
    assert deciles[1]["share_of_positives"] == pytest.approx(round(float((y[top] == 1).sum() / (y == 1).sum()), 4))


def test_what_moves_the_predictions_is_what_made_the_labels(home, model):
    r = _dash(home, model)
    ranked = [i["feature"] for i in r["importance"]]
    assert ranked.index("noise") > ranked.index("tenure")
    tenure = next(p for p in r["partial_dependence"] if p["feature"] == "tenure")
    assert tenure["y"][0] > tenure["y"][-1], "longer tenure, less churn"
    assert any(c["term"].startswith("plan = pro") for c in r["coefficients"])


def test_a_segment_it_fails_is_named(home, model):
    df = pd.read_csv(home / "test.csv")
    rng = np.random.default_rng(5)
    flip = (df["region"] == "S") & (rng.uniform(size=len(df)) < 0.5)
    df.loc[flip, "churned"] = np.where(df.loc[flip, "churned"] == "yes", "no", "yes")
    df.to_csv(home / "test.csv", index=False)
    r = _dash(home, model)
    assert r["weak_segments"][0]["column"] == "region" and r["weak_segments"][0]["value"] == "S"
    html = (home / "d.html").read_text(encoding="utf-8")
    assert "<b>W</b>" not in re.sub(r"<script\b.*?</script>", "", html, flags=re.S), "a segment value is text"


def test_drift_is_measured_against_the_training_rows(home, model):
    df = pd.read_csv(home / "test.csv")
    df["charges"] = df["charges"] + 25
    df.to_csv(home / "test.csv", index=False)
    r = _dash(home, model, train_file_path=str(home / "train.csv"))
    drift = {d["feature"]: d for d in r["drift"]}
    assert drift["charges"]["shift"] == "major" and drift["tenure"]["shift"] == "stable"


def test_the_leaderboard_ranks_models_on_the_same_rows(home, model):
    r = train_classifier(str(home / "train.csv"), "churned", "dtc", output_path=str(home / "tree.pkl"))
    assert r["success"] is True, r
    board = _dash(home, model, compare_model_paths=[str(home / "tree.pkl")])["leaderboard"]
    assert {b["model"] for b in board} == {"lr.pkl", "tree.pkl"}
    assert [b["auc"] for b in board] == sorted((b["auc"] for b in board), reverse=True)


def test_a_regressor_is_read_against_the_mean(home):
    df = _churn(800, 3)
    df["spend"] = 5 * df["tenure"] + df["charges"] + np.random.default_rng(4).normal(0, 5, len(df))
    df.drop(columns="churned").to_csv(home / "reg.csv", index=False)
    r = train_regressor(str(home / "reg.csv"), "spend", "lir", output_path=str(home / "reg.pkl"))
    assert r["success"] is True, r
    d = generate_model_dashboard(
        str(home / "reg.pkl"), str(home / "reg.csv"), output_path=str(home / "r.html"), open_after=False
    )
    assert d["success"] is True, d
    assert d["metrics"]["r2"] > 0.9 and d["metrics"]["rmse"] < d["baseline"]["rmse"]
    assert "residuals" in d["sections_generated"] and "threshold" not in d["sections_generated"]


@pytest.mark.parametrize(
    ("value", "shown"),
    [(19980.4, "19,980"), (454000.0, "454,000"), (1234.56, "1,235"), (3.14159, "3.142"), (0.0012345, "0.001234")],
)
def test_an_error_reads_in_the_targets_own_digits(value, shown):
    assert _num(value) == shown


def test_a_multiclass_model_has_no_threshold(home):
    df = _churn(900, 6)
    df["tier"] = pd.cut(df["charges"], [-np.inf, 60, 80, np.inf], labels=["low", "mid", "high"]).astype(str)
    df.drop(columns="churned").to_csv(home / "tier.csv", index=False)
    r = train_classifier(str(home / "tier.csv"), "tier", "dtc", output_path=str(home / "tier.pkl"))
    assert r["success"] is True, r
    d = generate_model_dashboard(
        str(home / "tier.pkl"), str(home / "tier.csv"), output_path=str(home / "t.html"), open_after=False
    )
    assert d["success"] is True, d
    assert "threshold" not in d and {c["class"] for c in d["per_class"]} == {"low", "mid", "high"}
    assert "confusion" in d["sections_generated"] and "roc" not in d["sections_generated"]
    assert sum(c["support"] for c in d["per_class"]) == len(df)


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"target_column": "nope"}, "Target column 'nope' is not in test.csv"),
        ({"cost_fn": -1}, "cost_fp and cost_fn are costs"),
        ({"segment_columns": ["nope"]}, "segment_columns names column(s) not in test.csv: nope"),
    ],
)
def test_what_it_cannot_read_is_refused_by_name(home, model, kw, says):
    r = generate_model_dashboard(str(model), str(home / "test.csv"), open_after=False, **kw)
    assert r["success"] is False and says in r["error"], r


def test_a_file_without_the_models_features_is_refused(home, model):
    pd.read_csv(home / "test.csv").drop(columns=["tenure"]).to_csv(home / "less.csv", index=False)
    r = generate_model_dashboard(str(model), str(home / "less.csv"), open_after=False)
    assert r["success"] is False and "lacks the model's feature column(s): tenure" in r["error"]


def test_a_regression_r2_is_sklearns(home):
    df = _churn(600, 8)
    df["spend"] = 3 * df["charges"] + np.random.default_rng(9).normal(0, 20, len(df))
    df.drop(columns="churned").to_csv(home / "s.csv", index=False)
    assert train_regressor(str(home / "s.csv"), "spend", "lir", output_path=str(home / "s.pkl"))["success"]
    d = generate_model_dashboard(
        str(home / "s.pkl"), str(home / "s.csv"), output_path=str(home / "s.html"), open_after=False
    )
    from servers.ml_advanced._adv_dashboard import _Scorer
    from servers.ml_advanced._adv_helpers import _load_model

    scorer = _Scorer(*_load_model(str(home / "s.pkl")))
    pred, _ = scorer.scores(df)
    assert d["metrics"]["r2"] == pytest.approx(round(r2_score(df["spend"], pred), 4))


def test_a_cluster_is_named_by_what_sets_it_apart(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    rng = np.random.default_rng(11)
    rows = [{"income": rng.normal(90, 5), "age": rng.normal(30, 3), "cluster_label": 0} for _ in range(60)]
    rows += [{"income": rng.normal(40, 5), "age": rng.normal(60, 3), "cluster_label": 1} for _ in range(40)]
    pd.DataFrame(rows).to_csv(tmp_path / "c.csv", index=False)
    r = generate_cluster_report(str(tmp_path / "c.csv"), ["income", "age"], "cluster_label", open_after=False)
    assert r["success"] is True, r
    personas = {p["cluster"]: p for p in r["personas"]}
    assert personas["0"]["name"] == "high income, low age" or personas["0"]["name"] == "low age, high income"
    assert personas["1"]["share"] == 0.4 and personas["1"]["name"].startswith(("low income", "high age"))
    page = Path(r["output_path"]).read_text(encoding="utf-8")
    assert "Personas" in page and "— 40 rows, 40.0%" in page


def test_the_response_is_json_a_strict_reader_accepts(home, model):
    df = pd.read_csv(home / "test.csv")
    df.loc[df.index[:40], "charges"] = np.nan
    df.to_csv(home / "test.csv", index=False)
    r = _dash(home, model, train_file_path=str(home / "train.csv"))
    json.dumps(r, allow_nan=False)
