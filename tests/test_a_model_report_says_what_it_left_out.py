"""read_model_report must not present a partial answer as a whole one.

Round 11 read five real models back off the live endpoint. Every response
carried the same three quiet omissions:

    feature_importance   10 entries, importances summing to 0.9456
    feature_columns      15 listed, capped at 20 elsewhere
    confusion_matrix     {}
    classification_report ''

The model had **15** features, not 10. Nothing in the response said so, so the
5.4% the listed importances do not account for reads as unexplained noise in
the model rather than as five features that were never shown. Sorting by
importance and cutting at ten is the right thing to do; doing it without
saying so is not.

`confusion_matrix: {}` came back for every model saved by tune_hyperparameters,
which records `best_score` and nothing else. An empty dict under
`success: true` is indistinguishable from a matrix that happens to be empty,
and gives the caller nothing to do about it.

`classification_report` was `''` on all five -- including the two classifiers
whose metrics are fully populated. It is `''` because **no producer anywhere in
this repo ever writes that key**: `_basic_train` stores `accuracy`,
`f1_weighted` and `confusion_matrix`, and the only other reference is the
sklearn function `_confusion_dict` calls to build per-class stats. The field
survived because the one test covering it hand-built a metadata dict with the
key already in it, proving the truncation branch worked while nothing proved a
real model could reach it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servers.ml_advanced import engine as adv  # noqa: E402
from servers.ml_basic import engine as basic  # noqa: E402


@pytest.fixture
def wide_csv(tmp_path: Path) -> str:
    """Twelve features, so a top-10 cut has something to hide."""
    cols = [f"f{i}" for i in range(12)]
    rows = [",".join(cols + ["target"])]
    for i in range(160):
        vals = [str((i * (j + 3)) % 17) for j in range(12)]
        rows.append(",".join(vals + ["a" if i % 2 else "b"]))
    p = tmp_path / "wide.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def wide_model(wide_csv, tmp_path) -> str:
    out = tmp_path / "m" / "wide.pkl"
    r = basic.train_classifier(wide_csv, "target", "dtc", output_path=str(out))
    assert r["success"] is True, r.get("error")
    return str(out)


class TestATruncatedImportanceListSaysSo:
    def test_the_total_feature_count_is_reported(self, wide_model):
        r = adv.read_model_report(wide_model)
        assert r["success"] is True, r.get("error")
        assert r["feature_importance_total"] == 12, r.get("feature_importance_total")

    def test_showing_fewer_than_all_is_flagged(self, wide_model):
        r = adv.read_model_report(wide_model)
        shown = len(r["feature_importance"])
        assert shown < r["feature_importance_total"], "fixture no longer truncates"
        assert r["feature_importance_truncated"] is True

    def test_the_note_explains_the_missing_mass(self, wide_model):
        # 0.9456 of 1.0 is not noise in the model; it is five features nobody
        # was told about. The note has to name both numbers.
        r = adv.read_model_report(wide_model)
        note = r["feature_importance_note"]
        assert str(len(r["feature_importance"])) in note, note
        assert str(r["feature_importance_total"]) in note, note

    def test_an_untruncated_list_is_not_flagged(self, tmp_path):
        rows = ["a,b,target"]
        for i in range(120):
            rows.append(f"{i},{i % 5},{'x' if i % 2 else 'y'}")
        csv = tmp_path / "narrow.csv"
        csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
        out = tmp_path / "m" / "narrow.pkl"
        basic.train_classifier(str(csv), "target", "dtc", output_path=str(out))
        r = adv.read_model_report(str(out))
        assert r["feature_importance_truncated"] is False
        assert "feature_importance_note" not in r


class TestATruncatedColumnListSaysSo:
    def test_the_total_column_count_is_reported(self, wide_model):
        r = adv.read_model_report(wide_model)
        assert r["feature_columns_total"] == 12


class TestAnAbsentMatrixIsNotAnEmptyOne:
    def test_a_model_without_one_says_why(self, wide_csv, tmp_path):
        # tune_hyperparameters records best_score only, so every model it saves
        # reads back with confusion_matrix {}.
        out = tmp_path / "m" / "tuned.pkl"
        t = adv.tune_hyperparameters(
            wide_csv, "target", "dtc", task="classification", search="random", n_iter=2, cv=2, output_path=str(out)
        )
        assert t["success"] is True, t.get("error")
        r = adv.read_model_report(str(out))
        assert r["confusion_matrix"] == {}
        note = r["confusion_matrix_note"]
        assert "train_classifier" in note, note

    def test_a_model_with_one_carries_no_note(self, wide_model):
        r = adv.read_model_report(wide_model)
        assert r["confusion_matrix"], "fixture model lost its matrix"
        assert "confusion_matrix_note" not in r


class TestTheDeadFieldIsGone:
    def test_a_real_model_does_not_carry_an_empty_report(self, wide_model):
        # No producer writes metadata["classification_report"], so on every real
        # model this key could only ever be ''. Carrying it says "this model has
        # no classification report", which is not what is true.
        r = adv.read_model_report(wide_model)
        assert "classification_report" not in r

    def test_a_populated_report_is_still_returned(self, tmp_path):
        # If a producer ever does write the key, the reader must still surface
        # it -- and still cap it. Note that reaching this branch takes a
        # hand-built metadata dict, which is the whole point.
        from sklearn.ensemble import RandomForestClassifier

        from shared.model_signing import dump_signed

        mp = tmp_path / "hand_built.pkl"
        payload = {
            "model": RandomForestClassifier(n_estimators=2, random_state=42),
            "metadata": {
                "task": "classification",
                "feature_columns": ["x"],
                "target_column": "y",
                "encoding_map": {},
                "metrics": {},
                "classification_report": "x" * 900,
            },
        }
        with open(mp, "wb") as fh:
            dump_signed(payload, fh)
        mp.with_suffix(".manifest.json").write_text("{}")

        r = adv.read_model_report(str(mp))
        assert 0 < len(r["classification_report"]) <= 500
