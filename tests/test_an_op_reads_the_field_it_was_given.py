"""run_preprocessing reads the fields it accepts, or says it does not.

Round 11 gave the Data_Analyst repo's apply_patch a declared field list, a
did-you-mean and a refusal for anything else, because a one-character typo in
an optional field silently wrote different numbers. That work was never ported
here, so run_preprocessing validated op names and two enumerated values and
dropped every other key in silence. Round 14 measured it, and all five of these
reported success:

    clip_column   column=n min=0 max=100  -> nothing clipped; 900 still 900
    label_encode  column=cat new_column=e -> no column e; cat overwritten
    log_transform column=n base=log5      -> natural log, reported as log5
    log_transform column=n method=log10   -> natural log, reported as natural
    drop_outliers column=n threshold=5    -> threshold ignored, 1.5*IQR used

Four of the five are a caller using the spelling the sibling tool in the
Data_Analyst repo documents: clip_values takes min/max, log_transform takes
method, cast_column takes dtype. Both servers are driven by the same model in
the same session, so those are the spellings to expect rather than to punish --
they are aliased, not rejected.

The last two were bare `else` branches, the shape round 13 named: an
if/elif chain ending in `else` turns an unrecognised value into a confident
wrong answer instead of a refusal.
"""

from __future__ import annotations

import pandas as pd
import pytest

from servers.ml_medium._medium_helpers import (
    ALLOWED_OPS,
    LOG_BASES,
    OP_FIELD_ALIASES,
    OP_FIELDS,
    known_op_fields,
)
from servers.ml_medium._medium_preprocess import run_preprocessing


@pytest.fixture
def csv(tmp_path):
    f = tmp_path / "in.csv"
    pd.DataFrame({"cat": ["a", "b", "a", "c"], "n": [1, 50, 3, 900]}).to_csv(f, index=False)
    return f


def _run(csv, tmp_path, op, name="out"):
    out = tmp_path / f"{name}.csv"
    r = run_preprocessing(str(csv), [op], output_path=str(out))
    return r, (pd.read_csv(out) if r["success"] and out.exists() else None)


# --- the five silent drops --------------------------------------------------


@pytest.mark.parametrize("spelling", [{"min": 0, "max": 100}, {"lower": 0, "upper": 100}])
def test_clip_column_clips_under_either_spelling(tmp_path, csv, spelling):
    r, got = _run(csv, tmp_path, {"op": "clip_column", "column": "n", **spelling})
    assert r["success"] is True, r.get("error")
    assert list(got["n"]) == [1, 50, 3, 100], "the 900 was not clipped"


def test_label_encode_honours_new_column(tmp_path, csv):
    r, got = _run(csv, tmp_path, {"op": "label_encode", "column": "cat", "new_column": "cat_enc"})
    assert r["success"] is True, r.get("error")
    assert "cat_enc" in got.columns
    assert list(got["cat"]) == ["a", "b", "a", "c"], "the source column was overwritten"


def test_label_encode_without_new_column_still_replaces_in_place(tmp_path, csv):
    """The documented behaviour is unchanged for callers who never pass it."""
    r, got = _run(csv, tmp_path, {"op": "label_encode", "column": "cat"})
    assert r["success"] is True
    assert list(got["cat"]) == [0, 1, 0, 2]


def test_an_unlisted_log_base_is_refused_not_silently_natural(tmp_path, csv):
    r, _ = _run(csv, tmp_path, {"op": "log_transform", "column": "n", "base": "log5"})
    assert r["success"] is False
    assert "log5" in r["error"]
    for base in LOG_BASES:
        assert base in r["error"]


def test_log_transform_method_spelling_gives_that_transform(tmp_path, csv):
    """method=log10 used to compute a natural log and report it as such."""
    r, got = _run(csv, tmp_path, {"op": "log_transform", "column": "n", "method": "log10"})
    assert r["success"] is True, r.get("error")
    assert got["n_log"].iloc[3] == pytest.approx(2.9542425094, rel=1e-9)  # log10(900), not ln(900)


def test_drop_outliers_reads_its_threshold(tmp_path):
    """A tighter fence must remove more rows than a wider one."""
    f = tmp_path / "w.csv"
    pd.DataFrame({"n": [1, 2, 3, 4, 5, 6, 7, 8, 9, 200]}).to_csv(f, index=False)
    tight, got_tight = _run(f, tmp_path, {"op": "drop_outliers", "column": "n", "threshold": 0.1}, "tight")
    wide, got_wide = _run(f, tmp_path, {"op": "drop_outliers", "column": "n", "threshold": 50}, "wide")
    assert tight["success"] and wide["success"]
    assert len(got_tight) < len(got_wide), "threshold changed nothing"


def test_an_unlisted_outlier_method_is_refused(tmp_path, csv):
    r, _ = _run(csv, tmp_path, {"op": "drop_outliers", "column": "n", "method": "zscore"})
    assert r["success"] is False
    assert "zscore" in r["error"] and "iqr" in r["error"]


# --- unknown fields ---------------------------------------------------------


def test_a_misspelled_field_is_refused_with_a_suggestion(tmp_path, csv):
    r, _ = _run(csv, tmp_path, {"op": "clip_column", "column": "n", "lowr": 0})
    assert r["success"] is False
    assert "lowr" in r["error"]
    assert "did you mean lower?" in r["error"]


def test_a_field_from_no_vocabulary_at_all_is_refused(tmp_path, csv):
    r, _ = _run(csv, tmp_path, {"op": "drop_column", "column": "n", "recursive": True})
    assert r["success"] is False
    assert "recursive" in r["error"]


def test_scale_accepts_one_column_as_a_bare_string(tmp_path, csv):
    """It reached StandardScaler as a 1-D Series and raised out of the tool."""
    r, got = _run(csv, tmp_path, {"op": "scale", "columns": "n"})
    assert r["success"] is True, r.get("error")
    assert got["n"].mean() == pytest.approx(0.0, abs=1e-9)


# --- the table itself -------------------------------------------------------


def test_every_allowed_op_declares_its_fields():
    """An op absent from OP_FIELDS accepts anything and drops what it does not
    read -- which is the hole this whole file exists to close."""
    assert set(OP_FIELDS) == ALLOWED_OPS


def test_no_alias_shadows_a_real_field():
    """An alias whose source name is also a field the op reads would rename a
    field out from under the handler."""
    for op_name, aliases in OP_FIELD_ALIASES.items():
        for given, canonical in aliases.items():
            assert canonical in OP_FIELDS[op_name], f"{op_name}: alias target {canonical} is not read"
            if op_name != "scale":  # scale maps the global column->columns alias deliberately
                assert given not in OP_FIELDS[op_name], f"{op_name}: alias {given} is also a real field"


def test_known_fields_covers_documented_and_aliased_spellings():
    known = known_op_fields("clip_column")
    for name in ("column", "lower", "upper", "min", "max", "op"):
        assert name in known
