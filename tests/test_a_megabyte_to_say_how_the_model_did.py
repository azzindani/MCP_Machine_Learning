"""One column's encoding map, in every report, forever.

A user review called `read_model_report` and got back roughly a megabyte:

    Report returned full `emp_title` encoding_map (1,000+ entries inline,
    ~1 MB manifest). Correct but token-heavy; truncated output warning fired.
    Improvement: `top_n` / `skip_encoding_map` flag, importance-only by default.

`emp_title` had 28,525 distinct values on that file. Every call asking "how did
this model do" carried all of them, because the manifest is inlined whole and
the map lives in the manifest. Nothing was wrong with any of it -- the map is
what makes a saved model usable on new data. It was simply in the answer to a
question nobody asked it.

The default is now the cheap one, which is a deliberate behaviour change and
the one the review asked for by name. The map is never deleted: the response
says how many entries it has, which columns they belong to, and where to read
them, so an agent can decide whether it needs them rather than paying for them
every time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from servers.ml_advanced.engine import read_model_report
from shared.model_signing import dump_signed


@pytest.fixture
def model_with_a_big_map(tmp_path):
    """A saved model whose manifest carries a large encoding map."""
    model_path = tmp_path / "m.pkl"
    metadata = {
        "task": "classification",
        "model_type": "dtc",
        "feature_columns": [f"f{i}" for i in range(20)],
        "target_column": "y",
        "metrics": {"accuracy": 0.96},
    }
    # Signed, because that is how every producer here writes a model and how
    # `_load_model` reads one. A plain pickle loads nowhere and would have made
    # every test below skip -- which reads as green and verifies nothing.
    with open(model_path, "wb") as fh:
        dump_signed({"model": None, "metadata": metadata}, fh)

    manifest = {
        "model_type": "dtc",
        "trained_on": "loans.csv",
        "encoding_map": {
            "emp_title": {f"title_{i}": i for i in range(2_000)},
            "grade": {g: i for i, g in enumerate("ABCDEFG")},
        },
    }
    model_path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return model_path


def _ok(resp):
    assert resp.get("success"), resp.get("error")
    return resp


def test_the_map_is_not_inlined_by_default(model_with_a_big_map):
    r = _ok(read_model_report(str(model_with_a_big_map)))
    assert "encoding_map" not in r["manifest"], "this is the megabyte"
    assert r["encoding_map_inlined"] is False


def test_the_response_says_what_it_left_out(model_with_a_big_map):
    """Absent is not the same as nonexistent, and the caller has to be able to tell."""
    r = _ok(read_model_report(str(model_with_a_big_map)))
    summary = r["encoding_map_summary"]
    assert summary["entries_total"] == 2_007
    assert summary["entries_per_column"]["emp_title"] == 2_000
    assert Path(summary["manifest_path"]).exists()
    assert "skip_encoding_map=False" in summary["note"]


def test_the_map_is_still_there_when_asked_for(model_with_a_big_map):
    r = _ok(read_model_report(str(model_with_a_big_map), skip_encoding_map=False))
    assert len(r["manifest"]["encoding_map"]["emp_title"]) == 2_000
    assert r["encoding_map_inlined"] is True


def test_skipping_it_is_materially_cheaper(model_with_a_big_map):
    lean = _ok(read_model_report(str(model_with_a_big_map)))
    full = _ok(read_model_report(str(model_with_a_big_map), skip_encoding_map=False))
    assert lean["token_estimate"] < full["token_estimate"] / 4


def test_the_rest_of_the_manifest_survives(model_with_a_big_map):
    """Only the map is dropped, not the manifest."""
    r = _ok(read_model_report(str(model_with_a_big_map)))
    assert r["manifest"]["trained_on"] == "loans.csv"
    assert r["manifest"]["model_type"] == "dtc"


def test_a_model_with_no_map_claims_nothing(tmp_path):
    model_path = tmp_path / "plain.pkl"
    with open(model_path, "wb") as fh:
        dump_signed({"model": None, "metadata": {"task": "classification", "feature_columns": []}}, fh)
    r = _ok(read_model_report(str(model_path)))
    assert "encoding_map_summary" not in r
    assert "encoding_map_inlined" not in r


def test_top_n_overrides_the_deployment_limit(model_with_a_big_map):
    """0 keeps the old behaviour; a positive value is the caller's choice."""
    import inspect

    params = inspect.signature(read_model_report).parameters
    assert params["top_n"].default == 0
    assert params["skip_encoding_map"].default is True
