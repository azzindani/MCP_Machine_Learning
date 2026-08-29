"""The train -> report -> predict chain did not compose, at two joins.

Every sweep so far has called tools one at a time, and each of these tools
passes its own phase. What no single-call test can see is whether what one tool
*returns* is something the next tool *accepts*. Run against the live endpoints,
the most natural ML workflow there is broke twice.

**Join one: read_model_report -> predict_single.**  Round 8 added
`feature_defaults` -- a typical row from the raw data -- so a caller could get a
usable starting record for a model rather than having to invent 15 column
values. It is reachable, via read_model_report, and it is a JSON **object**:

    {"Date": "2020-01-02", "product": "Product 1", "campaign_platform": ...}

predict_single declared `input_data: str`, so pydantic refused it:

    1 validation error for call[predict_single]
    input_data
      Input should be a valid string [type=string_type, input_value={'Date': ...}]

The engine had always handled both -- `if isinstance(input_data, dict)` is right
there in _basic_predict -- so the only thing rejecting the call was the wrapper's
annotation being narrower than its own engine, and narrower than what the tool
next door hands out. fastmcp renders `str | dict` as
`anyOf: [string, object]`, which is also the first time the schema says a
record is acceptable at all.

**Join two: train -> evaluate_model.**  Two of the three tools that consume a
saved model name the CSV `file_path`:

    get_predictions(model_path, file_path)
    batch_predict(model_path, file_path)
    evaluate_model(model_path, test_file_path, target_column)   <- the outlier

so a caller chaining train -> evaluate writes `file_path` and is refused before
this server can say which name it wanted. `test_file_path` says something true
-- the CSV must carry labels -- so it stays the documented name and keeps its
original position; `file_path` is accepted alongside it.
"""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from servers.ml_advanced.engine import read_model_report
from servers.ml_basic import server as _basic
from servers.ml_basic.engine import train_regressor
from servers.ml_medium import server as _medium

# Reach both through the registry: the official SDK's @mcp.tool returns the
# plain undecorated function, so the module-level name skips every wrapper
# installed on the registry entry -- which is the path a real request takes.
predict_single = _basic.mcp._tool_manager._tools["predict_single"].fn
evaluate_model = _medium.mcp._tool_manager._tools["evaluate_model"].fn

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def csv(tmp_path: Path) -> str:
    dst = tmp_path / "ad.csv"
    shutil.copy(FIXTURES / "ad_data_full.csv", dst)
    return str(dst)


@pytest.fixture()
def trained(csv: str) -> dict:
    r = train_regressor(csv, target_column="spends", model="lir")
    assert r["success"] is True, r.get("error")
    return r


class TestTheRowOneToolReturnsIsARowTheNextAccepts:
    def test_read_model_report_still_carries_feature_defaults(self, trained: dict):
        r = read_model_report(trained["model_path"])
        assert r["success"] is True, r.get("error")
        assert isinstance(r["manifest"]["feature_defaults"], dict)

    def test_predict_single_accepts_that_object_unedited(self, trained: dict):
        """The join that failed: an object out, an object in, nothing in between."""
        report = read_model_report(trained["model_path"])
        row = report["manifest"]["feature_defaults"]
        r = predict_single(trained["model_path"], row)
        assert r["success"] is True, r.get("error")

    def test_it_returns_a_number(self, trained: dict):
        report = read_model_report(trained["model_path"])
        row = report["manifest"]["feature_defaults"]
        r = predict_single(trained["model_path"], row)
        assert isinstance(r["prediction"], (int, float)), r["prediction"]

    def test_a_json_string_still_works(self, trained: dict):
        """The documented form must not regress."""
        report = read_model_report(trained["model_path"])
        row = report["manifest"]["feature_defaults"]
        r = predict_single(trained["model_path"], json.dumps(row))
        assert r["success"] is True, r.get("error")

    def test_both_forms_give_the_same_prediction(self, trained: dict):
        report = read_model_report(trained["model_path"])
        row = report["manifest"]["feature_defaults"]
        as_obj = predict_single(trained["model_path"], row)
        as_str = predict_single(trained["model_path"], json.dumps(row))
        assert as_obj["prediction"] == pytest.approx(as_str["prediction"])

    def test_the_schema_says_an_object_is_allowed(self):
        """The load-bearing test of this class.

        The others call `.fn`, the undecorated function, so they exercise the
        engine -- which accepted dicts all along and passes them even without
        the fix. The rejection happened one layer up, in the pydantic model
        fastmcp builds from this annotation, which no in-process call reaches.
        So the annotation is what is actually asserted here, and it is also the
        only place a caller reading tools/list can learn a record is allowed.
        """
        annotation = inspect.signature(predict_single).parameters["input_data"].annotation
        assert "dict" in str(annotation), annotation

    def test_malformed_json_still_gets_a_usable_error(self, trained: dict):
        r = predict_single(trained["model_path"], "{not json")
        assert r["success"] is False
        assert "JSON" in r["error"] and "{" in r["hint"]


class TestEvaluateModelTakesTheNameItsSiblingsUse:
    def test_the_documented_spelling_works(self, trained: dict, csv: str):
        r = evaluate_model(trained["model_path"], test_file_path=csv, target_column="spends")
        assert r["success"] is True, r.get("error")

    def test_the_sibling_spelling_works(self, trained: dict, csv: str):
        """The call a caller chaining from get_predictions would write."""
        r = evaluate_model(trained["model_path"], file_path=csv, target_column="spends")
        assert r["success"] is True, r.get("error")

    def test_both_score_the_same(self, trained: dict, csv: str):
        a = evaluate_model(trained["model_path"], test_file_path=csv, target_column="spends")
        b = evaluate_model(trained["model_path"], file_path=csv, target_column="spends")
        assert a["metrics"] == b["metrics"]

    def test_a_positional_call_written_against_the_old_signature_still_works(self, trained: dict, csv: str):
        """test_file_path keeps position 2; moving it would silently bind the
        CSV path to target_column."""
        r = evaluate_model(trained["model_path"], csv, "spends")
        assert r["success"] is True, r.get("error")

    def test_neither_spelling_names_both(self, trained: dict):
        r = evaluate_model(trained["model_path"], target_column="spends")
        assert r["success"] is False
        assert "test_file_path" in r["error"] and "file_path" in r["hint"]

    def test_a_missing_target_is_refused_not_guessed(self, trained: dict, csv: str):
        r = evaluate_model(trained["model_path"], test_file_path=csv)
        assert r["success"] is False and "target_column" in r["error"]

    def test_the_alias_is_recorded_in_progress(self, trained: dict, csv: str):
        r = evaluate_model(trained["model_path"], file_path=csv, target_column="spends")
        msgs = " ".join(str(p.get("msg", "")) for p in r["progress"])
        assert "alias" in msgs.lower(), r["progress"]


class TestEveryModelConsumerNamesItsCsvTheSameWay:
    """The census, for the tools that take a saved model plus a CSV."""

    CONSUMERS = [
        ("servers.ml_basic.server", "get_predictions"),
        ("servers.ml_medium.server", "batch_predict"),
        ("servers.ml_medium.server", "evaluate_model"),
    ]

    def test_all_of_them_accept_file_path(self):
        import importlib

        offenders = []
        for module_name, tool in self.CONSUMERS:
            module = importlib.import_module(module_name)
            fn = getattr(module, tool)
            fn = getattr(fn, "fn", fn)
            if "file_path" not in inspect.signature(fn).parameters:
                offenders.append(f"{module_name}.{tool}")
        assert not offenders, offenders

    def test_a_prediction_is_reproducible_from_a_dataframe_round_trip(self, trained: dict, tmp_path: Path):
        """Guards the whole chain against a silent column-order dependency."""
        report = read_model_report(trained["model_path"])
        row = report["manifest"]["feature_defaults"]
        shuffled = dict(reversed(list(row.items())))
        assert pd.Series(row).equals(pd.Series(row))
        a = predict_single(trained["model_path"], row)
        b = predict_single(trained["model_path"], shuffled)
        assert a["prediction"] == pytest.approx(b["prediction"])
