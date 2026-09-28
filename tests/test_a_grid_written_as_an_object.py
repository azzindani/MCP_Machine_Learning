"""param_grid is read as an object, and a rejected argument is named.

The sweep passed tune_hyperparameters the grid the way anyone writes one --
{"max_depth": [2, 4, 8]} -- and was refused with "rejected its arguments: 1
validation error for tune_hyperparametersArguments": the schema said string,
and the domain tool kept only pydantic's header line, which names neither the
argument nor the type it wanted. The object is now read; the JSON string still
is; and a domain refusal names the field and what was wrong with it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from servers.ml_domain.server import mcp


def _tune(data, **args) -> dict:
    base = {"file_path": str(data), "target_column": "churned", "model": "dtc", "task": "classification", "cv": 3}
    result = asyncio.run(mcp.call_tool("ml_train", {"action": "tune_hyperparameters", "args": {**base, **args}}))
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


@pytest.fixture(autouse=True)
def _out(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))


@pytest.mark.parametrize(
    "grid", [{"max_depth": [2, 4]}, json.dumps({"max_depth": [2, 4]})], ids=["object", "json-string"]
)
def test_the_grid_searched_is_the_one_given(classification_simple, grid):
    r = _tune(classification_simple, param_grid=grid)
    assert r["success"] is True, r
    assert r["best_params"]["max_depth"] in (2, 4)


def test_a_list_is_refused_naming_param_grid(classification_simple):
    r = _tune(classification_simple, param_grid="[2, 4]")
    assert r["success"] is False
    assert "param_grid" in r["error"] and "got list" in r["error"]


def test_the_engine_refuses_a_grid_that_is_not_an_object(classification_simple):
    from servers.ml_advanced.engine import tune_hyperparameters

    r = tune_hyperparameters(str(classification_simple), "churned", "dtc", "classification", param_grid="[2, 4]", cv=3)
    assert r["success"] is False and "must be an object" in r["error"]


def test_a_rejected_argument_is_named_with_what_was_wrong(classification_simple):
    r = _tune(classification_simple, cv="three")
    assert r["success"] is False
    assert "cv: Input should be a valid integer" in r["error"]
    assert "validation error for" not in r["error"]
