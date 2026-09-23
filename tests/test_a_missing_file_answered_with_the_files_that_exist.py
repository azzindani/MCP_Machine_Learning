"""A missing dataset or model is answered with the nearest files that exist.

"File not found: ad_data.csv", hint "Check that file_path is absolute and the
CSV file exists"; "Model file not found: churn.pkl", hint "Train a model
first." A remote caller shares no filesystem with the server and cannot look,
so it guessed again -- and the second hint sent it to retrain a model that was
sitting beside the name it mistyped.

Asserted through the registered tools of every tier, and asserted never to
name anything outside the served folders.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "servers" / "ml_medium"), str(ROOT / "servers" / "ml_basic")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.missing_file import suggest  # noqa: E402

TIERS = ["ml_basic", "ml_medium", "ml_advanced"]


def _tool(tier: str, name: str):
    return importlib.import_module(f"servers.{tier}.server").mcp._tool_manager._tools[name].fn


@pytest.fixture
def served(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "models").mkdir(parents=True)
    monkeypatch.setenv("MCP_CONFINE_PATHS", "1")
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(data))
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.delenv("MCP_DATA_ROOT", raising=False)
    monkeypatch.delenv("MCP_ALLOWED_ROOTS", raising=False)
    monkeypatch.delenv("MCP_CONSTRAINED_MODE", raising=False)
    pd.DataFrame({"x": range(10), "y": [0, 1] * 5}).to_csv(data / "Ad_Data.csv", index=False)
    (data / "models" / "churn_model.pkl").write_bytes(b"")
    (data / ".hidden.csv").write_text("x\n1\n")
    (data / "Ad_Data.csv.mcp_receipt.json").write_text("{}")
    return data


class TestThroughTheTool:
    def test_the_exact_name_found_elsewhere_is_said_to_be_there(self, served):
        # Found by the sweep: "Nothing is named X there. Closest: X" -- the
        # exact name, found where the tool did not look, reported as absent.
        r = _tool("ml_basic", "inspect_dataset")(file_path="models/Ad_Data.csv")
        assert r["success"] is False and r["did_you_mean"][0] == "Ad_Data.csv"
        assert "Nothing is named" not in r["hint"]
        assert "'Ad_Data.csv' is in the data folder" in r["hint"]

    def test_a_case_fold_away_is_named_first(self, served):
        if (served / "ad_data.csv").exists():
            pytest.skip("case-insensitive filesystem: the file is found, nothing to suggest")
        r = _tool("ml_basic", "inspect_dataset")(file_path="ad_data.csv")
        assert r["success"] is False
        assert r["did_you_mean"][0] == "Ad_Data.csv"
        assert "absolute" not in r["hint"]

    def test_a_close_spelling_finds_the_real_one(self, served):
        # A non-CSV extension is refused for its type before anything is looked
        # for, so the misremembered-extension case is the model one below.
        r = _tool("ml_basic", "inspect_dataset")(file_path="Ad-Data.csv")
        assert r["did_you_mean"][0] == "Ad_Data.csv"

    def test_the_fleets_own_sidecars_are_never_suggested(self, served):
        r = _tool("ml_basic", "inspect_dataset")(file_path="Ad-Data.csv")
        assert not [name for name in r["did_you_mean"] if ".mcp_" in name], r["did_you_mean"]

    def test_a_missing_model_names_the_model_that_is_there(self, served):
        r = _tool("ml_basic", "predict_single")(model_path="churn_model.joblib", input_data='{"x": 1}')
        assert r["success"] is False
        assert str(Path("models") / "churn_model.pkl") in r["did_you_mean"]
        assert "Train a model first" not in r["hint"], "the model exists; retraining is the wrong advice"

    def test_nothing_close_lists_what_the_folder_holds(self, served):
        r = _tool("ml_basic", "inspect_dataset")(file_path="zzqx.csv")
        assert "did_you_mean" not in r
        assert "Ad_Data.csv" in r["hint"]
        assert ".hidden.csv" not in r["hint"]
        assert ".mcp_" not in r["hint"], "a receipt is bookkeeping, not a file to pass"

    @pytest.mark.parametrize("tier", TIERS)
    def test_every_tier_installs_it(self, tier):
        tools = importlib.import_module(f"servers.{tier}.server").mcp._tool_manager._tools.values()
        assert tools
        assert all(getattr(t.fn, "__suggests_missing_files__", False) for t in tools)


class TestNeverOutside:
    def test_a_folder_outside_the_served_ones_is_never_listed(self, served, tmp_path):
        outside = tmp_path / "private"
        outside.mkdir()
        (outside / "secrets.csv").write_text("k\n1\n")
        r = suggest(
            {"success": False, "error": "File not found: secret.csv", "hint": "h"},
            {"file_path": str(outside / "secret.csv")},
        )
        assert "secrets.csv" not in str(r)


class TestLeftAlone:
    @pytest.mark.parametrize(
        "result",
        [
            {"success": True, "error": "File not found: a.csv"},
            {"success": False, "error": "Feature columns not found: ['a']"},
        ],
    )
    def test_only_a_missing_file_is_touched(self, served, result):
        assert suggest(dict(result), {"file_path": "Ad_Data.csv"}) == result
