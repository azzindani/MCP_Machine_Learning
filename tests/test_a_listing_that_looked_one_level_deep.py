"""It said there were no models, and there were four.

    list_models(directory="/workspace/data")
      -> success: true, model_count: 0, models: []

Four models had been trained into that tree minutes earlier, each in the
subdirectory its phase worked in. The scan was `search_dir.glob("*.pkl")` --
top level only -- while the docstring says "List all saved .pkl models".

A zero that means "nothing was looked at here" reads exactly like "there are
none", and the response said nothing to tell them apart: no note about depth,
no hint, just an empty list under success. The same shape the sibling servers
have been fixed for twice, in fs_index and in fs_query.

The nested layout is not unusual, either -- it is the default. A trainer called
without an explicit path writes into `.mcp_models` beside the CSV, so the models
this very server creates land one level below anywhere a caller would naturally
point it.

rglob now, and an empty listing says where it looked and what to do about it.

Found in a round-15 sweep report: "returns model_count=0 despite models in
subdirectories. No error text -- it reports success with a misleading answer."
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from servers.ml_basic import engine as mb


def train_into(target: Path, source: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    r = mb.train_classifier(str(source), target_column="churned", model="dtc", output_path=str(target))
    assert r["success"] is True, r.get("error")
    return target


@pytest.fixture()
def tree(tmp_path: Path, classification_simple: Path) -> Path:
    """A root whose models all live one or two levels down."""
    root = tmp_path / "workspace"
    root.mkdir()
    src = root / "data.csv"
    shutil.copy(classification_simple, src)
    train_into(root / "phase_one" / "a.pkl", src)
    train_into(root / "phase_two" / "models" / "b.pkl", src)
    return root


class TestItFindsModelsBelowTheTopLevel:
    def test_both_nested_models_are_listed(self, tree: Path) -> None:
        r = mb.list_models(str(tree))
        assert r["success"] is True, r.get("error")
        assert sorted(m["name"] for m in r["models"]) == ["a.pkl", "b.pkl"], r["models"]

    def test_the_count_matches_the_list(self, tree: Path) -> None:
        r = mb.list_models(str(tree))
        assert r["model_count"] == len(r["models"]) == 2, r

    def test_it_says_the_search_was_recursive(self, tree: Path) -> None:
        """So a caller can tell a real zero from an unexplored one."""
        assert mb.list_models(str(tree))["searched_recursively"] is True

    def test_a_model_at_the_top_level_is_still_found(self, tree: Path) -> None:
        train_into(tree / "top.pkl", tree / "data.csv")
        names = {m["name"] for m in mb.list_models(str(tree))["models"]}
        assert names == {"a.pkl", "b.pkl", "top.pkl"}, names

    def test_the_entries_still_carry_their_manifest_detail(self, tree: Path) -> None:
        """Recursing must not cost the metadata the flat scan attached."""
        entry = mb.list_models(str(tree))["models"][0]
        assert entry.get("model_type"), entry
        assert entry.get("target_column") == "churned", entry
        assert entry.get("loadable") is True, entry


class TestSnapshotsAreNotModels:
    def test_versions_directories_stay_excluded(self, tree: Path) -> None:
        """Recursion reaches .mcp_versions; the filter has to still hold."""
        versions = tree / "phase_one" / ".mcp_versions"
        versions.mkdir(parents=True, exist_ok=True)
        shutil.copy(tree / "phase_one" / "a.pkl", versions / "a_2026-01-01T00-00-00Z.pkl")
        names = [m["name"] for m in mb.list_models(str(tree))["models"]]
        assert names.count("a_2026-01-01T00-00-00Z.pkl") == 0, names
        assert sorted(names) == ["a.pkl", "b.pkl"], names


class TestAnEmptyListingExplainsItself:
    def test_it_still_reports_zero(self, tmp_path: Path) -> None:
        r = mb.list_models(str(tmp_path))
        assert r["success"] is True
        assert r["model_count"] == 0

    def test_it_says_where_it_looked(self, tmp_path: Path) -> None:
        hint = mb.list_models(str(tmp_path))["hint"]
        assert str(tmp_path) in hint, hint
        assert "recursively" in hint, hint

    def test_it_names_what_to_do_next(self, tmp_path: Path) -> None:
        hint = mb.list_models(str(tmp_path))["hint"]
        assert "train_classifier" in hint, hint

    def test_a_listing_with_models_has_no_such_hint(self, tree: Path) -> None:
        r = mb.list_models(str(tree))
        assert "No .pkl model under" not in (r.get("hint") or ""), r.get("hint")
