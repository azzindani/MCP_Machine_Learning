"""read_receipt found no history for a file the same server had just written.

    run_clustering(in.csv, output_path="clustering_out.csv")   -> success
    read_receipt("clustering_out.csv")
      -> success: true
         entry_count: 0
         entries: []

No error, no hint. A round-16 phase called it out: the receipt log "is not
populated by write ops, so read_receipt's history is effectively always empty
here".

The diagnosis was half right, and the half it missed is the actual bug.
run_clustering *does* append a receipt -- against its **input**. When the labels
go to a different file, the file that was created is left with no provenance at
all. So the question a receipt exists to answer, "where did this file come
from?", is the one question it could not answer about a file this server had
just produced.

Worse, the empty answer was indistinguishable from the other empty answer.
success: true with entry_count 0 means both "nothing has ever been done to this
file" and "its history is filed under a different name", and nothing in the
response separated them.

run_preprocessing and merge_datasets did exactly the same thing, which is why
the fix is one shared helper rather than three copies -- a fix that stops at one
sibling is half a fix, and three hand-written copies are the next drift.

In-place operations are deliberately unchanged: when output == input the
existing receipt already covers it, and a second entry would double-count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from servers.ml_medium import engine as ml

ROWS = "\n".join(f"{i},{i * 2},{i % 3}" for i in range(1, 60))


@pytest.fixture()
def csv_factory(tmp_path: Path):
    def make(name: str) -> Path:
        p = tmp_path / name
        p.write_text(f"a,b,c\n{ROWS}\n", encoding="utf-8")
        return p

    return make


def _entries(path: Path) -> list[dict]:
    r = ml.read_receipt(str(path))
    assert r["success"] is True, r.get("error")
    return r["entries"]


class TestTheCreatedFileCarriesItsOwnProvenance:
    def test_run_clustering(self, csv_factory, tmp_path: Path) -> None:
        src, out = csv_factory("c_in.csv"), tmp_path / "c_out.csv"
        r = ml.run_clustering(
            str(src), feature_columns=["a", "b"], algorithm="kmeans", n_clusters=3, output_path=str(out)
        )
        assert r["success"] is True, r.get("error")
        assert out.is_file()
        entries = _entries(out)
        assert entries, "the file this tool created has no receipt"
        assert entries[0]["tool"] == "run_clustering"
        assert "c_in.csv" in entries[0]["result"], entries[0]

    def test_run_preprocessing(self, csv_factory, tmp_path: Path) -> None:
        src, out = csv_factory("p_in.csv"), tmp_path / "p_out.csv"
        r = ml.run_preprocessing(
            str(src), ops=[{"op": "fill_nulls", "column": "a", "method": "median"}], output_path=str(out)
        )
        assert r["success"] is True, r.get("error")
        entries = _entries(out)
        assert entries, "the file this tool created has no receipt"
        assert entries[0]["tool"] == "run_preprocessing"

    def test_merge_datasets(self, csv_factory, tmp_path: Path) -> None:
        a, b, out = csv_factory("m1.csv"), csv_factory("m2.csv"), tmp_path / "m_out.csv"
        r = ml.merge_datasets(str(a), str(b), on="a", how="inner", output_path=str(out))
        assert r["success"] is True, r.get("error")
        entries = _entries(out)
        assert entries, "the file this tool created has no receipt"
        assert entries[0]["tool"] == "merge_datasets"

    def test_the_receipt_names_the_source(self, csv_factory, tmp_path: Path) -> None:
        """Provenance that does not say what it came from is not provenance."""
        src, out = csv_factory("c_in.csv"), tmp_path / "c_out.csv"
        ml.run_clustering(str(src), feature_columns=["a", "b"], algorithm="kmeans", n_clusters=3, output_path=str(out))
        entry = _entries(out)[0]
        assert entry["args"]["source"] == "c_in.csv", entry


class TestTheInputStillKeepsItsOwn:
    """The new entry is in addition to the old one, not instead of it."""

    def test_input_receipt_survives(self, csv_factory, tmp_path: Path) -> None:
        src, out = csv_factory("c_in.csv"), tmp_path / "c_out.csv"
        ml.run_clustering(str(src), feature_columns=["a", "b"], algorithm="kmeans", n_clusters=3, output_path=str(out))
        assert any(e["tool"] == "run_clustering" for e in _entries(src)), "input lost its receipt"


class TestAnInPlaceRunIsNotDoubleCounted:
    def test_one_entry_when_output_equals_input(self, csv_factory) -> None:
        src = csv_factory("inplace.csv")
        r = ml.run_preprocessing(
            str(src), ops=[{"op": "fill_nulls", "column": "a", "method": "median"}], output_path=str(src)
        )
        assert r["success"] is True, r.get("error")
        runs = [e for e in _entries(src) if e["tool"] == "run_preprocessing"]
        assert len(runs) == 1, f"in-place run recorded {len(runs)} receipts"


class TestAnEmptyLogExplainsItself:
    """success + entry_count 0 + no hint was two different answers at once."""

    def test_it_hints_when_there_is_no_history(self, csv_factory, tmp_path: Path) -> None:
        untouched = csv_factory("untouched.csv")
        r = ml.read_receipt(str(untouched))
        assert r["success"] is True
        assert r["entry_count"] == 0
        assert "hint" in r, r
        assert untouched.name in r["hint"], r["hint"]

    def test_the_hint_says_which_tools_leave_a_record(self, csv_factory) -> None:
        r = ml.read_receipt(str(csv_factory("untouched.csv")))
        assert "run_clustering" in r["hint"], r["hint"]

    def test_the_hint_points_at_the_source_file(self, csv_factory) -> None:
        """The case that produced the finding: history under another name."""
        r = ml.read_receipt(str(csv_factory("untouched.csv")))
        assert "source file" in r["hint"], r["hint"]

    def test_no_hint_when_there_is_a_history(self, csv_factory, tmp_path: Path) -> None:
        src, out = csv_factory("c_in.csv"), tmp_path / "c_out.csv"
        ml.run_clustering(str(src), feature_columns=["a", "b"], algorithm="kmeans", n_clusters=3, output_path=str(out))
        r = ml.read_receipt(str(out))
        assert r["entry_count"] > 0
        assert "hint" not in r, r.get("hint")
