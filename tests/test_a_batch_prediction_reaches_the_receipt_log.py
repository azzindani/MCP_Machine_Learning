"""The one tool that can overwrite a file with model output logged nothing.

read_receipt reads one per-file log. filter_rows, run_preprocessing and
run_clustering on this server all write to it. batch_predict did not — even
though it takes a snapshot before overwriting an existing output file, so it
plainly knows it is making a change worth being able to undo. Ask read_receipt
what produced a predictions CSV and the answer was nothing at all.

Found by asking which functions take a snapshot but never write a receipt —
39 of 93 across the four repos, of which the ones worth fixing are the ones a
receipt *reader* exists to expose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "servers" / "ml_medium"), str(ROOT / "servers" / "ml_basic")):
    if p not in sys.path:
        sys.path.insert(0, p)

from servers.ml_basic import engine as basic  # noqa: E402
from servers.ml_medium import engine as medium  # noqa: E402


def logged(path) -> list[str]:
    # Through read_receipt_log(), which is what the read_receipt tool calls:
    # what matters is not that a line reached a file but that the reader shows it.
    from shared.receipt import read_receipt_log

    return [e.get("tool") for e in read_receipt_log(str(path), 50)]


@pytest.fixture
def trained(tmp_path):
    csv = tmp_path / "d.csv"
    rows = ["x,y,label"] + [f"{i},{i * 2},{i % 2}" for i in range(60)]
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    r = basic.train_classifier(str(csv), "label", "rf")
    assert r["success"] is True, r.get("error")
    return csv, r["model_path"]


class TestBatchPredictIsLogged:
    def test_writing_predictions_leaves_a_trace(self, trained, tmp_path):
        csv, model = trained
        out = tmp_path / "preds.csv"
        r = medium.batch_predict(str(model), str(csv), output_path=str(out))
        assert r["success"] is True, r.get("error")
        assert out.exists()
        assert "batch_predict" in logged(out), logged(out)

    def test_overwriting_records_the_snapshot_it_took(self, trained, tmp_path):
        from shared.receipt import read_receipt_log

        csv, model = trained
        out = tmp_path / "preds.csv"
        medium.batch_predict(str(model), str(csv), output_path=str(out))
        # Second run overwrites, so a snapshot is taken and must be findable.
        medium.batch_predict(str(model), str(csv), output_path=str(out))

        entries = [e for e in read_receipt_log(str(out), 50) if e.get("tool") == "batch_predict"]
        assert len(entries) == 2, entries
        assert any(e.get("backup") for e in entries), entries
