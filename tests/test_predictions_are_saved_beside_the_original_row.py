"""batch_predict must not write the model's view of a row back as the row.

The label encoding exists to build the feature matrix. It was applied to `df`
in place, and `df` is what gets written out -- so a row that went in as

    campaign_platform="Google Ads", device="Desktop"

was saved as

    campaign_platform=1, device=0

under those same column headers, reported as success with a correct prediction.
The file misrepresents itself, and nothing short of reading it back shows it:
the response is right, the number is right, and the header row is right.

Found by the round-13 re-run, which read the written CSV rather than trusting
the response -- the sweep's own technique 18, applied to a file this time.

Second, smaller thing locked here: `fillna(-1)` turns a category the model was
never trained on into a real category as far as the model is concerned. That is
the only thing to do with it, but the caller should be told it happened.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servers.ml_basic.engine import train_regressor  # noqa: E402
from servers.ml_medium._medium_data import batch_predict  # noqa: E402

HEADER = "platform,device,impressions,clicks"
ROWS = [
    ("Google Ads", "Desktop", 100, 4),
    ("Facebook", "Mobile", 250, 9),
    ("Google Ads", "Mobile", 80, 2),
    ("Instagram", "Desktop", 310, 11),
]


@pytest.fixture()
def trained(tmp_path):
    train = tmp_path / "train.csv"
    body = "\n".join(f"{p},{d},{i},{c}" for p, d, i, c in ROWS * 6)
    train.write_text(f"{HEADER}\n{body}\n")
    model = tmp_path / "m.pkl"
    r = train_regressor(file_path=str(train), target_column="clicks", model="rfr", output_path=str(model))
    assert r["success"] is True, r
    return model


def test_the_saved_row_is_the_row_that_went_in(tmp_path, trained):
    src = tmp_path / "one.csv"
    src.write_text(f"{HEADER}\nGoogle Ads,Desktop,100,4\n")
    out = tmp_path / "preds.csv"

    r = batch_predict(model_path=str(trained), file_path=str(src), output_path=str(out))
    assert r["success"] is True, r

    written = list(csv.DictReader(out.open()))
    assert len(written) == 1
    row = written[0]
    # The bug: these came back as "1" and "0".
    assert row["platform"] == "Google Ads"
    assert row["device"] == "Desktop"
    assert row["impressions"] == "100"
    assert "prediction" in row


def test_every_source_column_survives_unchanged(tmp_path, trained):
    src = tmp_path / "many.csv"
    body = "\n".join(f"{p},{d},{i},{c}" for p, d, i, c in ROWS)
    src.write_text(f"{HEADER}\n{body}\n")
    out = tmp_path / "preds.csv"

    assert batch_predict(model_path=str(trained), file_path=str(src), output_path=str(out))["success"]

    original = list(csv.DictReader(src.open()))
    written = list(csv.DictReader(out.open()))
    assert len(written) == len(original)
    for before, after in zip(original, written):
        for column, value in before.items():
            assert after[column] == value, column
    # and the one column the tool is entitled to add
    assert set(written[0]) - set(original[0]) == {"prediction"}


def test_an_unseen_category_is_reported_not_silently_encoded(tmp_path, trained):
    """-1 is the only sensible encoding for a category never trained on.

    Letting it ride under success:true is the part that is not sensible: the
    model scores it as a real category, and the caller has no way to know.
    """
    src = tmp_path / "novel.csv"
    src.write_text(f"{HEADER}\nTikTok,Desktop,100,4\n")
    out = tmp_path / "preds.csv"

    r = batch_predict(model_path=str(trained), file_path=str(src), output_path=str(out))
    assert r["success"] is True
    assert r["unmapped_categories"].get("platform") == 1
    # This repo's progress entries are {"icon": "⚠", "msg": ..., "detail": ...}
    # -- the sibling repos use {"status": ..., "message": ...}, and asserting
    # the wrong shape here silently matched nothing rather than failing loudly.
    warnings = [p for p in r["progress"] if p.get("icon") == "⚠"]
    assert any("Unseen categories" in p["msg"] for p in warnings), r["progress"]
    assert any("platform: 1 row" in p.get("detail", "") for p in warnings), warnings
    # The written row still carries what the caller sent.
    assert list(csv.DictReader(out.open()))[0]["platform"] == "TikTok"


def test_a_fully_known_input_reports_nothing_unmapped(tmp_path, trained):
    src = tmp_path / "known.csv"
    src.write_text(f"{HEADER}\nFacebook,Mobile,250,9\n")
    out = tmp_path / "preds.csv"
    r = batch_predict(model_path=str(trained), file_path=str(src), output_path=str(out))
    assert r["unmapped_categories"] == {}
