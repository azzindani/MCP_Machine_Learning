"""One file's snapshots must not be offered as another file's history.

Snapshots were named `{stem}_{timestamp}.bak`, with the extension dropped, and
looked up with `glob(f"{stem}_*.bak")`. Two consequences, both reachable from
the deployed endpoints with ordinary filenames:

  * `report.csv` and `report.docx` in one directory share a history. Calling
    restore_version on the CSV with no timestamp restored the newest snapshot
    under that stem -- the Word document -- and answered success: true. A
    12-byte CSV came back as 37,117 bytes of .docx. Here it is worse than in
    the sibling repos: a model .pkl and its source .csv routinely sit in the
    same directory, so restoring a dataset could hand back a pickle.
  * `Ad_Data_test.csv`'s snapshots answered a query about `Ad_Data.csv`,
    because `Ad_Data_*` matches both.

File_System already wrote `{stem}_{ts}{ext}.bak`; this is the sibling half of
that fix. Reading stays deliberately more forgiving than writing -- a snapshot
taken before this change is still listed and still restorable -- but only where
the old name cannot be ambiguous.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.version_control import list_snapshots, restore_version, snapshot  # noqa: E402


def names(file_path: Path) -> list[str]:
    return [Path(s["path"]).name for s in list_snapshots(str(file_path))]


class TestTheExtensionIsPartOfTheName:
    def test_two_namesakes_do_not_share_a_history(self, tmp_path):
        csv = tmp_path / "report.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")
        pkl = tmp_path / "report.pkl"
        pkl.write_bytes(b"\x80\x04" + b"x" * 400)

        snapshot(str(csv))
        snapshot(str(pkl))

        assert len(names(csv)) == 1, names(csv)
        assert len(names(pkl)) == 1, names(pkl)
        assert not set(names(csv)) & set(names(pkl))

    def test_restoring_a_csv_does_not_hand_back_a_pickle(self, tmp_path):
        csv = tmp_path / "report.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")
        snapshot(str(csv))
        time.sleep(0.01)
        pkl = tmp_path / "report.pkl"
        pkl.write_bytes(b"\x80\x04" + b"x" * 400)
        snapshot(str(pkl))

        csv.write_text("edited\n", encoding="utf-8")
        newest = list_snapshots(str(csv))[0]["timestamp"]
        r = restore_version(str(csv), newest)
        assert r["success"] is True, r.get("error")
        assert csv.read_text(encoding="utf-8") == "a,b\n1,2\n"

    def test_a_longer_name_is_not_a_version_of_a_shorter_one(self, tmp_path):
        base = tmp_path / "Ad_Data.csv"
        base.write_text("a\n1\n", encoding="utf-8")
        other = tmp_path / "Ad_Data_test.csv"
        other.write_text("a\n2\n", encoding="utf-8")
        snapshot(str(other))
        assert names(base) == []
        assert len(names(other)) == 1


class TestOlderSnapshotsAreStillReachable:
    def test_a_legacy_name_is_listed_when_nothing_shares_the_stem(self, tmp_path):
        csv = tmp_path / "solo.csv"
        csv.write_text("a\n1\n", encoding="utf-8")
        versions = tmp_path / ".mcp_versions"
        versions.mkdir()
        legacy = versions / "solo_2026-08-01T00-00-00-000000Z.bak"
        legacy.write_text("a\n0\n", encoding="utf-8")
        assert names(csv) == [legacy.name]

    def test_a_legacy_snapshot_still_restores(self, tmp_path):
        csv = tmp_path / "solo.csv"
        csv.write_text("current\n", encoding="utf-8")
        versions = tmp_path / ".mcp_versions"
        versions.mkdir()
        (versions / "solo_2026-08-01T00-00-00-000000Z.bak").write_text("older\n", encoding="utf-8")
        r = restore_version(str(csv), "2026-08-01T00-00-00-000000Z")
        assert r["success"] is True, r.get("error")
        assert csv.read_text(encoding="utf-8") == "older\n"

    def test_a_legacy_name_is_withheld_when_a_namesake_exists(self, tmp_path):
        csv = tmp_path / "shared.csv"
        csv.write_text("a\n1\n", encoding="utf-8")
        (tmp_path / "shared.pkl").write_bytes(b"\x80\x04")
        versions = tmp_path / ".mcp_versions"
        versions.mkdir()
        (versions / "shared_2026-08-01T00-00-00-000000Z.bak").write_text("?", encoding="utf-8")
        # Ambiguous: it could be either file's. Better to show nothing than to
        # restore a pickle over a dataset.
        assert names(csv) == []


class TestSnapshotsStillWork:
    def test_a_snapshot_round_trips(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("original\n", encoding="utf-8")
        backup = snapshot(str(f))
        assert Path(backup).read_text(encoding="utf-8") == "original\n"
        assert Path(backup).name.endswith(".csv.bak")

    def test_the_timestamp_survives_the_extension(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("x\n", encoding="utf-8")
        snapshot(str(f))
        ts = list_snapshots(str(f))[0]["timestamp"]
        assert "." not in ts and ts.startswith("20"), ts

    def test_a_file_with_no_extension_still_snapshots(self, tmp_path):
        f = tmp_path / "README"
        f.write_text("x\n", encoding="utf-8")
        backup = snapshot(str(f))
        assert Path(backup).exists()
        assert names(f) == [Path(backup).name]
