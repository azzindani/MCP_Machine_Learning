"""A snapshot that holds bytes must not report its size as 0.0.

A coverage sweep checking restore_version listed two real 34-byte .bak files as
`size_kb: 0.0` -- which is exactly what an empty backup looks like, and this is
the number someone decides a restore on. `round(n / 1024, 1)` sends everything
under 51 bytes to zero; generate_eda_report used `// 1024`, so anything under a
kilobyte read as 0.

The sibling File_System repo had the same division feeding its delete
confirmation ("Permanently deletes 1 item(s) (0 KB). Cannot be undone.").
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.version_control import list_snapshots, size_kb, snapshot  # noqa: E402


class TestOnlyAnEmptyFileReportsZero:
    @pytest.mark.parametrize(
        "n_bytes,nonzero",
        [(0, False), (1, True), (34, True), (51, True), (900, True), (1024, True), (5000, True)],
    )
    def test_the_helper(self, n_bytes: int, nonzero: bool):
        assert (size_kb(n_bytes) > 0) is nonzero, n_bytes

    def test_a_kilobyte_still_reads_as_one(self):
        assert size_kb(1024) == 1.0
        assert size_kb(1536) == 1.5
        assert size_kb(1024 * 1024) == 1024.0

    def test_a_negative_or_missing_size_is_zero_not_negative(self):
        assert size_kb(0) == 0.0
        assert size_kb(-5) == 0.0


class TestASmallSnapshotIsListedWithItsSize:
    def test_a_thirty_four_byte_backup_does_not_list_as_zero(self, tmp_path: Path):
        src = tmp_path / "tiny.csv"
        src.write_text("a,b\n1,2\n", encoding="utf-8")  # well under 51 bytes
        assert snapshot(str(src))
        snaps = list_snapshots(str(src))
        assert snaps, "no snapshot listed"
        assert snaps[0]["size_kb"] > 0, snaps[0]

    def test_the_listed_size_matches_the_file(self, tmp_path: Path):
        src = tmp_path / "tiny.csv"
        src.write_text("x" * 2048, encoding="utf-8")
        assert snapshot(str(src))
        snaps = list_snapshots(str(src))
        assert snaps[0]["size_kb"] == pytest.approx(2.0, abs=0.1), snaps[0]
