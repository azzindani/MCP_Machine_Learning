"""New targeted tests to append to test_ml_shared.py"""

SHARED_ADDITIONS = r'''

# ===========================================================================
# shared/file_utils.py — encoding fallback (skip enc==encoding, latin-1 last resort)
# ===========================================================================


class TestFileUtilsEncodingEdgeCases:
    """Lines 90, 95, 99-103 of shared/file_utils.py."""

    def test_encoding_skip_when_enc_equals_encoding(self, tmp_path):
        """Line 90: fallback loop skips enc when enc == encoding parameter."""
        from shared.file_utils import read_csv

        # File with \xe9 byte (fails utf-8 and utf-8-sig, but passes cp1252 and latin-1)
        # Call with encoding="utf-8-sig" — the loop will skip "utf-8-sig" via line 90
        p = tmp_path / "enc.csv"
        p.write_bytes(b"name,value\ncaf\xe9,1\n")
        df = read_csv(str(p), encoding="utf-8-sig")
        assert "name" in df.columns

    def test_latin1_last_resort_fallback(self, tmp_path):
        """Line 95: all fallbacks fail, latin-1 is the last resort."""
        from shared.file_utils import read_csv

        # Byte 0x81 is undefined in cp1252 but valid in latin-1
        p = tmp_path / "latin.csv"
        p.write_bytes(b"name,value\nx\x81z,1\n")
        df = read_csv(str(p))
        assert "name" in df.columns
        assert len(df) >= 1

    def test_bad_lines_skip_on_tokenization_error(self, tmp_path):
        """Lines 99-103: CSV with mismatched field counts triggers on_bad_lines=skip."""
        from shared.file_utils import read_csv

        # Create CSV with an extra field in one row — causes ParserError (field count)
        p = tmp_path / "bad_lines.csv"
        p.write_bytes(b"a,b\n1,2\n3,4,5,6,7\n5,6\n")
        df = read_csv(str(p))
        assert "a" in df.columns
        assert len(df) >= 1  # at least the good rows are read


class TestAtomicWriteOslinkFailure:
    """Lines 144-145 of shared/file_utils.py: os.unlink also fails during cleanup."""

    def test_unlink_oserror_swallowed_during_cleanup(self, tmp_path):
        """Lines 144-145: os.unlink raises OSError but is silenced; original exc re-raised."""
        import pytest
        from unittest.mock import patch
        from shared.file_utils import atomic_write

        target = tmp_path / "out.bin"
        with patch("shutil.move", side_effect=OSError("move failed")):
            with patch("os.unlink", side_effect=OSError("unlink also failed")):
                with pytest.raises(OSError, match="move failed"):
                    atomic_write(target, b"data")
'''
