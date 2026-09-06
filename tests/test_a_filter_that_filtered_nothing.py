"""`search_columns(dtype="float64")` answered with every column in the frame.

    Search columns by condition. Returns names only, no data.

The second half held. The first did not: `dtype` was compared against four
literal group names -- numeric, categorical, bool, datetime -- in an if/elif
chain with no else, so any other value matched no branch, filtered nothing, and
the whole frame came back under `success: true`. Measured on `Ad_Data.csv`:

    dtype="float64"  -> 16 of 16 columns, including Date, product, phase
    dtype="object"   -> 16 of 16 columns
    has_nulls=True   ->  1 column, correct

`float64` is not an exotic input. It is the string `inspect_dataset` prints in
its own `dtype` field, so it is exactly what a caller reads off one tool and
hands to the next.

**The sibling is what settles it.** MCP_Data_Analyst exposes `search_columns`
with the same name and the same claim, and answers `dtype="float64"` with the
four numeric columns -- because that repo hit this bug first and fixed it. Two
identically-described tools disagreed, both said `success: true`, and nothing
in either response told the caller which one they were holding.

So the alias table is ported here, an unlisted value is refused with a hint
naming the vocabulary, and an alias that widens the filter says so -- `float64`
means `numeric`, which also matches integer columns, and a count that quietly
includes them would disagree with the word the caller typed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from servers.ml_basic.engine import DTYPE_FILTER_ALIASES, DTYPE_FILTERS, search_columns


@pytest.fixture()
def frame(tmp_path):
    """Mixed dtypes, named the way a real file is."""
    csv = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=6).astype(str),
            "product": ["a", "b", "c", "d", "e", "f"],
            "spends": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "clicks": [1, 2, 3, 4, 5, 6],
            "flagged": [True, False, True, False, True, False],
        }
    ).to_csv(csv, index=False)
    return str(csv)


class TestTheFilterActuallyFilters:
    def test_a_pandas_dtype_name_no_longer_returns_the_whole_frame(self, frame):
        out = search_columns(frame, dtype="float64")
        assert out["success"] is True, out.get("error")
        assert out["columns"] != []
        assert "product" not in out["columns"]
        assert "spends" in out["columns"]

    def test_object_does_not_return_the_numerics(self, frame):
        out = search_columns(frame, dtype="object")
        assert out["success"] is True, out.get("error")
        assert "spends" not in out["columns"]
        assert "product" in out["columns"]

    def test_the_group_names_still_work(self, frame):
        out = search_columns(frame, dtype="numeric")
        assert out["success"] is True, out.get("error")
        assert set(out["columns"]) >= {"spends", "clicks"}
        assert "product" not in out["columns"]

    def test_bool_is_its_own_group_here(self, frame):
        out = search_columns(frame, dtype="bool")
        assert out["columns"] == ["flagged"]

    def test_the_condition_that_always_worked_still_does(self, frame):
        out = search_columns(frame, has_nulls=True)
        assert out["success"] is True
        assert out["columns"] == []

    def test_no_dtype_means_no_dtype_filtering(self, frame):
        out = search_columns(frame)
        assert len(out["columns"]) == 5


class TestAnUnlistedValueIsRefusedRatherThanIgnored:
    def test_a_nonsense_dtype_is_refused(self, frame):
        out = search_columns(frame, dtype="banana")
        assert out["success"] is False
        assert "banana" in out["error"]

    def test_and_the_hint_names_the_vocabulary(self, frame):
        hint = search_columns(frame, dtype="banana")["hint"]
        for group in DTYPE_FILTERS:
            assert group in hint

    def test_the_hint_also_says_pandas_names_are_accepted(self, frame):
        hint = search_columns(frame, dtype="banana")["hint"]
        assert "float64" in hint


class TestAWidenedFilterSaysSo:
    def test_float64_warns_that_it_filtered_by_numeric(self, frame):
        out = search_columns(frame, dtype="float64")
        assert any("numeric" in str(p) for p in out["progress"])

    def test_an_exact_group_name_carries_no_such_warning(self, frame):
        out = search_columns(frame, dtype="numeric")
        assert not [p for p in out["progress"] if "not 'numeric'" in str(p)]


class TestTheTwoReposAgreeOnWhatTheyAccept:
    @pytest.mark.parametrize("alias", sorted(DTYPE_FILTER_ALIASES))
    def test_every_alias_resolves_to_a_real_group(self, alias):
        assert DTYPE_FILTER_ALIASES[alias] in DTYPE_FILTERS

    # `bool` is a group this tier has and the sibling does not -- Data_Analyst
    # sorts booleans into numeric or object and offers three groups where this
    # one offers four. That difference is deliberate and is why the comparison
    # below is over the pandas names both repos can answer, not over all of them.
    TIER_ONLY = {"boolean"}

    def test_the_sibling_accepts_every_pandas_name_this_one_does(self):
        """Same tool name, same description, so the same call must not 404."""
        import pathlib

        sibling = pathlib.Path("/root/MCP_Data_Analyst/servers/data_basic/engine.py")
        if not sibling.exists():
            pytest.skip("sibling repo not present in this checkout")
        text = sibling.read_text(encoding="utf-8")
        for alias in set(DTYPE_FILTER_ALIASES) - self.TIER_ONLY:
            assert f'"{alias}"' in text, f"{alias} is accepted here and unknown to the sibling"

    def test_the_one_difference_is_the_one_we_documented(self):
        """A new divergence should fail here rather than surprise a caller."""
        assert self.TIER_ONLY <= set(DTYPE_FILTER_ALIASES)
        assert all(DTYPE_FILTER_ALIASES[a] == "bool" for a in self.TIER_ONLY)

    def test_the_description_names_the_vocabulary(self):
        assert "numeric" in (search_columns.__doc__ or "")
