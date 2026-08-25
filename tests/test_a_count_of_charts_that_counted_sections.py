"""It said it had drawn six charts, and the file held four.

    generate_eda_report(...)
      -> charts_generated: 6
         the written HTML: 4 plotly-graph-div

The field was `len(sections)`, and a section is not a chart. Of the seven the
report can build, two are tables with no chart in them at all -- "Data Quality"
is metric cards plus an alert list, "Summary Statistics" is a describe() table
-- and one, "Categorical Columns", is a single section holding one chart per
column. So the number was wrong in both directions at once: inflated by the
tables, and flattened wherever the categorical section drew more than one.

Nothing downstream can tell, either. The number is not used to lay the page
out; it is only reported. A caller that trusts it to decide whether the report
is worth opening is told about panels that are not there.

Found by round 15's axis -- open the file and count what is in it, rather than
read the count back off the reply that claimed it. Fourteen rounds of judging
these tools by their replies never surfaced it, because the reply is
self-consistent: it is the file that disagrees.

The sibling that does this correctly is data_advanced's generate_distribution_plot,
whose `chart_count: n * 2` really is the panel count, and ml_advanced's
run_profiling_report, which reports the same quantity honestly as
`sections_generated`. That name is kept here too, so the old number is still
available under the thing it actually measures.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from servers.ml_medium import engine as mm

# One figure embedded by shared.html_theme.plotly_div carries exactly one of
# these. Counting the marker counts the charts, however many a section holds.
_PLOT_DIV = re.compile(r'class="[^"]*\bplotly-graph-div\b')


def charts_in(html_path: Path) -> int:
    return len(_PLOT_DIV.findall(html_path.read_text(encoding="utf-8", errors="ignore")))


def report(source: Path, out: Path, **kw) -> dict:
    r = mm.generate_eda_report(str(source), output_path=str(out), open_after=False, **kw)
    assert r["success"] is True, r.get("error")
    return r


@pytest.fixture()
def mixed(tmp_path: Path) -> Path:
    """Numeric columns, two categorical columns, and a column with nulls.

    Chosen so every optional section fires: missing, distributions,
    correlation, and a categorical section holding more than one chart.
    """
    rows = ["a,b,c,region,tier"]
    for i in range(60):
        gap = "" if i % 7 == 0 else str(i * 3)
        rows.append(f"{i},{i * 2},{gap},{'north' if i % 2 else 'south'},{'gold' if i % 3 else 'basic'}")
    src = tmp_path / "mixed.csv"
    src.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return src


class TestTheCountMatchesTheFile:
    def test_it_reports_the_charts_the_file_holds(self, mixed: Path, tmp_path: Path) -> None:
        out = tmp_path / "eda.html"
        r = report(mixed, out)
        assert r["charts_generated"] == charts_in(out), (r["charts_generated"], charts_in(out))

    def test_the_file_holds_some(self, mixed: Path, tmp_path: Path) -> None:
        """A marker that never matched would make the check above pass for free."""
        out = tmp_path / "eda.html"
        report(mixed, out)
        assert charts_in(out) >= 3, charts_in(out)

    def test_a_table_only_section_is_not_counted(self, mixed: Path, tmp_path: Path) -> None:
        """Data Quality and Summary Statistics draw no chart between them."""
        out = tmp_path / "eda.html"
        r = report(mixed, out)
        assert r["charts_generated"] < r["sections_generated"], r

    def test_a_categorical_section_counts_each_of_its_charts(self, mixed: Path, tmp_path: Path) -> None:
        """One section, two columns, two charts -- the old count said one.

        The chart titles are matched loosely: plotly serialises the em dash in
        "region — Top Values" as an escape inside its JSON payload, so the
        readable form is not in the file to grep for.
        """
        out = tmp_path / "eda.html"
        r = report(mixed, out)
        html = out.read_text(encoding="utf-8", errors="ignore")
        assert html.count("Top Values") == 2, "fixture did not build a chart per categorical column"
        assert r["charts_generated"] == charts_in(out)


class TestItTracksWhatTheDataProduces:
    def test_a_frame_with_no_nulls_draws_no_missing_chart(self, tmp_path: Path) -> None:
        clean = tmp_path / "clean.csv"
        clean.write_text(
            "a,b,region\n" + "".join(f"{i},{i * 2},{'north' if i % 2 else 'south'}\n" for i in range(40)),
            encoding="utf-8",
        )
        out = tmp_path / "clean.html"
        r = report(clean, out)
        assert "Missing Values" not in out.read_text(encoding="utf-8", errors="ignore")
        assert r["charts_generated"] == charts_in(out)

    def test_a_target_column_adds_a_section_but_not_a_chart(self, mixed: Path, tmp_path: Path) -> None:
        """The sharpest case for why the two numbers had to be separated.

        `show_cats` excludes the target column, so naming one moves that
        column's chart out of "Categorical Columns" and into a section of its
        own. A section appears; no chart does. The old field, counting
        sections, would have reported one more chart than the file gained.
        """
        without = tmp_path / "without.html"
        with_target = tmp_path / "with.html"
        a = report(mixed, without)
        b = report(mixed, with_target, target_column="region")

        assert b["sections_generated"] == a["sections_generated"] + 1, (a, b)
        assert b["charts_generated"] == a["charts_generated"], (a["charts_generated"], b["charts_generated"])
        assert b["charts_generated"] == charts_in(with_target)
        assert a["charts_generated"] == charts_in(without)


class TestTheOldNumberIsStillAvailable:
    def test_sections_generated_is_reported(self, mixed: Path, tmp_path: Path) -> None:
        r = report(mixed, tmp_path / "eda.html")
        assert isinstance(r["sections_generated"], int)
        assert r["sections_generated"] >= 4, r["sections_generated"]

    def test_it_counts_headings_in_the_page(self, mixed: Path, tmp_path: Path) -> None:
        out = tmp_path / "eda.html"
        r = report(mixed, out)
        html = out.read_text(encoding="utf-8", errors="ignore")
        present = sum(
            1
            for heading in (
                "Data Quality",
                "Missing Values",
                "Numeric Distributions",
                "Correlation",
                "Categorical Columns",
                "Summary Statistics",
            )
            if heading in html
        )
        assert r["sections_generated"] == present, (r["sections_generated"], present)
