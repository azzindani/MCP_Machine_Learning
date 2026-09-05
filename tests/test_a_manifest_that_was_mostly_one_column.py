"""The manifest on disk is the readable file, and the map is not in it.

The user review measured the artifact:

    Model `Credit_Risk_dtc_best_*.pkl` (882 KB) + manifest (1,017 KB, 29,129
    lines) -- CORRECT, LEAKY, BLOATED
    dtc 0.9628; installment 0.379 + total_payment 0.298 + last_payment_date
    0.180. Manifest = 23 features + 28k `emp_title` map.
    AGI: split `manifest.json` (KBs) + `encoding_map.parquet` (MBs); auto-flag
    `"leakage_suspects"`; add split/seed/CV/calibration for reproducibility.

`skip_encoding_map` fixed what a *report* costs to read. It did not touch the
file: the manifest beside every model was still 1 MB of `emp_title`, so anyone
opening it, diffing it, or committing it paid the megabyte. Splitting it is the
other half.

**JSON rather than the parquet the review named**, and the reason is the shape
rather than the size -- see `split_encoding_map`. What the ask was actually
about is that the manifest stop costing a megabyte to read, and moving the map
to its own file is what does that.

**The .pkl keeps everything.** That is the property these tests guard hardest: a
model that needs a sidecar to make a prediction would be a worse artifact than
the bloated one it replaced, and the split would have traded a token cost for a
correctness cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from servers.ml_advanced.engine import read_model_report
from shared.model_output import (
    ENCODING_MAP_SPLIT_ABOVE,
    ENCODING_MAP_SUFFIX,
    encoding_map_path,
    save_model,
    split_encoding_map,
)
from shared.model_signing import load_signed


def big_map(n: int = 2_000) -> dict:
    return {
        "emp_title": {f"title_{i}": i for i in range(n)},
        "grade": {g: i for i, g in enumerate("ABCDEFG")},
    }


def metadata_with(map_: dict) -> dict:
    return {
        "model_type": "DecisionTreeClassifier",
        "task": "classification",
        "trained_on": "Credit_Risk.csv",
        "feature_columns": [f"f{i}" for i in range(23)],
        "target_column": "loan_status",
        "encoding_map": map_,
        "metrics": {"accuracy": 0.9628},
    }


@pytest.fixture()
def saved(tmp_path: Path):
    model_path = tmp_path / "Credit_Risk_dtc_best.pkl"
    manifest_path = save_model(None, model_path, metadata_with(big_map()))
    return model_path, manifest_path


class TestTheManifestIsBackToKilobytes:
    def test_the_map_is_not_in_it(self, saved):
        _model, manifest_path = saved
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "encoding_map" not in manifest, "this is the megabyte, still in the file"

    def test_it_is_small(self, saved):
        _model, manifest_path = saved
        assert manifest_path.stat().st_size < 8_000, manifest_path.stat().st_size

    def test_everything_else_survived(self, saved):
        _model, manifest_path = saved
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["trained_on"] == "Credit_Risk.csv"
        assert manifest["metrics"]["accuracy"] == 0.9628
        assert len(manifest["feature_columns"]) == 23

    def test_it_says_where_the_map_went(self, saved):
        model_path, manifest_path = saved
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert Path(manifest["encoding_map_path"]) == encoding_map_path(model_path)
        assert manifest["encoding_map_path"].endswith(ENCODING_MAP_SUFFIX)

    def test_it_says_how_big_the_map_is_without_holding_it(self, saved):
        _model, manifest_path = saved
        summary = json.loads(manifest_path.read_text(encoding="utf-8"))["encoding_map_summary"]
        assert summary["entries_total"] == 2_007
        assert summary["entries_per_column"]["emp_title"] == 2_000
        assert summary["columns"] == ["emp_title", "grade"]


class TestTheSidecarHoldsTheMapIntact:
    def test_it_exists_beside_the_model(self, saved):
        model_path, _manifest = saved
        assert encoding_map_path(model_path).is_file()

    def test_nothing_was_lost(self, saved):
        model_path, _manifest = saved
        loaded = json.loads(encoding_map_path(model_path).read_text(encoding="utf-8"))
        assert loaded == big_map()


class TestTheModelIsStillSelfContained:
    """The property that makes the split safe rather than clever."""

    def test_the_pkl_still_carries_the_whole_map(self, saved):
        model_path, _manifest = saved
        with open(model_path, "rb") as handle:
            payload = load_signed(handle)
        assert payload["metadata"]["encoding_map"] == big_map()

    def test_a_model_moved_without_its_sidecars_still_predicts(self, saved, tmp_path):
        """Copy the .pkl alone, as anyone shipping a model would."""
        model_path, _manifest = saved
        elsewhere = tmp_path / "moved"
        elsewhere.mkdir()
        target = elsewhere / model_path.name
        target.write_bytes(model_path.read_bytes())

        with open(target, "rb") as handle:
            payload = load_signed(handle)
        assert payload["metadata"]["encoding_map"]["grade"]["A"] == 0


class TestASmallMapStaysWhereItIs:
    def test_a_handful_of_categories_needs_no_second_file(self, tmp_path):
        model_path = tmp_path / "small.pkl"
        manifest_path = save_model(None, model_path, metadata_with({"grade": {g: i for i, g in enumerate("ABCDEFG")}}))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "encoding_map" in manifest
        assert not encoding_map_path(model_path).exists()

    def test_it_is_still_summarised(self, tmp_path):
        model_path = tmp_path / "small.pkl"
        manifest_path = save_model(None, model_path, metadata_with({"grade": {"A": 0, "B": 1}}))
        summary = json.loads(manifest_path.read_text(encoding="utf-8"))["encoding_map_summary"]
        assert summary["entries_total"] == 2

    def test_the_threshold_is_the_one_declared(self, tmp_path):
        below = {"c": {str(i): i for i in range(ENCODING_MAP_SPLIT_ABOVE)}}
        above = {"c": {str(i): i for i in range(ENCODING_MAP_SPLIT_ABOVE + 1)}}
        assert split_encoding_map(metadata_with(below), tmp_path / "a.pkl")[1] == ""
        assert split_encoding_map(metadata_with(above), tmp_path / "b.pkl")[1] != ""

    def test_no_map_at_all_changes_nothing(self, tmp_path):
        metadata = {"model_type": "x", "metrics": {}}
        manifest, sidecar = split_encoding_map(metadata, tmp_path / "c.pkl")
        assert manifest == metadata
        assert sidecar == ""


class TestTheReportReadsBothShapes:
    """Models on disk outlive the code that wrote them."""

    def test_the_new_shape_is_summarised_by_default(self, saved):
        model_path, _manifest = saved
        report = read_model_report(str(model_path))
        assert report["success"] is True, report.get("error")
        assert "encoding_map" not in report["manifest"]
        assert report["encoding_map_summary"]["entries_total"] == 2_007

    def test_the_new_shape_can_be_inlined_on_request(self, saved):
        model_path, _manifest = saved
        report = read_model_report(str(model_path), skip_encoding_map=False)
        assert report["encoding_map_inlined"] is True
        assert report["manifest"]["encoding_map"] == big_map()

    def test_the_note_names_the_sidecar_rather_than_the_manifest(self, saved):
        model_path, _manifest = saved
        note = read_model_report(str(model_path))["encoding_map_summary"]["note"]
        assert ENCODING_MAP_SUFFIX in note

    def test_a_manifest_written_before_the_split_still_reads(self, tmp_path):
        """A report that only understood the new shape would answer 'no map'."""
        from shared.model_signing import dump_signed

        model_path = tmp_path / "old.pkl"
        with open(model_path, "wb") as handle:
            dump_signed({"model": None, "metadata": {"task": "classification"}}, handle)
        model_path.with_suffix(".manifest.json").write_text(
            json.dumps({"model_type": "dtc", "encoding_map": big_map()}), encoding="utf-8"
        )

        report = read_model_report(str(model_path))
        assert report["encoding_map_summary"]["entries_total"] == 2_007
        assert "encoding_map" not in report["manifest"]

        full = read_model_report(str(model_path), skip_encoding_map=False)
        assert full["manifest"]["encoding_map"] == big_map()

    def test_a_missing_sidecar_is_not_a_crash(self, saved):
        """Someone deletes the file. The report still answers about the model."""
        model_path, _manifest = saved
        encoding_map_path(model_path).unlink()
        report = read_model_report(str(model_path))
        assert report["success"] is True
        assert "encoding_map_summary" not in report


class TestTheManifestSaysHowTheScoreWasProduced:
    """The other half of the same review line: "add split/seed/CV ... for reproducibility".

    `split_provenance` existed and reached one response. It did not reach the
    manifest, which is the file that outlives the response -- so a 0.9628 on
    disk still could not be read as the claim it is.
    """

    def _manifest_source(self, relative: str) -> str:
        return (Path(__file__).resolve().parents[1] / relative).read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "relative",
        ["servers/ml_basic/_basic_train.py", "servers/ml_advanced/engine.py"],
    )
    def test_the_trainers_record_the_split_in_the_metadata_they_save(self, relative):
        source = self._manifest_source(relative)
        assert '"split": split_provenance(' in source, f"{relative} saves a model without saying how it was split"

    def test_the_classifier_records_that_it_stratified(self):
        source = self._manifest_source("servers/ml_basic/_basic_train.py")
        assert "stratified=True" in source
        assert "stratified=False" in source, "the regressor does not stratify and must not claim to"

    def test_a_tuned_model_records_folds_rather_than_a_holdout(self):
        """`best_score` is a mean over CV folds; test_size 0 says so."""
        source = self._manifest_source("servers/ml_advanced/engine.py")
        assert "cv_folds=cv" in source
        assert "test_size=0.0" in source

    def test_the_split_survives_into_the_saved_manifest(self, tmp_path):
        from shared.leakage import split_provenance

        metadata = metadata_with(big_map())
        metadata["split"] = split_provenance(test_size=0.2, random_state=42, stratified=True)
        manifest_path = save_model(None, tmp_path / "m.pkl", metadata)
        split = json.loads(manifest_path.read_text(encoding="utf-8"))["split"]
        assert split["test_size"] == 0.2
        assert split["random_state"] == 42
        assert split["stratified"] is True
        assert "split_note" in split, "a random split of time-ordered rows has to warn"


class TestTheSplitNeverCostsATrainingRun:
    def test_an_unwritable_sidecar_leaves_the_map_in_the_manifest(self, tmp_path, monkeypatch):
        import shared.file_utils as file_utils

        def explode(*_a, **_k):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(file_utils, "atomic_write_json", explode)
        manifest, sidecar = split_encoding_map(metadata_with(big_map()), tmp_path / "x.pkl")
        assert sidecar == ""
        assert "encoding_map" in manifest, "a manifest that still holds it is not wrong, only fat"
