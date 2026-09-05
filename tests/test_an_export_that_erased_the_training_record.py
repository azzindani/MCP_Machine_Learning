"""Exporting a model must not destroy how it was trained.

`export_model(model_path)` with no `output_dir` writes beside the source, under
the source's own name — so `manifest_dst` *is* the training manifest. It used to
replace that file wholesale with an export descriptor, and the snapshot guard
above it skips the same-path case, so there was not even a backup.

What a single export deleted: `split` (how the score was produced — the thing a
user review asked for by name, so a 0.9628 could be read as a claim),
`encoding_map_path` (which orphans the split-out map, leaving
`read_model_report` unable to find it), `feature_defaults`, `hyperparameters`,
`leakage_warning`, `n_classes`, `scaler` and `model_key`.

Nothing reported it. The export returned `success: true`, the manifest it left
behind was valid JSON with plausible contents, and the loss only showed up if
you knew which keys had been there a moment earlier. That is the same shape as
every other defect this round: an operation that is honest about what it did and
silent about what it undid.

**An export descriptor is extra information about a file, not a replacement for
its provenance.** So the training metadata is carried through and the export
fields are layered on top.

Found by a smoke assertion in CI, and only after that assertion was fixed to
print the manifest's actual keys instead of guessing that a missing key meant a
missing feature.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from servers.ml_advanced.engine import export_model
from servers.ml_basic.engine import train_classifier
from shared.model_output import encoding_map_path

# Keys a training run records that an export has no business removing. Named
# rather than inferred, so deleting one from a trainer fails here too.
TRAINING_KEYS = (
    "split",
    "feature_defaults",
    "hyperparameters",
    "leakage_warning",
    "n_classes",
    "scaler",
    "model_key",
    "encoding_map",
)


@pytest.fixture()
def trained(tmp_path: Path):
    random.seed(11)
    csv = tmp_path / "dataset.csv"
    rows = ["f1,f2,grade,label"]
    for _ in range(150):
        rows.append(f"{random.random():.4f},{random.random():.4f},{random.choice('ABC')},{random.choice([0, 1])}")
    csv.write_text("\n".join(rows), encoding="utf-8")

    model = tmp_path / "m.pkl"
    result = train_classifier(str(csv), "label", model="rf", output_path=str(model))
    assert result["success"] is True, result.get("error")
    manifest = model.with_suffix(".manifest.json")
    assert manifest.is_file()
    return model, manifest


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheTrainingManifestSurvivesAnInPlaceExport:
    def test_the_fixture_actually_records_training_keys(self, trained):
        """Without this the assertions below could pass on an empty manifest."""
        _model, manifest = trained
        before = read(manifest)
        present = [k for k in TRAINING_KEYS if k in before]
        assert "split" in present
        assert len(present) >= 4, sorted(before)

    def test_nothing_is_lost(self, trained):
        model, manifest = trained
        before = read(manifest)
        result = export_model(str(model))
        assert result["success"] is True, result.get("error")

        after = read(manifest)
        lost = sorted(set(before) - set(after))
        assert not lost, f"export deleted {lost} from the training manifest"

    def test_split_provenance_specifically_survives(self, trained):
        """The review's ask: a score on disk has to say how it was produced."""
        model, manifest = trained
        before = read(manifest)["split"]
        export_model(str(model))
        assert read(manifest)["split"] == before

    def test_the_export_fields_are_added(self, trained):
        model, manifest = trained
        export_model(str(model))
        after = read(manifest)
        assert after["file_format"] == "hmac-signed-pickle"
        assert after["signature_bytes"] == 32
        assert "how_to_load" in after

    def test_it_records_that_it_was_exported(self, trained):
        model, manifest = trained
        export_model(str(model))
        after = read(manifest)
        assert after["exported_from"] == str(model)
        assert after["exported_at"]

    def test_exporting_twice_is_still_lossless(self, trained):
        model, manifest = trained
        before = read(manifest)
        export_model(str(model))
        export_model(str(model))
        assert not set(before) - set(read(manifest))


class TestExportingElsewhereTakesWhatItNeeds:
    def test_the_copy_carries_the_training_record(self, trained, tmp_path):
        model, _manifest = trained
        out = tmp_path / "shipped"
        result = export_model(str(model), output_dir=str(out))
        assert result["success"] is True, result.get("error")
        copied = read(Path(result["manifest_path"]))
        assert "split" in copied
        assert copied["exported_from"] == str(model)

    def test_the_source_manifest_is_untouched(self, trained, tmp_path):
        model, manifest = trained
        before = read(manifest)
        export_model(str(model), output_dir=str(tmp_path / "elsewhere"))
        assert read(manifest) == before

    def test_a_split_out_encoding_map_travels_with_it(self, trained, tmp_path):
        """A manifest naming a sidecar that did not travel is a broken export."""
        model, manifest = trained
        # Force the split shape regardless of how small the fixture's map is.
        big = {f"title_{i}": i for i in range(500)}
        encoding_map_path(model).write_text(json.dumps({"emp_title": big}), encoding="utf-8")
        data = read(manifest)
        data["encoding_map_path"] = str(encoding_map_path(model))
        manifest.write_text(json.dumps(data), encoding="utf-8")

        out = tmp_path / "shipped"
        result = export_model(str(model), output_dir=str(out))
        assert result["success"] is True, result.get("error")
        assert encoding_map_path(Path(result["model_path"])).is_file(), (
            "the manifest points at an encoding map that was left behind"
        )

    def test_a_model_with_no_sidecar_exports_fine(self, trained, tmp_path):
        model, _manifest = trained
        assert not encoding_map_path(model).exists()
        result = export_model(str(model), output_dir=str(tmp_path / "plain"))
        assert result["success"] is True, result.get("error")
