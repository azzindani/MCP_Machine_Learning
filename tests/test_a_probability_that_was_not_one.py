"""The fourth thing the review asked the manifest to record.

    AGI: split `manifest.json` (KBs) + `encoding_map.parquet` (MBs); auto-flag
    "leakage_suspects"; add split/seed/CV/calibration for reproducibility.

Three of those four shipped. `calibration` did not, and its absence is the
quietest of the set: every classifier here exposes `predict_proba`, nothing
calibrates it, and a caller reading 0.8 as "an 80% chance" is wrong in a way no
field in the response contradicts. Tree ensembles are the usual offenders --
a random forest's averaged votes are systematically pushed toward the middle,
and an SVM's decision function is not a probability at all until something maps
it to one.

So `split` now carries `calibration: "none"` rather than omitting the key. An
absent key reads as "not applicable"; the truth is that it *is* applicable and
the answer is none. The note beside it says what that costs the reader, and the
day a trainer calibrates it passes `"sigmoid"` or `"isotonic"` and the note
disappears on its own.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from servers.ml_basic.engine import train_classifier
from shared.leakage import split_provenance


class TestTheFieldItself:
    def test_an_uncalibrated_split_says_none_rather_than_nothing(self):
        out = split_provenance(test_size=0.2, random_state=42)
        assert out["calibration"] == "none"

    def test_and_explains_what_that_costs_the_reader(self):
        note = split_provenance(test_size=0.2, random_state=42)["calibration_note"]
        assert "not probabilities" in note or "uncalibrated" in note

    def test_a_calibrated_split_says_which_method(self):
        out = split_provenance(test_size=0.2, random_state=42, calibration="isotonic")
        assert out["calibration"] == "isotonic"

    def test_and_then_carries_no_caveat(self):
        """The note is about the absence, so it goes when the absence does."""
        assert "calibration_note" not in split_provenance(test_size=0.2, random_state=42, calibration="sigmoid")

    def test_the_other_three_are_untouched(self):
        out = split_provenance(test_size=0.25, random_state=7, stratified=True, cv_folds=5)
        assert out["test_size"] == 0.25
        assert out["random_state"] == 7
        assert out["stratified"] is True
        assert out["cv_folds"] == 5


class TestItReachesTheManifestThatOutlivesTheResponse:
    @pytest.fixture()
    def trained(self, tmp_path: Path):
        random.seed(5)
        rows = ["f1,f2,label"]
        for _ in range(150):
            rows.append(f"{random.random():.4f},{random.random():.4f},{random.choice([0, 1])}")
        csv = tmp_path / "d.csv"
        csv.write_text("\n".join(rows), encoding="utf-8")
        model = tmp_path / "m.pkl"
        result = train_classifier(str(csv), "label", model="rf", output_path=str(model))
        assert result["success"] is True, result.get("error")
        return model.with_suffix(".manifest.json")

    def test_the_manifest_records_it(self, trained):
        split = json.loads(trained.read_text(encoding="utf-8"))["split"]
        assert split["calibration"] == "none"

    def test_beside_the_split_fields_it_already_had(self, trained):
        split = json.loads(trained.read_text(encoding="utf-8"))["split"]
        for key in ("test_size", "random_state", "stratified", "cv_folds"):
            assert key in split, split


class TestOneModuleAcrossTheFleet:
    def test_the_data_analyst_copy_matches(self):
        """It is one file in two repos; changing one and not the other is the bug."""
        import hashlib

        ours = Path(__file__).resolve().parents[1] / "shared" / "leakage.py"
        sibling = Path("/root/MCP_Data_Analyst/shared/leakage.py")
        if not sibling.exists():
            pytest.skip("sibling repo not present in this checkout")
        assert hashlib.sha256(ours.read_bytes()).hexdigest() == hashlib.sha256(sibling.read_bytes()).hexdigest()
