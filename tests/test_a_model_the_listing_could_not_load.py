"""list_models advertised four models every loading tool refused.

From the round-14 sweep, phase 6, against the deployed server:

    list_models()                 model_count: 4, each with path, metrics,
                                  trained_on, target_column
    get_predictions(<any of them>)
                                  "Model file signature is invalid —
                                   refusing to unpickle."

All four had been written by this same server. Trained models are HMAC-signed
with a server-local key and verified before unpickling -- the check is right,
and `pickle.load` on a caller-supplied path is exactly the vector it defends
against. What was wrong is where the key lived: `~/.mcp_ml_signing_key`, inside
the container image. A rebuild issued a brand new key, and every model already
saved to the *mounted* output volume stopped verifying, with a message that
reads as tampering for files the server wrote itself a day earlier.

Three parts, and only the first is code that tests can reach:

* `list_models` now verifies each file it lists. A catalogue whose entries the
  sibling tools refuse is a list of dead ends, and checking costs one HMAC over
  bytes already read.
* The refusal names the likely cause -- signed with a different key, so written
  elsewhere or before the key changed -- and what to do about it.
* `MCP_ML_SIGNING_KEY_FILE` moves the key onto the persisted volume, so a
  rebuild stops orphaning every model. That part lives in docker-compose.yml.

Two smaller things from the same phase are pinned here too. `predict_single`
refuses a missing feature by name and dropped an *unknown* key without a word,
so a typo that happened not to collide with a real feature name looked like it
had been used. And list_models' docstring said "Empty = ~/.mcp_models" while
the code deliberately reads MCP_OUTPUT_DIR first -- the sweep found its models
in /workspace/data, which the docstring said was not where it would look.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servers.ml_basic import engine  # noqa: E402
from shared.model_signing import (  # noqa: E402
    ModelIntegrityError,
    dump_signed,
    is_signed_by_us,
    load_signed,
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """An output dir holding one real model and one signed by someone else."""
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_ML_SIGNING_KEY_FILE", str(tmp_path / "keys" / "signing.key"))
    rng = np.random.default_rng(0)
    csv = tmp_path / "t.csv"
    pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120), "y": rng.integers(0, 2, 120)}).to_csv(
        csv, index=False
    )
    r = engine.train_classifier(str(csv), "y", "lr", output_path=str(tmp_path / "mine.pkl"))
    assert r["success"] is True, r.get("error")
    # A model written under a key this server no longer has -- what a rebuilt
    # container sees when it reads the volume it wrote yesterday.
    (tmp_path / "stale.pkl").write_bytes(b"\x00" * 32 + b"pickled under another key")
    return tmp_path


class TestTheListingSaysWhatItCanLoad:
    def test_a_model_this_server_wrote_is_loadable(self, workspace):
        rows = {m["name"]: m for m in engine.list_models()["models"]}
        assert rows["mine.pkl"]["loadable"] is True

    def test_a_model_signed_elsewhere_is_not(self, workspace):
        rows = {m["name"]: m for m in engine.list_models()["models"]}
        assert rows["stale.pkl"]["loadable"] is False

    def test_the_counts_are_both_reported(self, workspace):
        r = engine.list_models()
        assert r["model_count"] == 2
        assert r["loadable_count"] == 1

    def test_the_unloadable_ones_are_named(self, workspace):
        r = engine.list_models()
        assert r["unloadable"] == ["stale.pkl"]

    def test_the_hint_names_the_tools_that_will_fail(self, workspace):
        hint = engine.list_models()["hint"]
        assert "stale.pkl" in hint
        for tool in ("get_predictions", "predict_single", "evaluate_model"):
            assert tool in hint

    def test_a_warning_is_logged_not_only_a_field(self, workspace):
        warnings = [p for p in engine.list_models()["progress"] if p.get("icon") not in ("✔", "ℹ")]
        assert any("cannot be loaded" in p["msg"] for p in warnings)

    def test_a_clean_listing_says_nothing_extra(self, workspace):
        (workspace / "stale.pkl").unlink()
        r = engine.list_models()
        assert r["loadable_count"] == r["model_count"] == 1
        assert "unloadable" not in r
        assert "hint" not in r

    def test_the_listing_and_the_loader_now_agree(self, workspace):
        """The whole point: what it advertises is what will load."""
        for m in engine.list_models()["models"]:
            loaded_ok = True
            try:
                with open(m["path"], "rb") as fh:
                    load_signed(fh)
            except ModelIntegrityError, Exception:
                loaded_ok = False
            assert m["loadable"] is loaded_ok, m["name"]


class TestTheRefusalExplainsItself:
    def test_it_names_the_likely_cause(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCP_ML_SIGNING_KEY_FILE", str(tmp_path / "k.key"))
        f = tmp_path / "x.pkl"
        f.write_bytes(b"\x01" * 32 + b"body")
        with pytest.raises(ModelIntegrityError) as exc:
            with open(f, "rb") as fh:
                load_signed(fh)
        assert "different key" in str(exc.value)
        assert "retrain" in str(exc.value).lower()

    def test_an_unsigned_file_is_still_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCP_ML_SIGNING_KEY_FILE", str(tmp_path / "k.key"))
        f = tmp_path / "short.pkl"
        f.write_bytes(b"tiny")
        with pytest.raises(ModelIntegrityError):
            with open(f, "rb") as fh:
                load_signed(fh)


class TestTheKeyCanLiveSomewhereThatSurvives:
    def test_the_env_var_moves_it(self, tmp_path, monkeypatch):
        target = tmp_path / "vol" / "signing.key"
        monkeypatch.setenv("MCP_ML_SIGNING_KEY_FILE", str(target))
        f = tmp_path / "m.pkl"
        with open(f, "wb") as fh:
            dump_signed({"hello": 1}, fh)
        assert target.exists(), "the key was written where it was told to go"
        assert is_signed_by_us(f) is True

    def test_a_model_survives_a_home_directory_that_does_not(self, tmp_path, monkeypatch):
        """The rebuild case: same key file, different home."""
        keyfile = tmp_path / "vol" / "signing.key"
        monkeypatch.setenv("MCP_ML_SIGNING_KEY_FILE", str(keyfile))
        f = tmp_path / "m.pkl"
        with open(f, "wb") as fh:
            dump_signed({"hello": 1}, fh)

        monkeypatch.setattr(Path, "home", lambda: tmp_path / "a-brand-new-home")
        with open(f, "rb") as fh:
            assert load_signed(fh) == {"hello": 1}

    def test_without_the_env_var_it_falls_back_to_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MCP_ML_SIGNING_KEY_FILE", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        f = tmp_path / "m.pkl"
        with open(f, "wb") as fh:
            dump_signed({"hello": 1}, fh)
        assert (tmp_path / ".mcp_ml_signing_key").exists()


class TestPredictSingleSaysWhatItIgnored:
    @pytest.fixture
    def model(self, workspace):
        return str(workspace / "mine.pkl")

    def _record(self):
        return {"a": 0.5, "b": -0.2}

    def test_a_clean_record_reports_nothing_ignored(self, model):
        r = engine.predict_single(model, self._record())
        assert r["success"] is True, r.get("error")
        assert r["ignored_fields"] == []

    def test_an_unknown_field_is_named(self, model):
        r = engine.predict_single(model, {**self._record(), "bogus_extra_col": 999})
        assert r["success"] is True, r.get("error")
        assert r["ignored_fields"] == ["bogus_extra_col"]

    def test_it_warns_rather_than_only_recording(self, model):
        r = engine.predict_single(model, {**self._record(), "bogus_extra_col": 999})
        warnings = [p for p in r["progress"] if p.get("icon") not in ("✔", "ℹ")]
        assert any("not features of this model" in p["msg"] for p in warnings)

    def test_the_prediction_is_unaffected(self, model):
        clean = engine.predict_single(model, self._record())
        noisy = engine.predict_single(model, {**self._record(), "junk": "x"})
        assert clean["prediction"] == noisy["prediction"]

    def test_a_missing_feature_is_still_refused_loudly(self, model):
        r = engine.predict_single(model, {"a": 0.5})
        assert r["success"] is False
        assert "missing features: b" in r["error"]
