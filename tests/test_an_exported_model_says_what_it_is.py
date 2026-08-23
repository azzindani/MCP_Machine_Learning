"""An exported model file must describe the format it is actually in.

Every model this server writes is HMAC-signed -- a 32-byte SHA-256 prefix ahead
of the pickle bytes -- because unpickling a caller-supplied path is a
remote-code-execution vector. That is the right call and it is not changing.

What was wrong is that nothing said so. export_model's docstring advertised
"format: pickle", and the file it hands you fails plain pickle.load with
`KeyError: 170`. A coverage sweep hit this, tried both pickle and joblib on
every .pkl in the shared directory, and wrote it up as a "service-internal
storage format" it could not identify -- which is exactly the position anyone
receiving an exported model is in.

The bytes are unchanged. The manifest that travels beside the file, the tool's
response, and its docstring now say what the file is and how to read it.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "ad_data_full.csv"


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> tuple[str, str]:
    import shutil

    from servers.ml_basic.engine import train_regressor

    d = tmp_path_factory.mktemp("export")
    csv = d / "data.csv"
    shutil.copy2(FIXTURE, csv)
    r = train_regressor(str(csv), "spends", "rfr")
    assert r["success"] is True, r.get("error")
    return r["model_path"], str(d)


class TestTheExportSaysHowToReadIt:
    def test_the_response_names_the_format(self, trained):
        from servers.ml_advanced.engine import export_model

        model_path, out_dir = trained
        r = export_model(model_path, output_dir=str(Path(out_dir) / "exported"))
        assert r["success"] is True, r.get("error")
        assert r["file_format"] == "hmac-signed-pickle"
        assert "32" in r["how_to_load"]

    def test_the_manifest_names_the_format(self, trained):
        from servers.ml_advanced.engine import export_model

        model_path, out_dir = trained
        r = export_model(model_path, output_dir=str(Path(out_dir) / "exported2"))
        manifest = json.loads(Path(r["manifest_path"]).read_text(encoding="utf-8"))
        assert manifest["file_format"] == "hmac-signed-pickle"
        assert manifest["signature_bytes"] == 32
        assert "pickle.loads" in manifest["how_to_load"]

    def test_the_documented_recipe_actually_works(self, trained):
        """The one assertion that matters: follow the manifest, get a model."""
        from servers.ml_advanced.engine import export_model

        model_path, out_dir = trained
        r = export_model(model_path, output_dir=str(Path(out_dir) / "exported3"))
        raw = Path(r["model_path"]).read_bytes()

        with pytest.raises(Exception):
            pickle.loads(raw)  # what a recipient tries first, and why they need the note

        payload = pickle.loads(raw[32:])
        assert payload is not None
        blob = payload if isinstance(payload, dict) else {"model": payload}
        assert any(hasattr(v, "predict") for v in blob.values()), list(blob)

    def test_the_exported_bytes_are_unchanged(self, trained):
        import hashlib

        from servers.ml_advanced.engine import export_model

        model_path, out_dir = trained
        r = export_model(model_path, output_dir=str(Path(out_dir) / "exported4"))
        src = hashlib.md5(Path(model_path).read_bytes()).hexdigest()
        dst = hashlib.md5(Path(r["model_path"]).read_bytes()).hexdigest()
        assert src == dst, "export must be a faithful copy"

    def test_the_server_still_loads_its_own_export(self, trained):
        from servers.ml_advanced.engine import export_model, read_model_report

        model_path, out_dir = trained
        r = export_model(model_path, output_dir=str(Path(out_dir) / "exported5"))
        back = read_model_report(r["model_path"])
        assert back["success"] is True, back.get("error")


class TestTheDocstringDoesNotPromisePlainPickle:
    def test_it_mentions_the_signature(self):
        from servers.ml_advanced import server

        # fastmcp v2 wraps the function in a FunctionTool; the docstring lives on .fn
        fn = getattr(server.export_model, "fn", server.export_model)
        doc = fn.__doc__ or ""
        assert "signature" in doc.lower(), doc
        assert len(doc.strip()) <= 80, f"{len(doc.strip())} chars: {doc}"
