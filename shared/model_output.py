"""Where a trained model goes, and why the caller must be able to say.

Five tools across three servers save a model, and every one of them built its
own name from a wall-clock timestamp:

    train_classifier       {stem}_{model}_{ts}.pkl
    train_regressor        {stem}_{model}_{ts}.pkl
    train_with_cv          {stem}_{model}_cv_{ts}.pkl
    compare_models         {stem}_{best}_best_{ts}.pkl
    tune_hyperparameters   {stem}_{model}_tuned_{ts}.pkl

So an identical call always landed somewhere new, and none of them took an
output path. Round 11's sweep called each tool twice with byte-identical
arguments and measured the result against the live endpoints:

    train_classifier   two files, 53,812,898 B each
    train_with_cv      two files,  7,098,115 B each
    compare_models     two files,  7,098,045 B each

The metrics matched to the digit each time -- the seeding works -- and the
sweep unpickled a pair to confirm they differed only in the embedded
`training_date`. So a client whose call timed out and re-sent it pays for a
model it already has, and the largest of them is 54 MB a go.
tune_hyperparameters is the sharpest case: the most expensive call on the
fleet, hence the likeliest to time out, hence the likeliest to be retried.

Four sibling tools -- split_dataset, run_clustering, batch_predict and
export_model -- already took an output path. These five, writing much the
largest artifacts, took none.

The timestamped default is unchanged, so nothing relying on it breaks. Given an
output path the caller decides, a retry overwrites, and the previous model is
snapshotted first by the caller's existing collision branch.
"""

from __future__ import annotations

import os
from pathlib import Path

from shared.file_utils import resolve_path


def default_models_dir(source: Path) -> Path:
    """Where models land when the caller names no path."""
    override = os.environ.get("MCP_OUTPUT_DIR", "").strip()
    return Path(override) if override else source.parent / ".mcp_models"


def resolve_model_path(output_path: str, source: Path, default_name: str, progress: list | None = None) -> Path:
    """The path to save a model to, honouring an explicit output_path.

    A missing or wrong extension is corrected rather than refused: `.pkl` is
    what every loader here expects, and list_models finds models by that glob,
    so a model saved as `mine.dat` would be invisible to the tool that lists it.

    Pass `progress` and the caller is told when that happened. It used to be
    silent, so `output_path="cv_lr.json"` wrote cv_lr.pkl with nothing to say
    the name had changed -- and a sweep reading its own arguments back
    reasonably concluded output_path had been ignored altogether. The
    correction is right; not mentioning it is what made it look like a defect.
    The chart tools in the sibling repo have warned about exactly this since an
    earlier round.
    """
    if output_path:
        out = resolve_path(output_path, ())
        if out.suffix.lower() != ".pkl":
            asked = out.name
            out = out.with_suffix(".pkl")
            if progress is not None:
                from shared.progress import warn

                progress.append(
                    warn(
                        "Output extension changed",
                        f"{asked} → {out.name}; models are saved as .pkl so list_models can find them",
                    )
                )
        return out
    return default_models_dir(source) / default_name


# The megabyte, moved out of the readable file. A user review measured it:
# "Model `Credit_Risk_dtc_best_*.pkl` (882 KB) + manifest (1,017 KB, 29,129
# lines) -- CORRECT, LEAKY, BLOATED... AGI: split `manifest.json` (KBs) +
# `encoding_map.parquet` (MBs)".
ENCODING_MAP_SUFFIX = ".encoding_map.json"

# Below this many encoded values the map is not the problem, and a second file
# for eleven entries is noise. `emp_title` alone was 28,000 on the review's file.
ENCODING_MAP_SPLIT_ABOVE = 200


def encoding_map_path(model_path: Path) -> Path:
    """The sidecar that holds a split-out encoding map."""
    return model_path.with_suffix(ENCODING_MAP_SUFFIX)


def split_encoding_map(metadata: dict, model_path: Path) -> tuple[dict, str]:
    """Return (manifest without the big map, sidecar path or "").

    JSON rather than the parquet the review named, and the reason is the shape
    rather than the size. An encoding map is `{column: {value: code}}` -- ragged,
    string-keyed, and read as a lookup. Parquet would need it flattened into
    (column, value, code) triples, which is a re-shape every reader would have to
    undo, in exchange for a dependency this repo does not otherwise carry. The
    ask was that the manifest stop costing a megabyte to read; moving the map to
    its own file is what does that, in either format.

    The .pkl is untouched and still carries the full metadata, so a saved model
    stays usable on new data without its sidecars. That is the split: the model
    is self-contained, the manifest is the part a human or an agent reads.
    """
    raw = metadata.get("encoding_map")
    if not isinstance(raw, dict) or not raw:
        return metadata, ""

    per_column = {col: len(vals) if hasattr(vals, "__len__") else 1 for col, vals in raw.items()}
    total = sum(per_column.values())
    summary = {
        "columns": sorted(per_column),
        "entries_per_column": per_column,
        "entries_total": total,
    }
    if total <= ENCODING_MAP_SPLIT_ABOVE:
        manifest = dict(metadata)
        manifest["encoding_map_summary"] = summary
        return manifest, ""

    from shared.file_utils import atomic_write_json

    sidecar = encoding_map_path(model_path)
    try:
        atomic_write_json(sidecar, raw)
    except OSError:
        # A manifest that still holds the map is worse than one that does not,
        # but it is not wrong -- and a training run must not fail because a
        # sidecar could not be written.
        return metadata, ""

    manifest = {k: v for k, v in metadata.items() if k != "encoding_map"}
    summary["path"] = str(sidecar)
    summary["note"] = (
        f"{total:,} encoded values across {len(per_column)} column(s) live in "
        f"{sidecar.name} rather than here. The .pkl still carries the full map, so the "
        "model works without this file; read_model_report(skip_encoding_map=False) inlines it."
    )
    manifest["encoding_map_summary"] = summary
    manifest["encoding_map_path"] = str(sidecar)
    return manifest, str(sidecar)


def save_model(model_obj: object, path: Path, metadata: dict) -> Path:
    """Write the signed model and its manifest atomically. Returns the manifest.

    There were two of these, one per server, and they had drifted. The basic
    copy created the parent directory and the advanced one did not, so the same
    call was robust from one server and raised FileNotFoundError from the other.
    Neither returned anything, which is why three trainers reported the .pkl
    they wrote and never the .manifest.json beside it -- and read_model_report
    reads the manifest, so a caller who copied the path they were given left
    the report behind.

    One function, in shared, next to the path rules the same tools already
    share. It returns the manifest path so a caller can name both files.

    A large encoding map goes to its own sidecar on the way through -- see
    `split_encoding_map`. The .pkl keeps everything.
    """
    import shutil
    import tempfile

    from shared.exchange import apply_default_mode
    from shared.file_utils import atomic_write_json
    from shared.model_signing import dump_signed

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_obj, "metadata": metadata}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=path.parent) as tmp:
        dump_signed(payload, tmp)
        tmp_path = tmp.name
    apply_default_mode(tmp_path)
    shutil.move(tmp_path, str(path))
    manifest_path = path.with_suffix(".manifest.json")
    manifest, _sidecar = split_encoding_map(metadata, path)
    atomic_write_json(manifest_path, manifest)
    return manifest_path
