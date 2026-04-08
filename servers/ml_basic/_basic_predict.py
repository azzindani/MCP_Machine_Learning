"""ml_basic predict/utility functions — get_predictions, restore_version, etc."""

from __future__ import annotations

import json as _json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from shared.file_utils import get_output_dir, resolve_path
from shared.platform_utils import get_max_rows
from shared.progress import info, ok
from shared.progress import name as pname
from shared.receipt import append_receipt
from shared.version_control import restore_version as _restore_version
from shared.version_control import snapshot

from ._basic_helpers import (
    Path,
    _error,
    _load_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 7. get_predictions
# ---------------------------------------------------------------------------
def get_predictions(
    model_path: str,
    file_path: str,
    max_rows: int = 20,
    return_proba: bool = False,
) -> dict:
    """Run predictions with saved model. Returns bounded prediction list."""
    progress: list[dict] = []
    try:
        mpath = Path(model_path).resolve()
        if not mpath.exists():
            return _error(
                f"Model file not found: {model_path}",
                "Use train_classifier() or train_regressor() to train a model first.",
            )

        model_obj, metadata = _load_model(model_path)
        progress.append(ok(f"Loaded model {pname(model_path)}", metadata.get("model_type", "")))

        data_path = resolve_path(file_path, (".csv",))
        if not data_path.exists():
            return _error(
                f"File not found: {file_path}",
                "Check that file_path is absolute and the CSV file exists.",
            )

        df = pd.read_csv(data_path, low_memory=False)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df):,} rows"))

        # apply same encoding as training
        encoding_map: dict = metadata.get("encoding_map", {})
        for col, mapping in encoding_map.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map({str(k): v for k, v in mapping.items()}).fillna(-1).astype(int)

        feature_cols: list[str] = metadata.get("feature_columns", [])
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return _error(
                f"Feature columns missing in data: {', '.join(missing)}",
                "Ensure the data file has the same columns used during training.",
            )

        x = df[feature_cols].fillna(0).values.astype(float)

        # apply scaler if used
        sc = metadata.get("scaler")
        if sc is not None:
            x = sc.transform(x)

        # apply poly if used
        poly = metadata.get("poly")
        if poly is not None:
            x = poly.transform(x)

        cap = min(max_rows, get_max_rows())
        x_slice = x[:cap]
        truncated = len(x) > cap

        task = metadata.get("task", "classification")
        model_key = metadata.get("model_key", "")

        proba_list: list | None = None
        if model_key == "xgb":
            dmat = xgb.DMatrix(x_slice)
            raw_preds = model_obj.predict(dmat)
            if task == "classification":
                n_classes = metadata.get("n_classes", 2)
                if n_classes > 2:
                    preds_list = [int(np.argmax(line)) for line in raw_preds]
                    if return_proba:
                        proba_list = [raw_preds[i].tolist() for i in range(len(raw_preds))]
                else:
                    preds_list = [int(p > 0.5) for p in raw_preds]
                    if return_proba:
                        proba_list = [[round(1 - float(p), 4), round(float(p), 4)] for p in raw_preds]
            else:
                preds_list = [float(p) for p in raw_preds]
        else:
            raw = model_obj.predict(x_slice)
            if task == "classification":
                preds_list = [int(p) for p in raw]
                if return_proba and hasattr(model_obj, "predict_proba"):
                    proba_raw = model_obj.predict_proba(x_slice)
                    proba_list = [[round(float(v), 4) for v in row] for row in proba_raw]
            else:
                preds_list = [float(p) for p in raw]

        predictions = [{"row": i, "prediction": preds_list[i]} for i in range(len(preds_list))]
        if proba_list:
            for i, entry in enumerate(predictions):
                entry["probabilities"] = proba_list[i]
        progress.append(ok("Generated predictions", f"{len(predictions)} rows"))

        response: dict = {
            "success": True,
            "op": "get_predictions",
            "model_path": pname(model_path),
            "task": task,
            "predictions": predictions,
            "returned": len(predictions),
            "total_rows": len(x),
            "truncated": truncated,
            "progress": progress,
        }
        if truncated:
            response["hint"] = f"Results capped at {cap}. Pass max_rows to increase (up to {get_max_rows()})."
        response["token_estimate"] = len(str(response)) // 4
        return response

    except Exception as exc:
        logger.debug("get_predictions error: %s", exc)
        return _error(
            str(exc),
            "Check model_path and that the data file matches the training schema.",
        )


# ---------------------------------------------------------------------------
# 8. restore_version
# ---------------------------------------------------------------------------
def restore_version(file_path: str, timestamp: str = "") -> dict:
    """Restore file/model to previous snapshot. Empty timestamp = list."""
    progress: list[dict] = []
    try:
        result = _restore_version(file_path, timestamp)
        if result.get("success"):
            if timestamp:
                progress.append(ok(f"Restored {pname(file_path)}", result.get("restored_from", "")))
            else:
                progress.append(
                    info(
                        f"Listed snapshots for {pname(file_path)}",
                        f"{len(result.get('snapshots', []))} available",
                    )
                )
        result["progress"] = progress
        result["token_estimate"] = len(str(result)) // 4
        return result
    except Exception as exc:
        logger.debug("restore_version error: %s", exc)
        return _error(
            str(exc),
            "Check that the file path is correct and snapshots exist in .mcp_versions/.",
        )


# ---------------------------------------------------------------------------
# predict_single
# ---------------------------------------------------------------------------
def predict_single(model_path: str, input_data: str) -> dict:
    """Predict on one record from a JSON dict string. No CSV required."""
    progress: list[dict] = []
    try:
        mp = resolve_path(model_path)
    except ValueError as exc:
        return _error(str(exc), "Check that model_path is inside your home directory.")
    if not mp.exists():
        return _error(f"Model file not found: {model_path}", "Train a model first.")

    # Parse input JSON
    try:
        if isinstance(input_data, dict):
            record = input_data
        else:
            record = _json.loads(input_data)
    except Exception as exc:
        return _error(
            f"Invalid input_data JSON: {exc}",
            'Pass a JSON string like: {"age": 30, "tenure": 5}',
        )

    try:
        model, metadata = _load_model(str(mp))
    except Exception as exc:
        return _error(f"Failed to load model: {exc}", "Check model_path is a valid .pkl file.")

    feature_columns: list[str] = metadata.get("feature_columns", [])
    if not feature_columns:
        return _error(
            "Model has no feature_columns in metadata.",
            "Retrain the model to regenerate metadata.",
        )

    missing = [c for c in feature_columns if c not in record]
    if missing:
        return _error(
            f"Input missing features: {', '.join(missing)}",
            f"Provide all features: {', '.join(feature_columns)}",
        )

    # Build single-row DataFrame and apply encoding
    row_df = pd.DataFrame([{c: record[c] for c in feature_columns}])
    encoding_map: dict = metadata.get("encoding_map", {})
    for col, mapping in encoding_map.items():
        if col in row_df.columns:
            row_df[col] = row_df[col].map(mapping).fillna(-1)
    # Fill missing numerics
    for col in row_df.select_dtypes(include="number").columns:
        if bool(row_df[col].isnull().any()):
            row_df[col] = row_df[col].fillna(0)

    x = row_df.values
    task = metadata.get("task", "classification")
    scaler = metadata.get("scaler")
    if scaler is not None:
        x = scaler.transform(x)

    try:
        if hasattr(model, "predict") and not (
            hasattr(model, "predict_proba") is False and type(model).__name__ == "Booster"
        ):
            prediction = model.predict(x)[0]
            prob = None
            if task == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)[0]
                prob = {str(i): round(float(p), 4) for i, p in enumerate(proba)}
        else:
            # XGBoost Booster
            dmat = xgb.DMatrix(x)
            raw = model.predict(dmat)
            n_classes = int(metadata.get("n_classes", 2))
            if n_classes > 2:
                prediction = int(np.argmax(raw[0]))
                prob = {str(i): round(float(p), 4) for i, p in enumerate(raw[0])}
            else:
                prediction = int(raw[0] > 0.5)
                prob = {"0": round(float(1 - raw[0]), 4), "1": round(float(raw[0]), 4)}
    except Exception as exc:
        return _error(
            f"Prediction failed: {exc}",
            "Check input values match training data types.",
        )

    progress.append(ok("Loaded model", mp.name))
    progress.append(ok("Predicted", f"result={prediction}"))

    resp: dict = {
        "success": True,
        "op": "predict_single",
        "model_path": str(mp),
        "task": task,
        "input": record,
        "prediction": int(prediction) if task == "classification" else float(prediction),
        "probabilities": prob,
        "feature_columns": feature_columns,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------
def list_models(directory: str = "") -> dict:
    """List all saved .pkl models with their metadata summaries."""
    progress: list[dict] = []
    if directory:
        try:
            search_dir = resolve_path(directory)
        except ValueError as exc:
            return _error(str(exc), "Check that directory is inside your home directory.")
    else:
        search_dir = get_output_dir()

    models: list[dict] = []
    for pkl in sorted(search_dir.glob("*.pkl")):
        if ".mcp_versions" in str(pkl):
            continue
        manifest = pkl.with_suffix(".manifest.json")
        entry: dict = {
            "path": str(pkl),
            "name": pkl.name,
            "size_kb": round(pkl.stat().st_size / 1024, 1),
            "modified": pkl.stat().st_mtime,
        }
        if manifest.exists():
            try:
                meta = _json.loads(manifest.read_text())
                entry.update(
                    {
                        "model_type": meta.get("model_type", ""),
                        "task": meta.get("task", ""),
                        "trained_on": meta.get("trained_on", ""),
                        "training_date": meta.get("training_date", "")[:10],
                        "target_column": meta.get("target_column", ""),
                        "metrics": meta.get("metrics", {}),
                    }
                )
            except Exception:
                pass
        models.append(entry)

    progress.append(ok("Scanned for models", f"{len(models)} found"))

    resp: dict = {
        "success": True,
        "op": "list_models",
        "directory": str(search_dir),
        "model_count": len(models),
        "models": models,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------
def split_dataset(
    file_path: str,
    test_size: float = 0.2,
    stratify_column: str = "",
    output_dir: str = "",
    random_state: int = 42,
) -> dict:
    """Split CSV into train/test files. Saves both to disk."""
    from sklearn.model_selection import train_test_split as _tts

    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check the file path.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv, got {path.suffix!r}", "Provide a CSV file.")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is valid.")
    progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df):,} rows"))

    if test_size <= 0 or test_size >= 1:
        return _error(
            f"test_size={test_size} must be in (0, 1).",
            "Use e.g. test_size=0.2 for 20% test.",
        )

    out_dir = resolve_path(output_dir) if output_dir else get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    stratify = None
    if stratify_column:
        if stratify_column not in df.columns:
            return _error(
                f"Column '{stratify_column}' not found.",
                "Use inspect_dataset() to see column names.",
            )
        stratify = df[stratify_column]

    df_train, df_test = _tts(df, test_size=test_size, random_state=random_state, stratify=stratify)

    train_path = out_dir / f"{path.stem}_train.csv"
    test_path = out_dir / f"{path.stem}_test.csv"

    # Snapshot if they already exist
    backup_train = backup_test = ""
    for p_check, var_name in [(train_path, "train"), (test_path, "test")]:
        if p_check.exists():
            try:
                b = snapshot(str(p_check))
                if var_name == "train":
                    backup_train = b
                else:
                    backup_test = b
            except Exception:
                pass

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    progress.append(ok("Split complete", f"{len(df_train):,} train / {len(df_test):,} test"))
    progress.append(ok("Saved train", train_path.name))
    progress.append(ok("Saved test", test_path.name))

    append_receipt(
        file_path,
        "split_dataset",
        {"test_size": test_size, "stratify_column": stratify_column},
        "success",
        backup_train or backup_test,
    )

    resp: dict = {
        "success": True,
        "op": "split_dataset",
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows": len(df_train),
        "test_rows": len(df_test),
        "test_size_actual": round(len(df_test) / len(df), 4),
        "stratified": bool(stratify_column),
        "backup_train": backup_train,
        "backup_test": backup_test,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp
