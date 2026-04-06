"""ml_basic training functions — train_classifier and train_regressor."""

from __future__ import annotations

from ._basic_helpers import (
    ALLOWED_CLASSIFIERS,
    ALLOWED_REGRESSORS,
    MIN_ROWS_CLASSIFIER,
    MIN_ROWS_REGRESSOR,
    MODELS_DIR,
    SVC,
    UTC,
    Any,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GaussianNB,
    KNeighborsClassifier,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PolynomialFeatures,
    RandomForestClassifier,
    RandomForestRegressor,
    Ridge,
    StandardScaler,
    _auto_preprocess,
    _check_memory,
    _confusion_dict,
    _error,
    _save_model,
    accuracy_score,
    append_receipt,
    datetime,
    f1_score,
    logger,
    mean_squared_error,
    np,
    ok,
    pd,
    pname,
    r2_score,
    resolve_path,
    sklearn,
    snapshot,
    sys,
    train_test_split,
    xgb,
)


# ---------------------------------------------------------------------------
# 5. train_classifier
# ---------------------------------------------------------------------------
def train_classifier(
    file_path: str,
    target_column: str,
    model: str,
    test_size: float = 0.2,
    random_state: int = 42,
    class_weight: str = "",
    return_train_score: bool = False,
    dry_run: bool = False,
) -> dict:
    """Train classifier on CSV. model: lr svm rf dtc knn nb xgb."""
    progress: list[dict] = []
    backup: str | None = None
    try:
        # --- validation ---
        path = resolve_path(file_path, (".csv",))
        if not path.exists():
            return _error(
                f"File not found: {file_path}",
                "Check that file_path is absolute and the CSV file exists.",
            )

        model = model.strip().lower()
        if model not in ALLOWED_CLASSIFIERS:
            return _error(
                f"Unknown algorithm: '{model}'. Allowed: {', '.join(sorted(ALLOWED_CLASSIFIERS))}",
                "Use one of: lr svm rf dtc knn nb xgb",
            )

        df_raw = pd.read_csv(path, low_memory=False)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df_raw):,} rows × {len(df_raw.columns)} cols"))

        # RAM check
        required_gb = df_raw.memory_usage(deep=True).sum() / 1e9 * 3
        mem_err = _check_memory(required_gb)
        if mem_err:
            return mem_err

        if target_column not in df_raw.columns:
            return _error(
                f"Column '{target_column}' not found. Available: {', '.join(list(df_raw.columns)[:10])}",
                "Use inspect_dataset() to list all column names.",
            )

        df, encoding_map, encoded_cols = _auto_preprocess(df_raw, target_column)
        if encoded_cols:
            progress.append(ok(f"Encoded {len(encoded_cols)} categorical columns", "LabelEncoder"))

        if len(df) < MIN_ROWS_CLASSIFIER:
            return _error(
                f"Dataset has only {len(df)} rows. Need at least {MIN_ROWS_CLASSIFIER}.",
                "Provide a dataset with more samples before training.",
            )

        n_classes = df[target_column].nunique()
        if n_classes < 2:
            return _error(
                f"Target column '{target_column}' has only {n_classes} unique value — cannot train classifier.",
                "Choose a column with at least 2 distinct class values.",
            )

        feature_cols = [c for c in df.columns if c != target_column]
        x = df[feature_cols].values
        y = df[target_column].values

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "op": "train_classifier",
                "model": model,
                "target_column": target_column,
                "feature_columns": feature_cols,
                "row_count": len(df),
                "would_train": True,
                "progress": progress,
                "token_estimate": 80,
            }

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )
        progress.append(ok("Split dataset", f"{len(x_train):,} train / {len(x_test):,} test (stratified)"))

        # --- model training ---
        scaler: StandardScaler | None = None
        model_class_name = ""

        cw = class_weight if class_weight in ("balanced",) else None

        if model == "lr":
            clf = LogisticRegression(random_state=42, max_iter=200, class_weight=cw)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            model_class_name = "LogisticRegression"
            trained: Any = clf

        elif model == "svm":
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)
            clf = SVC(kernel="rbf", gamma="auto", random_state=42, class_weight=cw, probability=True)
            clf.fit(x_train_s, y_train)
            y_pred = clf.predict(x_test_s)
            model_class_name = "SVC"
            trained = clf

        elif model == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=cw)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            model_class_name = "RandomForestClassifier"
            trained = clf

        elif model == "dtc":
            clf = DecisionTreeClassifier(random_state=42, class_weight=cw)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            model_class_name = "DecisionTreeClassifier"
            trained = clf

        elif model == "knn":
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)
            clf = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
            clf.fit(x_train_s, y_train)
            y_pred = clf.predict(x_test_s)
            model_class_name = "KNeighborsClassifier"
            trained = clf

        elif model == "nb":
            clf = GaussianNB()
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            model_class_name = "GaussianNB"
            trained = clf

        else:  # xgb
            nc = int(n_classes)
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtest = xgb.DMatrix(x_test, label=y_test)
            params: dict = {
                "max_depth": 3,
                "eta": 0.3,
                "verbosity": 0,
                "objective": "multi:softprob" if nc > 2 else "binary:logistic",
            }
            if nc > 2:
                params["num_class"] = nc
            xgb_model = xgb.train(params, dtrain, num_boost_round=10)
            preds = xgb_model.predict(dtest)
            if nc > 2:
                y_pred = np.asarray([np.argmax(line) for line in preds])
            else:
                y_pred = (preds > 0.5).astype(int)
            model_class_name = "XGBClassifier"
            trained = xgb_model

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = _confusion_dict(y_test, y_pred)
        metrics: dict = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4), "confusion_matrix": cm}

        # AUC-ROC for binary classification
        if int(n_classes) == 2:
            try:
                from sklearn.metrics import roc_auc_score

                if hasattr(trained, "predict_proba"):
                    y_prob = trained.predict_proba(x_test_s if model in ("svm", "knn") else x_test)[:, 1]
                    metrics["auc_roc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
            except Exception:
                pass

        # Train score for overfit diagnosis
        if return_train_score:
            y_train_pred = (
                trained.predict(x_train_s if model in ("svm", "knn") else x_train)
                if model != "xgb"
                else (
                    np.asarray([np.argmax(row) for row in trained.predict(xgb.DMatrix(x_train))])
                    if int(n_classes) > 2
                    else (trained.predict(xgb.DMatrix(x_train)) > 0.5).astype(int)
                )
            )
            train_acc = float(accuracy_score(y_train, y_train_pred))
            train_f1 = float(f1_score(y_train, y_train_pred, average="weighted", zero_division=0))
            metrics["train_accuracy"] = round(train_acc, 4)
            metrics["train_f1_weighted"] = round(train_f1, 4)
            metrics["overfit_gap"] = round(train_acc - acc, 4)

        progress.append(ok(f"Trained {model_class_name}", f"accuracy={acc:.3f}, f1={f1:.3f}"))

        # --- save model ---
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        models_dir = path.parent / MODELS_DIR
        models_dir.mkdir(exist_ok=True)
        model_filename = f"{path.stem}_{model}_{ts}.pkl"
        model_path = models_dir / model_filename

        # snapshot if overwriting
        if model_path.exists():
            backup = snapshot(str(model_path))

        metadata: dict = {
            "model_type": model_class_name,
            "task": "classification",
            "model_key": model,
            "trained_on": path.name,
            "training_date": datetime.now(UTC).isoformat(),
            "feature_columns": feature_cols,
            "target_column": target_column,
            "encoding_map": encoding_map,
            "scaler": scaler,
            "metrics": metrics,
            "n_classes": int(n_classes),
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
        }
        _save_model(trained, model_path, metadata)
        progress.append(ok("Saved model", pname(str(model_path))))

        append_receipt(
            file_path,
            "train_classifier",
            {"target": target_column, "model": model},
            f"accuracy={acc:.3f}",
            backup,
        )

        response: dict = {
            "success": True,
            "op": "train_classifier",
            "model": model,
            "model_class": model_class_name,
            "task": "classification",
            "target_column": target_column,
            "feature_columns": feature_cols,
            "row_count": len(df),
            "train_size": len(x_train),
            "test_size": len(x_test),
            "metrics": metrics,
            "model_path": str(model_path),
            "backup": backup or "",
            "progress": progress,
        }
        response["token_estimate"] = len(str(response)) // 4
        return response

    except Exception as exc:
        logger.debug("train_classifier error: %s", exc)
        return _error(
            str(exc),
            "Use inspect_dataset() and read_column_profile() to verify your data first.",
            backup,
        )


# ---------------------------------------------------------------------------
# 6. train_regressor
# ---------------------------------------------------------------------------
def train_regressor(
    file_path: str,
    target_column: str,
    model: str,
    degree: int = 5,
    alpha: float = 0.01,
    n_estimators: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train regressor on CSV. model: lir pr lar rr dtr rfr xgb."""
    progress: list[dict] = []
    backup: str | None = None
    try:
        path = resolve_path(file_path, (".csv",))
        if not path.exists():
            return _error(
                f"File not found: {file_path}",
                "Check that file_path is absolute and the CSV file exists.",
            )

        model = model.strip().lower()
        if model not in ALLOWED_REGRESSORS:
            return _error(
                f"Unknown algorithm: '{model}'. Allowed: {', '.join(sorted(ALLOWED_REGRESSORS))}",
                "Use one of: lir pr lar rr dtr rfr xgb",
            )

        df_raw = pd.read_csv(path, low_memory=False)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df_raw):,} rows × {len(df_raw.columns)} cols"))

        required_gb = df_raw.memory_usage(deep=True).sum() / 1e9 * 3
        mem_err = _check_memory(required_gb)
        if mem_err:
            return mem_err

        if target_column not in df_raw.columns:
            return _error(
                f"Column '{target_column}' not found. Available: {', '.join(list(df_raw.columns)[:10])}",
                "Use inspect_dataset() to list all column names.",
            )

        df, encoding_map, encoded_cols = _auto_preprocess(df_raw, target_column)
        if encoded_cols:
            progress.append(ok(f"Encoded {len(encoded_cols)} categorical columns", "LabelEncoder"))

        if len(df) < MIN_ROWS_REGRESSOR:
            return _error(
                f"Dataset has only {len(df)} rows. Need at least {MIN_ROWS_REGRESSOR}.",
                "Provide a dataset with more samples before training.",
            )

        feature_cols = [c for c in df.columns if c != target_column]
        x = df[feature_cols].values.astype(float)
        y = df[target_column].values.astype(float)

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "op": "train_regressor",
                "model": model,
                "target_column": target_column,
                "feature_columns": feature_cols,
                "row_count": len(df),
                "would_train": True,
                "progress": progress,
                "token_estimate": 80,
            }

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        progress.append(ok("Split dataset", f"{len(x_train):,} train / {len(x_test):,} test"))

        poly: PolynomialFeatures | None = None
        model_class_name = ""
        scaler: StandardScaler | None = None

        if model == "lir":
            reg = LinearRegression()
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            model_class_name = "LinearRegression"
            trained: Any = reg

        elif model == "pr":
            poly = PolynomialFeatures(degree=degree)
            x_train_p = poly.fit_transform(x_train)
            x_test_p = poly.transform(x_test)
            reg = LinearRegression()
            reg.fit(x_train_p, y_train)
            y_pred = reg.predict(x_test_p)
            model_class_name = "PolynomialRegression"
            trained = reg

        elif model == "lar":
            reg = Lasso(alpha=alpha, max_iter=200, tol=0.1)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            model_class_name = "Lasso"
            trained = reg

        elif model == "rr":
            reg = Ridge(alpha=alpha, max_iter=100, tol=0.1)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            model_class_name = "Ridge"
            trained = reg

        elif model == "dtr":
            reg = DecisionTreeRegressor(random_state=42)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            model_class_name = "DecisionTreeRegressor"
            trained = reg

        elif model == "rfr":
            reg = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            model_class_name = "RandomForestRegressor"
            trained = reg

        else:  # xgb
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtest = xgb.DMatrix(x_test, label=y_test)
            params = {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.3, "verbosity": 0}
            xgb_model = xgb.train(params, dtrain, num_boost_round=5)
            y_pred = xgb_model.predict(dtest)
            model_class_name = "XGBRegressor"
            trained = xgb_model

        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        metrics = {"mse": round(mse, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}
        progress.append(ok(f"Trained {model_class_name}", f"r2={r2:.3f}, rmse={rmse:.2f}"))

        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        models_dir = path.parent / MODELS_DIR
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{path.stem}_{model}_{ts}.pkl"

        if model_path.exists():
            backup = snapshot(str(model_path))

        metadata: dict = {
            "model_type": model_class_name,
            "task": "regression",
            "model_key": model,
            "trained_on": path.name,
            "training_date": datetime.now(UTC).isoformat(),
            "feature_columns": feature_cols,
            "target_column": target_column,
            "encoding_map": encoding_map,
            "poly": poly,
            "scaler": scaler,
            "metrics": metrics,
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
        }
        _save_model(trained, model_path, metadata)
        progress.append(ok("Saved model", pname(str(model_path))))

        append_receipt(
            file_path,
            "train_regressor",
            {"target": target_column, "model": model},
            f"r2={r2:.3f}",
            backup,
        )

        response: dict = {
            "success": True,
            "op": "train_regressor",
            "model": model,
            "model_class": model_class_name,
            "task": "regression",
            "target_column": target_column,
            "feature_columns": feature_cols,
            "row_count": len(df),
            "train_size": len(x_train),
            "test_size": len(x_test),
            "metrics": metrics,
            "model_path": str(model_path),
            "backup": backup or "",
            "progress": progress,
        }
        response["token_estimate"] = len(str(response)) // 4
        return response

    except Exception as exc:
        logger.debug("train_regressor error: %s", exc)
        return _error(
            str(exc),
            "Use inspect_dataset() and read_column_profile() to verify your data first.",
            backup,
        )
