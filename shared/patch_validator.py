"""Validate preprocessing op arrays before applying any operation.

validate_ops() checks the entire array first. If any op is invalid,
no ops are applied — fail fast, fail clean.

Empty ops list is valid (no-op pass-through).
"""

ALLOWED_PREPROCESSING_OPS: set[str] = {
    "fill_nulls",
    "drop_outliers",
    "label_encode",
    "onehot_encode",
    "scale",
    "drop_duplicates",
    "drop_column",
    "rename_column",
    "convert_dtype",
    "bin_numeric",
    "add_date_parts",
    "log_transform",
    "drop_null_rows",
    "clip_column",
}

# Required fields per op type
_REQUIRED_FIELDS: dict[str, list[str]] = {
    "fill_nulls": ["column", "strategy"],
    "drop_outliers": ["column", "method"],
    "label_encode": ["column"],
    "onehot_encode": ["column"],
    "scale": ["columns", "method"],
    "drop_duplicates": [],
    "drop_column": ["column"],
    "rename_column": ["from", "to"],
    "convert_dtype": ["column", "to"],
    "bin_numeric": ["column"],
    "add_date_parts": ["column"],
    "log_transform": ["column"],
    "drop_null_rows": [],
    "clip_column": ["column"],
}

ALLOWED_FILL_STRATEGIES: set[str] = {"mean", "median", "mode", "ffill", "bfill", "zero"}
ALLOWED_SCALE_METHODS: set[str] = {"standard", "minmax"}
ALLOWED_DTYPES: set[str] = {"int", "float", "str", "datetime", "bool", "numeric", "string"}
ALLOWED_OUTLIER_METHODS: set[str] = {"iqr", "std"}

MAX_OPS = 50


def validate_ops(
    ops: list[dict],
    allowed: set[str] | None = None,
) -> tuple[bool, str]:
    """Validate op array. Returns (True, '') or (False, error_message).

    Empty ops list is treated as valid (no-op pass-through).
    """
    if allowed is None:
        allowed = ALLOWED_PREPROCESSING_OPS

    if not isinstance(ops, list):
        return False, "ops must be a list of dicts."

    # Empty list is a valid no-op
    if len(ops) == 0:
        return True, ""

    if len(ops) > MAX_OPS:
        return False, f"Too many ops ({len(ops)}). Maximum is {MAX_OPS} per batch."

    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            return False, f"Op {i}: must be a dict, got {type(op).__name__}."

        op_name = op.get("op", "")
        if not op_name:
            return False, f"Op {i}: missing 'op' key."

        if op_name not in allowed:
            return False, (f"Op {i}: unknown op '{op_name}'. Allowed: {', '.join(sorted(allowed))}")

        for field in _REQUIRED_FIELDS.get(op_name, []):
            if field not in op:
                return False, f"Op {i} ('{op_name}'): missing required field '{field}'."

        # enum validation
        if op_name == "fill_nulls":
            strategy = op.get("strategy", "")
            if strategy not in ALLOWED_FILL_STRATEGIES:
                return False, (
                    f"Op {i}: invalid fill_nulls strategy '{strategy}'. "
                    f"Allowed: {', '.join(sorted(ALLOWED_FILL_STRATEGIES))}"
                )

        if op_name == "scale":
            method = op.get("method", "")
            if method not in ALLOWED_SCALE_METHODS:
                return False, (
                    f"Op {i}: invalid scale method '{method}'. Allowed: {', '.join(sorted(ALLOWED_SCALE_METHODS))}"
                )
            if not isinstance(op.get("columns"), list):
                return False, f"Op {i} (scale): 'columns' must be a list of column names."

        if op_name == "drop_outliers":
            method = op.get("method", "")
            if method not in ALLOWED_OUTLIER_METHODS:
                return False, (
                    f"Op {i}: invalid drop_outliers method '{method}'. "
                    f"Allowed: {', '.join(sorted(ALLOWED_OUTLIER_METHODS))}"
                )

        if op_name == "convert_dtype":
            dtype = op.get("to", "")
            if dtype not in ALLOWED_DTYPES:
                return False, (f"Op {i}: invalid dtype '{dtype}'. Allowed: {', '.join(sorted(ALLOWED_DTYPES))}")

    return True, ""
