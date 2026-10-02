"""query_data: a table too big to load is narrowed with SQL where it lies, then trained on."""

from __future__ import annotations

import logging
from pathlib import Path

from shared import sql_query, sql_remote
from shared.file_utils import resolve_path
from shared.handover import make_context, make_handover
from shared.isolation import memory_budget_mb, worker_threads
from shared.progress import fail, info, ok
from shared.receipt import append_receipt
from shared.sql_query import DEFAULT_PREVIEW, QueryRefused, run_query
from shared.version_control import snapshot

from ._basic_helpers import _error

logger = logging.getLogger("ml_basic")

sql_query.WORKBOOK_ADVICE = "save the sheet as csv or parquet"


def _snapshot_if_exists(path: Path) -> str:
    """Snapshot a file about to be written over; "" when nothing is there or the copy fails."""
    if not path.is_file():
        return ""
    try:
        return snapshot(str(path))
    except Exception:
        return ""


def _missing(path: Path, what: str) -> dict:
    return _error(f"{what} not found: {path.name}", "Check the path is absolute and the file exists.")


def query_data(
    sql: str,
    file_path: str = "",
    tables: dict[str, str] | None = None,
    database: str = "",
    output_path: str = "",
    max_rows: int = DEFAULT_PREVIEW,
    memory_mb: int = 0,
) -> dict:
    progress: list[dict] = []
    try:
        named: dict[str, Path] = {}
        if file_path:
            named["data"] = resolve_path(file_path)
        for name, location in (tables or {}).items():
            if name in named:
                return _error(
                    f"Table name {name!r} is used twice (file_path is the table called `data`).",
                    "Give each table its own name in `tables`.",
                )
            named[name] = resolve_path(location)
        for name, path in named.items():
            if not path.is_file():
                return _missing(path, f"Table {name!r} file")
        if database and "://" in database:
            return _error(
                "A database is named here, not given by URL.",
                "Ask the operator to set MCP_DB_<NAME>_URL on the server and pass <name>: a password in an "
                "argument is read by the model and kept in transcripts."
                + (f" Configured: {', '.join(sql_remote.names())}." if sql_remote.names() else ""),
            )
        # A database server is named, not located: the operator configured it (MCP_DB_<NAME>_URL), and no
        # password ever passes through a call.
        server = database if database and sql_remote.is_profile(database) else ""
        db = resolve_path(database) if database and not server else None
        if db is not None and not db.is_file():
            missing = _missing(db, "Database")
            if sql_remote.names():
                missing["hint"] += f" Database servers configured: {', '.join(sql_remote.names())}."
            return missing
        out = resolve_path(output_path) if output_path else None

        budget = memory_mb if memory_mb > 0 else memory_budget_mb()
        backup = ""
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            backup = _snapshot_if_exists(out)
            if backup:
                progress.append(info("Snapshot created", Path(backup).name))

        answer = run_query(
            sql,
            tables=named,
            database=db,
            remote=server,
            output=out,
            preview_rows=max_rows,
            memory_mb=budget,
            threads=worker_threads(),
        )
        source = ", ".join(f"{n} = {p.name}" for n, p in named.items()) or (db.name if db else server)
        total = answer["rows_total"]
        progress.append(
            ok(
                f"Queried {source}",
                f"{answer['engine']}: {total:,} row(s) in {answer['seconds']} s"
                if total is not None
                else f"{answer['engine']}: first {answer['rows_returned']} row(s) in {answer['seconds']} s",
            )
        )
        result: dict = {
            "success": True,
            "op": "query_data",
            **answer,
            "memory_limit_mb": budget if answer["engine"] == "duckdb" else None,
            "progress": progress,
        }
        if out is not None:
            progress.append(ok("Result written", f"{out.name} ({total:,} rows)"))
            result["output_path"] = str(out)
            result["output_name"] = out.name
            if backup:
                result["backup"] = backup
            append_receipt(
                str(out),
                tool="query_data",
                args={
                    "sql": sql[:500],
                    "tables": {n: p.name for n, p in named.items()},
                    "database": db.name if db else server,
                },
                result=f"{total:,} rows",
                backup=backup,
            )
            if out.suffix.lower() == ".csv":
                result["hint"] = (
                    "The whole result is in the file; inspect_dataset or train_classifier read it from there."
                )
                result["handover"] = make_handover(
                    "INSPECT",
                    ["inspect_dataset", "train_classifier", "train_regressor"],
                    {"file_path": str(out)},
                )
            else:
                result["hint"] = (
                    "The whole result is in the file. The other tools read CSV, not Parquet: write a .csv "
                    "output_path (or a narrower query) to train on it; query_data reads this file again."
                )
        elif answer["is_preview"]:
            result["hint"] = (
                f"Showing the first {answer['rows_returned']} row(s) of a larger result. Pass output_path "
                "(.csv or .parquet) to keep all of it, or aggregate or LIMIT in the query."
            )
        else:
            result["hint"] = "Pass output_path (.csv or .parquet) to keep this result as a file the other tools read."
        result["context"] = make_context(
            "query_data",
            f"Queried {source}: {answer['engine']}",
            [{"type": out.suffix.lstrip(".").lower(), "path": str(out), "role": "query_result"}] if out else [],
        )
        result["token_estimate"] = len(str(result)) // 4
        return result
    except QueryRefused as exc:
        refused = _error(
            str(exc),
            "A query is one SELECT over the files named in file_path or tables (each is a table of that name).",
        )
        refused["op"] = "query_data"
        refused["progress"] = [fail("Query refused", str(exc).splitlines()[0][:200])]
        return refused
    except ValueError as exc:
        return _error(str(exc), "Check the paths are absolute and inside the data folder.")
    except Exception as exc:
        logger.exception("query_data error")
        return _error(str(exc), "Check the paths are absolute and the files readable.")
