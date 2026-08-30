"""Normalized Parquet layout for FFBridge postmortem analytics."""
from __future__ import annotations

import json
import os
import pathlib
import threading
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import polars as pl

import ffbridge_postmortem_archive as archive


LAYOUT_VERSION = 2
KEY_COLUMNS = ("session_id", "Board")
SEATS = ("N", "E", "S", "W")
HIERARCHICAL_MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "session_id": pl.String,
    "revision": pl.String,
    "Date": pl.Date,
    "year": pl.Int32,
    "series_id": pl.String,
    "archived_at": pl.String,
    "board_rows": pl.Int64,
    "result_rows": pl.Int64,
    "boards_path": pl.String,
    "results_path": pl.String,
}
_HIERARCHICAL_LOCK = threading.RLock()
_DEFAULT_CACHE_DIR = pathlib.Path(
    os.environ.get(
        "FFBRIDGE_POSTMORTEM_CACHE_DIR",
        str(pathlib.Path(__file__).resolve().parent / "cache"),
    )
)


def resolve_hierarchical_dir(
    cache_dir: pathlib.Path | None = None,
) -> pathlib.Path | None:
    """Return the hierarchical archive to read/write, or None if unpublished.

    Preference: explicit env, then the Elo data mount used by deploy, then
    the local postmortem cache. A path is used only when metadata.json exists
    so an empty default never shadows the flat archive.
    """
    env = os.environ.get("FFBRIDGE_POSTMORTEM_HIERARCHICAL_DIR", "").strip()
    if env:
        return pathlib.Path(env)
    cache_env = os.environ.get("FFBRIDGE_CACHE_DIR", "").strip()
    candidates = []
    if cache_env:
        candidates.append(
            pathlib.Path(cache_env) / "postmortem_archive_hierarchical"
        )
    candidates.append(pathlib.Path("/data/ffbridge/postmortem_archive_hierarchical"))
    cache_root = pathlib.Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    candidates.append(cache_root / "archive" / "hierarchical")
    for path in candidates:
        if (path / "metadata.json").is_file():
            return path
    return None


def hierarchical_has_session(output_dir: pathlib.Path, session_id: str) -> bool:
    manifest = latest_hierarchical_manifest(output_dir)
    if manifest.is_empty():
        return False
    return manifest.filter(pl.col("session_id") == str(session_id)).height > 0


def _atomic_write_parquet(frame: pl.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        frame.write_parquet(
            temporary,
            compression="zstd",
            statistics=True,
            row_group_size=131_072,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(value: Mapping[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _board_columns(frames: Sequence[pl.DataFrame]) -> set[str]:
    common = set.intersection(*(set(frame.columns) for frame in frames))
    candidates = common - set(KEY_COLUMNS) - {"_result_row_id"}
    invariant = set(candidates)
    for frame in frames:
        ordered = sorted(invariant)
        for offset in range(0, len(ordered), 200):
            batch = ordered[offset : offset + 200]
            grouped = frame.group_by("Board").agg(
                *[
                    pl.col(column).hash(seed=0).n_unique().alias(column)
                    for column in batch
                ]
            )
            maxima = grouped.select(
                *[pl.col(column).max().alias(column) for column in batch]
            ).row(0, named=True)
            invariant.difference_update(
                column for column, count in maxima.items() if count > 1
            )
    return invariant


def _namespace(column: str, table: str) -> str | None:
    if table == "boards":
        if column in {"PBN", "Dealer", "Vul", "iVul", "Vul_NS", "Vul_EW"}:
            return "deal"
        if column.startswith(("DD_", "DDScore_", "DD_Score_")):
            return "double_dummy"
        if column.startswith("Par"):
            return "par"
        if column.startswith("EV_"):
            return "expected_value"
        return None
    if column.startswith(("Player_ID_", "Player_Name_", "Pair_ID", "Pair_Name")):
        return "players"
    if column.startswith("Pair_Number_"):
        return "players"
    if column in {
        "Contract",
        "Declarer",
        "Declarer_Direction",
        "Declarer_ID",
        "Declarer_Name",
        "Result",
        "Tricks",
        "BidLvl",
        "BidSuit",
        "Dbl",
    }:
        return "contract"
    if column.startswith(("Score_", "Pct_", "MP_")):
        return "score"
    return None


def _correct_column_mapping(
    mapping: Mapping[str, Mapping[str, str | None]],
) -> dict[str, dict[str, str | None]]:
    corrected = {column: dict(entry) for column, entry in mapping.items()}
    if "Section_Name" in corrected:
        corrected["Section_Name"] = {
            "table": "results",
            "storage_column": "Section_Name",
            "field": None,
        }
    return corrected


def _pack_structs(
    frame: pl.DataFrame,
    table: str,
) -> tuple[pl.DataFrame, dict[str, dict[str, str | None]]]:
    groups: dict[str, list[str]] = {}
    mapping: dict[str, dict[str, str | None]] = {}
    for column in frame.columns:
        namespace = _namespace(column, table)
        if namespace is None:
            mapping[column] = {
                "table": table,
                "storage_column": column,
                "field": None,
            }
            continue
        groups.setdefault(namespace, []).append(column)
        mapping[column] = {
            "table": table,
            "storage_column": namespace,
            "field": column,
        }
    packed = frame
    for namespace, columns in groups.items():
        packed = packed.with_columns(
            pl.struct([pl.col(column) for column in columns]).alias(namespace)
        ).drop(columns)
    return packed, mapping


def build_normalized_subset(
    archive_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    session_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = pathlib.Path(archive_dir)
    output = pathlib.Path(output_dir)
    manifest = archive.latest_manifest(root)
    if session_ids is not None:
        wanted = [str(value) for value in session_ids]
        manifest = manifest.filter(pl.col("session_id").is_in(wanted))
    if manifest.is_empty():
        raise ValueError("No archived sessions selected for normalization")

    frames: list[pl.DataFrame] = []
    for row in manifest.iter_rows(named=True):
        frame = pl.read_parquet(root / row["fragment_path"]).with_columns(
            pl.lit(row["session_id"]).alias("session_id")
        )
        if "Board" not in frame.columns:
            raise ValueError(f"Session {row['session_id']} lacks Board")
        frames.append(frame.with_row_index("_result_row_id"))

    invariant = _board_columns(frames)
    invariant.discard("Section_Name")
    board_columns = [
        column
        for column in frames[0].columns
        if column in invariant and column not in KEY_COLUMNS
    ]
    result_columns = [
        column
        for column in frames[0].columns
        if column not in invariant
        and column not in KEY_COLUMNS
        and column != "_result_row_id"
    ]
    board_frames: list[pl.DataFrame] = []
    result_frames: list[pl.DataFrame] = []
    for frame in frames:
        boards = (
            frame.select(*KEY_COLUMNS, *board_columns)
            .unique(subset=list(KEY_COLUMNS), keep="first", maintain_order=True)
        )
        expected_boards = frame.select(*KEY_COLUMNS).unique().height
        if boards.height != expected_boards:
            raise ValueError("Board normalization changed the board key cardinality")
        board_frames.append(boards)
        result_frames.append(
            frame.select(*KEY_COLUMNS, "_result_row_id", *result_columns)
        )

    boards_flat = pl.concat(board_frames, how="vertical_relaxed").sort(
        ["session_id", "Board"]
    )
    results_flat = pl.concat(result_frames, how="vertical_relaxed").sort(
        ["session_id", "Board", "_result_row_id"]
    )
    boards, board_mapping = _pack_structs(boards_flat, "boards")
    results, result_mapping = _pack_structs(results_flat, "results")
    mapping = {**board_mapping, **result_mapping}

    sessions = manifest.select(
        "session_id",
        "Date",
        "year",
        "series_id",
        "group_id",
        "organization_id",
        "organization_name",
        "revision",
        "row_count",
        "column_count",
    ).sort(["Date", "session_id"])
    _atomic_write_parquet(sessions, output / "sessions.parquet")
    _atomic_write_parquet(boards, output / "boards.parquet")
    _atomic_write_parquet(results, output / "results.parquet")
    metadata = {
        "layout_version": LAYOUT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_count": sessions.height,
        "board_rows": boards.height,
        "result_rows": results.height,
        "original_columns": len(mapping),
        "board_logical_columns": len(board_mapping),
        "result_logical_columns": len(result_mapping),
        "boards_storage_columns": boards.width,
        "results_storage_columns": results.width,
        "column_mapping": mapping,
    }
    _atomic_write_json(metadata, output / "metadata.json")
    return metadata


def initialize_hierarchical_layout(
    output_dir: pathlib.Path,
    seed_metadata_path: pathlib.Path,
) -> dict[str, Any]:
    """Initialize a production layout from a validated representative subset."""
    output = pathlib.Path(output_dir)
    seed = json.loads(pathlib.Path(seed_metadata_path).read_text(encoding="utf-8"))
    if seed.get("layout_version") not in {1, LAYOUT_VERSION}:
        raise ValueError(
            f"Expected layout version 1 or {LAYOUT_VERSION}, got "
            f"{seed.get('layout_version')!r}"
        )
    if not isinstance(seed.get("column_mapping"), dict):
        raise ValueError("Seed metadata lacks column_mapping")
    seed["column_mapping"] = _correct_column_mapping(seed["column_mapping"])
    seed["layout_version"] = LAYOUT_VERSION
    destination = output / "metadata.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        existing["column_mapping"] = _correct_column_mapping(
            existing["column_mapping"]
        )
        existing["layout_version"] = LAYOUT_VERSION
        if existing["column_mapping"] != seed["column_mapping"]:
            raise ValueError("Hierarchical layout already has a different mapping")
        _atomic_write_json(existing, destination)
        return existing
    metadata = {
        **seed,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layout_source": str(pathlib.Path(seed_metadata_path).resolve()),
        "production": True,
    }
    _atomic_write_json(metadata, destination)
    return metadata


def _hierarchical_manifest_path(output_dir: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(output_dir) / "manifest.parquet"


def read_hierarchical_manifest(output_dir: pathlib.Path) -> pl.DataFrame:
    path = _hierarchical_manifest_path(output_dir)
    if not path.is_file():
        return pl.DataFrame(schema=HIERARCHICAL_MANIFEST_SCHEMA)
    frame = pl.read_parquet(path)
    missing = set(HIERARCHICAL_MANIFEST_SCHEMA) - set(frame.columns)
    if missing:
        raise ValueError(f"Hierarchical manifest lacks columns: {sorted(missing)}")
    return frame.cast(HIERARCHICAL_MANIFEST_SCHEMA)


def latest_hierarchical_manifest(output_dir: pathlib.Path) -> pl.DataFrame:
    manifest = read_hierarchical_manifest(output_dir)
    if manifest.is_empty():
        return manifest
    return (
        manifest.sort(["session_id", "archived_at", "revision"])
        .unique(subset=["session_id"], keep="last", maintain_order=True)
        .sort(["Date", "session_id"])
    )


def _pack_with_mapping(
    frame: pl.DataFrame,
    table: str,
    mapping: Mapping[str, Mapping[str, str | None]],
) -> pl.DataFrame:
    keys = [column for column in (*KEY_COLUMNS, "_result_row_id") if column in frame]
    table_columns = [
        column
        for column, entry in mapping.items()
        if entry["table"] == table and column in frame.columns and column not in keys
    ]
    direct = [
        column
        for column in table_columns
        if mapping[column]["field"] is None
    ]
    groups: dict[str, list[str]] = {}
    for column in table_columns:
        field = mapping[column]["field"]
        if field is not None:
            groups.setdefault(str(mapping[column]["storage_column"]), []).append(column)
    expressions: list[pl.Expr] = [pl.col(column) for column in keys]
    expressions.extend(pl.col(column) for column in direct)
    expressions.extend(
        pl.struct([pl.col(column) for column in columns]).alias(storage_column)
        for storage_column, columns in groups.items()
    )
    return frame.select(*expressions)


def prepare_hierarchical_session(
    frame: pl.DataFrame,
    *,
    session_id: str,
    revision: str,
    output_dir: pathlib.Path,
    series_id: str | None = None,
) -> dict[str, Any]:
    """Write one immutable boards/results hierarchy without updating its manifest."""
    output = pathlib.Path(output_dir)
    metadata = _load_metadata(output)
    mapping = metadata["column_mapping"]
    canonical = archive.canonicalize_frame(frame).with_columns(
        pl.lit(str(session_id)).alias("session_id")
    ).with_row_index("_result_row_id")
    expected = set(mapping) - {"session_id", "_result_row_id"}
    missing = sorted(expected - set(canonical.columns))
    extra = sorted(
        set(canonical.columns) - set(mapping) - {"session_id", "_result_row_id"}
    )
    if missing or extra:
        raise ValueError(
            "Hierarchical schema differs from the validated layout. "
            f"Missing={missing[:20]}, extra={extra[:20]}"
        )

    board_columns = {
        column
        for column, entry in mapping.items()
        if entry["table"] == "boards" and column not in KEY_COLUMNS
    }
    invariant = _board_columns([canonical])
    violations = sorted(board_columns - invariant)
    if violations:
        raise ValueError(
            "Columns classified as board-level vary within a board: "
            f"{violations[:20]}"
        )
    result_columns = [
        column
        for column, entry in mapping.items()
        if entry["table"] == "results"
        and column not in KEY_COLUMNS
        and column != "_result_row_id"
    ]
    boards_flat = (
        canonical.select(*KEY_COLUMNS, *sorted(board_columns))
        .unique(subset=list(KEY_COLUMNS), keep="first", maintain_order=True)
        .sort(["session_id", "Board"])
    )
    results_flat = canonical.select(
        *KEY_COLUMNS, "_result_row_id", *result_columns
    ).sort(["session_id", "Board", "_result_row_id"])
    boards = _pack_with_mapping(boards_flat, "boards", mapping)
    results = _pack_with_mapping(results_flat, "results", mapping)

    session_date = canonical["Date"].drop_nulls().first()
    if session_date is None:
        raise ValueError(f"Session {session_id} has no Date")
    if isinstance(session_date, datetime):
        session_date = session_date.date()
    elif not hasattr(session_date, "year"):
        session_date = datetime.fromisoformat(str(session_date)).date()
    year = int(session_date.year)
    relative = (
        pathlib.Path("fragments")
        / f"year={year}"
        / f"session_id={session_id}"
        / f"layout_version={LAYOUT_VERSION}"
        / f"revision={revision}"
    )
    boards_path = relative / "boards.parquet"
    results_path = relative / "results.parquet"
    absolute_boards = output / boards_path
    absolute_results = output / results_path
    if not absolute_boards.is_file():
        _atomic_write_parquet(boards, absolute_boards)
    if not absolute_results.is_file():
        _atomic_write_parquet(results, absolute_results)

    return {
        "session_id": str(session_id),
        "revision": revision,
        "Date": session_date,
        "year": year,
        "series_id": str(series_id) if series_id is not None else None,
        "archived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "board_rows": boards.height,
        "result_rows": results.height,
        "boards_path": boards_path.as_posix(),
        "results_path": results_path.as_posix(),
    }


def commit_hierarchical_records(
    records: Sequence[Mapping[str, Any]],
    output_dir: pathlib.Path,
) -> int:
    """Append prepared hierarchy records with one atomic manifest rewrite."""
    if not records:
        return 0
    output = pathlib.Path(output_dir)
    with _HIERARCHICAL_LOCK:
        manifest = read_hierarchical_manifest(output)
        existing_keys = set(
            zip(manifest["session_id"].to_list(), manifest["revision"].to_list())
        )
        rows: list[dict[str, Any]] = []
        for value in records:
            record = {name: value.get(name) for name in HIERARCHICAL_MANIFEST_SCHEMA}
            key = (str(record["session_id"]), str(record["revision"]))
            if key in existing_keys:
                continue
            for path_column in ("boards_path", "results_path"):
                fragment = output / str(record[path_column])
                if not fragment.is_file():
                    raise FileNotFoundError(
                        f"Prepared hierarchical fragment is missing: {fragment}"
                    )
            rows.append(record)
            existing_keys.add(key)
        if rows:
            additions = pl.DataFrame(
                rows,
                schema=HIERARCHICAL_MANIFEST_SCHEMA,
            )
            _atomic_write_parquet(
                pl.concat([manifest, additions]).sort(
                    ["session_id", "archived_at", "revision"]
                ),
                _hierarchical_manifest_path(output),
            )
        return len(rows)


def write_hierarchical_session(
    frame: pl.DataFrame,
    *,
    session_id: str,
    revision: str,
    output_dir: pathlib.Path,
    series_id: str | None = None,
) -> dict[str, Any]:
    """Write one immutable boards/results hierarchy and update its manifest."""
    record = prepare_hierarchical_session(
        frame,
        session_id=session_id,
        revision=revision,
        output_dir=output_dir,
        series_id=series_id,
    )
    committed = commit_hierarchical_records([record], output_dir)
    return {
        **record,
        "created": bool(committed),
        "boards_path": str(pathlib.Path(output_dir) / str(record["boards_path"])),
        "results_path": str(pathlib.Path(output_dir) / str(record["results_path"])),
    }


def _duckdb_compact(
    sources: Sequence[pathlib.Path],
    destination: pathlib.Path,
    order_columns: Sequence[str],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    source_sql = "[" + ", ".join(
        "'" + source.as_posix().replace("'", "''") + "'" for source in sources
    ) + "]"
    target = temporary.as_posix().replace("'", "''")
    order_sql = ", ".join(f'"{column}"' for column in order_columns)
    con = duckdb.connect()
    try:
        con.execute(
            f"COPY (SELECT * FROM read_parquet({source_sql}, union_by_name=true) "
            f"ORDER BY {order_sql}) TO '{target}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)"
        )
    finally:
        con.close()
    os.replace(temporary, destination)


def compact_hierarchical_archive(
    output_dir: pathlib.Path,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Compact latest board/result revisions into affected year/series partitions."""
    output = pathlib.Path(output_dir)
    latest = latest_hierarchical_manifest(output)
    if latest.is_empty():
        return {"partitions": 0, "board_rows": 0, "result_rows": 0}
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in latest.iter_rows(named=True):
        groups.setdefault(
            (int(row["year"]), str(row["series_id"] or "unknown")), []
        ).append(row)
    rebuilt = 0
    board_rows = 0
    result_rows = 0
    for (year, series_id), rows in sorted(groups.items()):
        safe_series = series_id.replace("/", "_").replace("\\", "_")
        partition = output / "dataset" / f"year={year}" / f"series_id={safe_series}"
        state_path = partition / "state.json"
        newest_source = max(row["archived_at"] for row in rows)
        if not force and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("newest_source") == newest_source:
                continue
        board_sources = [output / row["boards_path"] for row in rows]
        result_sources = [output / row["results_path"] for row in rows]
        board_destination = partition / "boards.parquet"
        result_destination = partition / "results.parquet"
        _duckdb_compact(
            board_sources,
            board_destination,
            ["session_id", "Board"],
        )
        _duckdb_compact(
            result_sources,
            result_destination,
            ["session_id", "Board", "_result_row_id"],
        )
        partition_board_rows = (
            pl.scan_parquet(board_destination).select(pl.len()).collect().item()
        )
        partition_result_rows = (
            pl.scan_parquet(result_destination).select(pl.len()).collect().item()
        )
        _atomic_write_json(
            {
                "layout_version": LAYOUT_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "newest_source": newest_source,
                "sessions": len(rows),
                "board_rows": partition_board_rows,
                "result_rows": partition_result_rows,
            },
            state_path,
        )
        rebuilt += 1
        board_rows += partition_board_rows
        result_rows += partition_result_rows
    return {
        "partitions": rebuilt,
        "board_rows": board_rows,
        "result_rows": result_rows,
    }


def hierarchical_info(output_dir: pathlib.Path | None) -> dict[str, Any]:
    if output_dir is None:
        return {"configured": False, "available": False}
    output = pathlib.Path(output_dir)
    metadata_path = output / "metadata.json"
    if not metadata_path.is_file():
        return {
            "configured": True,
            "available": False,
            "directory": str(output),
        }
    manifest = latest_hierarchical_manifest(output)
    board_files = _table_files(output, "boards")
    result_files = _table_files(output, "results")
    return {
        "configured": True,
        "available": True,
        "directory": str(output),
        "sessions": manifest.height,
        "date_min": str(manifest["Date"].min()) if manifest.height else None,
        "date_max": str(manifest["Date"].max()) if manifest.height else None,
        "board_files": len(board_files),
        "result_files": len(result_files),
        "bytes": sum(
            path.stat().st_size for path in [*board_files, *result_files]
        ),
    }


def _load_metadata(output_dir: pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(output_dir) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Normalized archive metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mapped_expr(
    mapping: Mapping[str, Mapping[str, str | None]],
    column: str,
) -> pl.Expr:
    entry = mapping[column]
    expression = pl.col(str(entry["storage_column"]))
    field = entry["field"]
    if field is not None:
        expression = expression.struct.field(str(field))
    return expression.alias(column)


def _table_files(output_dir: pathlib.Path, table: str) -> list[pathlib.Path]:
    output = pathlib.Path(output_dir)
    direct = output / f"{table}.parquet"
    if direct.is_file():
        return [direct]
    files = sorted(
        (output / "dataset").glob(f"year=*/series_id=*/{table}.parquet")
    )
    if files:
        return files
    return sorted(
        (output / "fragments").glob(
            f"year=*/session_id=*/revision=*/{table}.parquet"
        )
    )


def normalized_player_report(
    output_dir: pathlib.Path,
    *,
    session_id: str,
    player_ids: Sequence[str],
    columns: Sequence[str],
    only_player_rows: bool = True,
) -> pl.DataFrame:
    """Read only required leaves and reconstruct played-board report rows."""
    output = pathlib.Path(output_dir)
    metadata = _load_metadata(output)
    mapping = metadata["column_mapping"]
    manifest = latest_hierarchical_manifest(output)
    session_rows = manifest.filter(pl.col("session_id") == str(session_id))
    if session_rows.height:
        row = session_rows.row(0, named=True)
        results_files = [output / row["results_path"]]
        boards_files = [output / row["boards_path"]]
    else:
        results_files = _table_files(output, "results")
        boards_files = _table_files(output, "boards")
    if not results_files or not boards_files:
        raise FileNotFoundError(f"Normalized board/result files not found in {output}")
    missing = [column for column in columns if column not in mapping]
    selected_columns = [column for column in columns if column in mapping]
    if not selected_columns:
        raise ValueError(f"Normalized archive lacks report columns: {missing}")
    player_columns = [f"Player_ID_{seat}" for seat in SEATS]
    needed = list(dict.fromkeys([*selected_columns, *player_columns]))
    result_columns = [
        column
        for column in needed
        if column not in KEY_COLUMNS
        and column != "_result_row_id"
        and mapping[column]["table"] == "results"
    ]
    board_columns = [
        column
        for column in needed
        if column not in KEY_COLUMNS and mapping[column]["table"] == "boards"
    ]

    result_storage = list(
        dict.fromkeys(
            str(mapping[column]["storage_column"]) for column in result_columns
        )
    )
    results = (
        pl.scan_parquet(results_files)
        .filter(pl.col("session_id") == str(session_id))
        .select(
            "session_id",
            "Board",
            "_result_row_id",
            *result_storage,
        )
        .select(
            "session_id",
            "Board",
            "_result_row_id",
            *[_mapped_expr(mapping, column) for column in result_columns],
        )
    )
    if only_player_rows:
        results = results.filter(
            pl.any_horizontal(
                *[
                    pl.col(column).cast(pl.String).is_in(list(player_ids))
                    for column in player_columns
                ]
            )
        )
    if board_columns:
        board_storage = list(
            dict.fromkeys(
                str(mapping[column]["storage_column"]) for column in board_columns
            )
        )
        boards = (
            pl.scan_parquet(boards_files)
            .filter(pl.col("session_id") == str(session_id))
            .select("session_id", "Board", *board_storage)
            .select(
                "session_id",
                "Board",
                *[_mapped_expr(mapping, column) for column in board_columns],
            )
        )
        result = results.join(
            boards,
            on=["session_id", "Board"],
            how="left",
            validate="m:1",
        )
    else:
        result = results
    return (
        result.sort("_result_row_id")
        .select(*selected_columns)
        .collect(engine="streaming")
    )
