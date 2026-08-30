"""Durable, analytics-ready archive for FFBridge postmortem data."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import polars as pl


ARCHIVE_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.parquet"
SESSION_INDEX_FILENAME = "sessions.parquet"
PLAYER_INDEX_FILENAME = "player_sessions.parquet"
COMPACTION_FILENAME = "compaction.json"
SEATS = ("N", "E", "S", "W")
DEFAULT_ARCHIVE_DIR = pathlib.Path(
    os.environ.get(
        "FFBRIDGE_POSTMORTEM_ARCHIVE_DIR",
        pathlib.Path(
            os.environ.get(
                "FFBRIDGE_POSTMORTEM_CACHE_DIR",
                pathlib.Path(__file__).resolve().parent / "cache",
            )
        )
        / "archive",
    )
)
_ARCHIVE_LOCK = threading.RLock()

MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "session_id": pl.String,
    "revision": pl.String,
    "schema_version": pl.Int32,
    "schema_hash": pl.String,
    "content_sha256": pl.String,
    "Date": pl.Date,
    "year": pl.Int32,
    "series_id": pl.String,
    "group_id": pl.String,
    "organization_id": pl.String,
    "organization_name": pl.String,
    "source_updated_at": pl.String,
    "archived_at": pl.String,
    "row_count": pl.Int64,
    "column_count": pl.Int32,
    "size_bytes": pl.Int64,
    "fragment_path": pl.String,
}


def resolve_archive_dir(archive_dir: pathlib.Path | None = None) -> pathlib.Path:
    return pathlib.Path(archive_dir) if archive_dir is not None else DEFAULT_ARCHIVE_DIR


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_date(frame: pl.DataFrame) -> datetime:
    if "Date" not in frame.columns or frame.is_empty():
        raise ValueError("Canonical postmortem requires at least one Date value")
    value = frame["Date"].drop_nulls().first()
    if value is None:
        raise ValueError("Canonical postmortem Date is entirely null")
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Cannot parse postmortem Date {value!r}") from exc


def canonicalize_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove request-player perspective from an otherwise session-wide frame."""
    if frame.is_empty():
        raise ValueError("Cannot archive an empty postmortem frame")
    if "Pair_Direction" not in frame.columns:
        raise ValueError("Postmortem frame lacks Pair_Direction")
    return frame.with_columns(
        pl.lit(None, dtype=pl.String).alias("Pair_Direction"),
    )


def restore_pair_direction(frame: pl.DataFrame, pair_direction: str) -> pl.DataFrame:
    if pair_direction not in {"NS", "EW"}:
        raise ValueError(f"Invalid pair direction {pair_direction!r}")
    return frame.with_columns(pl.lit(pair_direction).alias("Pair_Direction"))


def _schema_hash(frame: pl.DataFrame) -> str:
    schema_json = json.dumps(
        [(name, str(dtype)) for name, dtype in frame.schema.items()],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(schema_json.encode("utf-8")).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_parquet(frame: pl.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(value: Mapping[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def manifest_path(archive_dir: pathlib.Path | None = None) -> pathlib.Path:
    return resolve_archive_dir(archive_dir) / MANIFEST_FILENAME


def read_manifest(archive_dir: pathlib.Path | None = None) -> pl.DataFrame:
    path = manifest_path(archive_dir)
    if not path.is_file():
        return pl.DataFrame(schema=MANIFEST_SCHEMA)
    frame = pl.read_parquet(path)
    missing = set(MANIFEST_SCHEMA) - set(frame.columns)
    if missing:
        raise ValueError(f"Archive manifest lacks columns: {sorted(missing)}")
    return frame.cast(MANIFEST_SCHEMA)


def latest_manifest(archive_dir: pathlib.Path | None = None) -> pl.DataFrame:
    manifest = read_manifest(archive_dir)
    if manifest.is_empty():
        return manifest
    return (
        manifest.sort(["session_id", "archived_at", "revision"])
        .unique(subset=["session_id"], keep="last", maintain_order=True)
        .sort(["Date", "session_id"])
    )


def _write_manifest(frame: pl.DataFrame, archive_dir: pathlib.Path) -> None:
    ordered = frame.cast(MANIFEST_SCHEMA).sort(
        ["session_id", "archived_at", "revision"]
    )
    _atomic_write_parquet(ordered, manifest_path(archive_dir))


def _validate_schema_compatibility(
    frame: pl.DataFrame,
    manifest: pl.DataFrame,
    root: pathlib.Path,
) -> None:
    """Permit additive columns but reject missing columns and dtype changes."""
    if manifest.is_empty():
        return
    reference_row = manifest.sort("archived_at").row(-1, named=True)
    reference_path = root / reference_row["fragment_path"]
    if not reference_path.is_file():
        raise FileNotFoundError(f"Schema reference fragment is missing: {reference_path}")
    reference = pl.read_parquet_schema(reference_path)
    current = frame.schema
    missing = sorted(set(reference) - set(current))
    changed = sorted(
        name
        for name in set(reference) & set(current)
        if reference[name] != current[name]
    )
    if missing or changed:
        raise ValueError(
            "Incompatible postmortem archive schema; increment the archive schema "
            f"version before migration. Missing={missing[:20]}, "
            f"dtype_changed={changed[:20]}"
        )


def prepare_archive_session(
    frame: pl.DataFrame,
    session_id: Any,
    *,
    context: Mapping[str, Any] | None = None,
    archive_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Write one immutable canonical session revision without updating the manifest."""
    started = time.perf_counter()
    root = resolve_archive_dir(archive_dir)
    canonical = canonicalize_frame(frame)
    session_text = str(session_id).strip()
    if not session_text:
        raise ValueError("session_id is required")
    session_date = _first_date(canonical)
    year = session_date.year
    context = context or {}

    staging_dir = root / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging = staging_dir / f"{session_text}-{os.getpid()}-{threading.get_ident()}.parquet"
    try:
        canonical.write_parquet(staging, compression="zstd", statistics=True)
        content_hash = _sha256_file(staging)
        revision = content_hash[:16]
        relative = (
            pathlib.Path("fragments")
            / f"year={year}"
            / f"session_id={session_text}"
            / f"revision={revision}"
            / "data.parquet"
        )
        destination = root / relative
        created = not destination.is_file()
        if created:
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging, destination)
            except FileExistsError:
                created = False
        record = {
            "session_id": session_text,
            "revision": revision,
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "schema_hash": _schema_hash(canonical),
            "content_sha256": content_hash,
            "Date": session_date.date(),
            "year": year,
            "series_id": _clean_string(context.get("series_id")),
            "group_id": _clean_string(context.get("group_id")),
            "organization_id": _clean_string(
                context.get("organization_id") or context.get("org_id")
            ),
            "organization_name": _clean_string(context.get("organization_name")),
            "source_updated_at": _clean_string(context.get("source_updated_at")),
            "archived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "row_count": canonical.height,
            "column_count": canonical.width,
            "size_bytes": destination.stat().st_size,
            "fragment_path": relative.as_posix(),
        }
        return {
            **record,
            "archive_file": str(destination),
            "created": created,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        staging.unlink(missing_ok=True)


def commit_archive_records(
    records: Sequence[Mapping[str, Any]],
    archive_dir: pathlib.Path | None = None,
) -> int:
    """Validate and append prepared records with one atomic manifest rewrite."""
    if not records:
        return 0
    root = resolve_archive_dir(archive_dir)
    with _ARCHIVE_LOCK:
        manifest = read_manifest(root)
        reference_schema = None
        if not manifest.is_empty():
            reference_row = manifest.sort("archived_at").row(-1, named=True)
            reference_schema = pl.read_parquet_schema(
                root / reference_row["fragment_path"]
            )
        rows: list[dict[str, Any]] = []
        existing_keys = set(
            zip(manifest["session_id"].to_list(), manifest["revision"].to_list())
        )
        for value in records:
            record = {name: value.get(name) for name in MANIFEST_SCHEMA}
            key = (str(record["session_id"]), str(record["revision"]))
            if key in existing_keys:
                continue
            fragment = root / str(record["fragment_path"])
            if not fragment.is_file():
                raise FileNotFoundError(f"Prepared archive fragment is missing: {fragment}")
            current_schema = pl.read_parquet_schema(fragment)
            if reference_schema is None:
                reference_schema = current_schema
            missing = sorted(set(reference_schema) - set(current_schema))
            changed = sorted(
                name
                for name in set(reference_schema) & set(current_schema)
                if reference_schema[name] != current_schema[name]
            )
            if missing or changed:
                raise ValueError(
                    "Incompatible prepared postmortem schema. "
                    f"Missing={missing[:20]}, dtype_changed={changed[:20]}"
                )
            rows.append(record)
            existing_keys.add(key)
        if rows:
            additions = pl.DataFrame(rows, schema=MANIFEST_SCHEMA)
            _write_manifest(pl.concat([manifest, additions]), root)
        return len(rows)


def archive_session(
    frame: pl.DataFrame,
    session_id: Any,
    *,
    context: Mapping[str, Any] | None = None,
    archive_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Write one immutable canonical session revision and update the manifest."""
    record = prepare_archive_session(
        frame,
        session_id,
        context=context,
        archive_dir=archive_dir,
    )
    committed = commit_archive_records([record], archive_dir)
    return {**record, "created": bool(committed)}


def resolve_archived_session(
    session_id: Any,
    archive_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    root = resolve_archive_dir(archive_dir)
    rows = latest_manifest(root).filter(pl.col("session_id") == str(session_id))
    if rows.is_empty():
        raise FileNotFoundError(f"No archived postmortem for session {session_id}")
    path = root / rows["fragment_path"][0]
    if not path.is_file():
        raise FileNotFoundError(f"Archived fragment is missing: {path}")
    return path


def archived_sessions_for_player(
    player_ids: Sequence[str],
    archive_dir: pathlib.Path | None = None,
) -> pl.DataFrame:
    root = resolve_archive_dir(archive_dir)
    path = root / "indexes" / PLAYER_INDEX_FILENAME
    if not path.is_file():
        return pl.DataFrame()
    wanted = [str(value) for value in player_ids]
    return (
        pl.scan_parquet(path)
        .filter(pl.col("player_id").is_in(wanted))
        .collect()
        .sort(["Date", "session_id"], descending=[True, False])
    )


def _available_columns(path: pathlib.Path) -> set[str]:
    return set(pl.scan_parquet(path).collect_schema().names())


def rebuild_indexes(archive_dir: pathlib.Path | None = None) -> dict[str, int]:
    """Rebuild small session and long-form player/session indexes."""
    root = resolve_archive_dir(archive_dir)
    latest = latest_manifest(root)
    index_dir = root / "indexes"
    session_path = index_dir / SESSION_INDEX_FILENAME
    player_path = index_dir / PLAYER_INDEX_FILENAME
    if latest.is_empty():
        _atomic_write_parquet(latest, session_path)
        _atomic_write_parquet(
            pl.DataFrame(
                schema={
                    "player_id": pl.String,
                    "player_name": pl.String,
                    "seat": pl.String,
                    "session_id": pl.String,
                    "Date": pl.Date,
                    "series_id": pl.String,
                    "organization_name": pl.String,
                    "fragment_path": pl.String,
                }
            ),
            player_path,
        )
        return {"sessions": 0, "player_sessions": 0}

    _atomic_write_parquet(latest, session_path)
    player_frames: list[pl.DataFrame] = []
    for row in latest.iter_rows(named=True):
        fragment = root / row["fragment_path"]
        columns = _available_columns(fragment)
        for seat in SEATS:
            id_column = f"Player_ID_{seat}"
            if id_column not in columns:
                continue
            name_column = f"Player_Name_{seat}"
            selected = [id_column]
            if name_column in columns:
                selected.append(name_column)
            players = (
                pl.scan_parquet(fragment)
                .select(selected)
                .filter(pl.col(id_column).is_not_null())
                .unique()
                .collect()
                .rename({id_column: "player_id"})
                .with_columns(
                    pl.col("player_id").cast(pl.String),
                    (
                        pl.col(name_column).cast(pl.String)
                        if name_column in selected
                        else pl.lit(None, dtype=pl.String)
                    ).alias("player_name"),
                    pl.lit(seat).alias("seat"),
                    pl.lit(row["session_id"]).alias("session_id"),
                    pl.lit(row["Date"]).cast(pl.Date).alias("Date"),
                    pl.lit(row["series_id"], dtype=pl.String).alias("series_id"),
                    pl.lit(
                        row["organization_name"], dtype=pl.String
                    ).alias("organization_name"),
                    pl.lit(row["fragment_path"]).alias("fragment_path"),
                )
                .select(
                    "player_id",
                    "player_name",
                    "seat",
                    "session_id",
                    "Date",
                    "series_id",
                    "organization_name",
                    "fragment_path",
                )
            )
            player_frames.append(players)
    player_index = (
        pl.concat(player_frames, how="vertical_relaxed")
        .unique(subset=["player_id", "session_id", "seat"])
        .sort(["player_id", "Date", "session_id"])
        if player_frames
        else pl.DataFrame()
    )
    _atomic_write_parquet(player_index, player_path)
    return {"sessions": latest.height, "player_sessions": player_index.height}


def _partition_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return int(row["year"]), str(row.get("series_id") or "unknown")


def compact_archive(
    archive_dir: pathlib.Path | None = None,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Rebuild only analytics partitions containing new session revisions."""
    root = resolve_archive_dir(archive_dir)
    latest = latest_manifest(root)
    if latest.is_empty():
        return {"partitions": 0, "rows": 0}
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in latest.iter_rows(named=True):
        groups.setdefault(_partition_key(row), []).append(row)

    rebuilt = 0
    rows_written = 0
    for (year, series_id), rows in sorted(groups.items()):
        safe_series = series_id.replace("/", "_").replace("\\", "_")
        destination = (
            root
            / "dataset"
            / f"year={year}"
            / f"series_id={safe_series}"
            / "data.parquet"
        )
        newest_source = max(row["archived_at"] for row in rows)
        state_path = destination.with_suffix(".json")
        if not force and destination.is_file() and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("newest_source") == newest_source:
                continue
        sources = [(root / row["fragment_path"]).as_posix() for row in rows]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        con = duckdb.connect()
        try:
            quoted_sources = "[" + ", ".join(
                "'" + source.replace("'", "''") + "'" for source in sources
            ) + "]"
            order_columns = [
                column
                for column in ("Date", "Board", "Pair_Number_NS", "Pair_Number_EW")
                if column in _available_columns(pathlib.Path(sources[0]))
            ]
            order_sql = (
                " ORDER BY " + ", ".join(f'"{column}"' for column in order_columns)
                if order_columns
                else ""
            )
            target = str(temporary).replace("\\", "/").replace("'", "''")
            con.execute(
                f"COPY (SELECT * FROM read_parquet({quoted_sources}, "
                f"union_by_name=true){order_sql}) TO '{target}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)"
            )
        finally:
            con.close()
        os.replace(temporary, destination)
        partition_rows = pl.scan_parquet(destination).select(pl.len()).collect().item()
        _atomic_write_json(
            {
                "schema_version": ARCHIVE_SCHEMA_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "newest_source": newest_source,
                "session_count": len(rows),
                "row_count": partition_rows,
            },
            state_path,
        )
        rebuilt += 1
        rows_written += partition_rows
    _atomic_write_json(
        {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "latest_session_count": latest.height,
            "partition_count": len(groups),
        },
        root / COMPACTION_FILENAME,
    )
    return {"partitions": rebuilt, "rows": rows_written}


def dataset_files(archive_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    root = resolve_archive_dir(archive_dir)
    return sorted((root / "dataset").glob("year=*/series_id=*/data.parquet"))


def archive_info(archive_dir: pathlib.Path | None = None) -> dict[str, Any]:
    root = resolve_archive_dir(archive_dir)
    manifest = read_manifest(root)
    latest = latest_manifest(root)
    files = dataset_files(root)
    return {
        "archive_dir": str(root),
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "revisions": manifest.height,
        "sessions": latest.height,
        "fragment_bytes": int(latest["size_bytes"].sum()) if latest.height else 0,
        "dataset_files": len(files),
        "dataset_bytes": sum(path.stat().st_size for path in files),
        "date_min": str(latest["Date"].min()) if latest.height else None,
        "date_max": str(latest["Date"].max()) if latest.height else None,
    }


def copy_cache_to_archive(
    cache_files: Iterable[pathlib.Path],
    archive_dir: pathlib.Path | None = None,
) -> dict[str, int]:
    """Best-effort cache migration for audit use; raw-source backfill is preferred."""
    created = 0
    unchanged = 0
    for path in cache_files:
        stem = path.stem
        parts = stem.split("-", 2)
        if len(parts) != 3 or parts[0] != "df":
            continue
        result = archive_session(
            pl.read_parquet(path),
            parts[1],
            archive_dir=archive_dir,
            context={"source_updated_at": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat(timespec="seconds")},
        )
        if result["created"]:
            created += 1
        else:
            unchanged += 1
    return {"created": created, "unchanged": unchanged}
