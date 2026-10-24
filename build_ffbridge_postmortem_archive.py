"""Backfill and compact the canonical FFBridge postmortem Parquet archive."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import sys
import time
from datetime import date, datetime
from typing import Any

import polars as pl
from tqdm import tqdm

import ffbridge_postmortem_archive as archive
import ffbridge_postmortem_create as create
import ffbridge_postmortem_normalized as normalized


APP_DIR = pathlib.Path(__file__).resolve().parent
ELO_RATINGS_DIR = APP_DIR.parent / "elo"
DEFAULT_SOURCE_DIR = pathlib.Path(r"E:\bridge\data\ffbridge\data")


class UnsupportedSessionError(ValueError):
    """The source session cannot produce a full augmented postmortem."""


# Stage 1's hand-records cache, loaded once per worker process.
_WORKER_HRS_CACHE: pl.DataFrame | None = None
PAR_COLUMNS = ("ParScore", "ParNumber", "ParContracts")


def load_dd_sd_cache(hrs_cache_path: pathlib.Path) -> pl.DataFrame:
    """Stage 1's cached DD and SD results, with Par cleared.

    The cache holds the same ddss DD tables and SD samples (seed 42, 10
    productions) the archive would compute, but stage 1 counts ParNumber
    differently. Clearing Par makes mlBridge recompute it from the cached DD
    table, so the archive keeps its own Par values while the solving (mostly
    single-dummy) is skipped.
    """
    pipeline = _load_quality_pipeline()
    cache = pipeline.load_hrs_cache(hrs_cache_path)
    if cache is None:
        raise FileNotFoundError(
            f"Hand-records cache not found: {hrs_cache_path} "
            "(stage 1 build_ffbridge_quality_parquets.py writes it)"
        )
    return cache.with_columns(
        pl.lit(None, dtype=cache.schema[column]).alias(column) for column in PAR_COLUMNS
    )


def _init_worker(hrs_cache_path_text: str) -> None:
    global _WORKER_HRS_CACHE
    _WORKER_HRS_CACHE = load_dd_sd_cache(pathlib.Path(hrs_cache_path_text))


def _session_hrs_cache(prepared: pl.DataFrame) -> pl.DataFrame:
    """Cache rows for this session's deals only, so the joins stay small."""
    if _WORKER_HRS_CACHE is None:
        raise RuntimeError("Worker started without the hand-records cache")
    return _WORKER_HRS_CACHE.join(
        prepared.select("PBN").unique(), on="PBN", how="semi"
    )


def _load_quality_pipeline() -> Any:
    if not ELO_RATINGS_DIR.is_dir():
        raise FileNotFoundError(f"elo source directory not found: {ELO_RATINGS_DIR}")
    if str(ELO_RATINGS_DIR) not in sys.path:
        sys.path.insert(0, str(ELO_RATINGS_DIR))
    import ffbridge_quality_pipeline

    return ffbridge_quality_pipeline


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD, got {value!r}") from exc


def _reject_ambiguous_scores(raw: pl.DataFrame) -> None:
    if not {"nsScore", "ewScore"}.issubset(raw.columns):
        return
    # Adjusted ('%Tou', '60%+') and passed-out ('PASSE') boards fill both
    # columns legitimately; only two trick scores are ambiguous.
    ambiguous_scores = raw.filter(
        pl.col("nsScore").cast(pl.String).str.strip_chars().str.contains(r"^-?\d+$")
        & pl.col("ewScore").cast(pl.String).str.strip_chars().str.contains(r"^-?\d+$")
    )
    if ambiguous_scores.height:
        raise UnsupportedSessionError(
            "Raw session has rows with numeric nsScore and ewScore both populated"
        )


def _prepare_raw_session(
    raw: pl.DataFrame,
    session_date: date,
) -> pl.DataFrame:
    _reject_ambiguous_scores(raw)
    converted = create.mlBridgeFFLib.convert_ffdf_lancelot_to_mldf(raw)
    additions: list[pl.Expr] = [
        pl.lit(session_date).alias("Date"),
        pl.lit("NS").alias("Pair_Direction"),
    ]
    if "section_name" in converted.columns:
        additions.append(pl.col("section_name").alias("Section_Name"))
    else:
        additions.append(pl.lit("FFBridge").alias("Section_Name"))
    prepared = converted.with_columns(additions)
    prepared = prepared.filter(
        pl.col("PBN").is_not_null()
        & (pl.col("PBN").cast(pl.String).str.strip_chars() != "")
        & pl.col("Contract").is_not_null()
        & (pl.col("Contract").cast(pl.String).str.strip_chars() != "")
    )
    if prepared.is_empty():
        raise UnsupportedSessionError(
            "Raw session has no rows with both PBN and Contract"
        )
    reduced = create.reduce_mldf_for_augment(prepared)
    pair_number_columns = [
        column
        for column in ("Pair_Number_NS", "Pair_Number_EW")
        if column in reduced.columns
    ]
    if pair_number_columns:
        reduced = reduced.with_columns(
            pl.col(column).cast(pl.Int64) for column in pair_number_columns
        )
    return reduced


def _session_context(
    metadata: dict[str, Any],
    raw: pl.DataFrame,
) -> dict[str, Any]:
    group_id = None
    if "group_id" in raw.columns and raw["group_id"].drop_nulls().len():
        group_id = raw["group_id"].drop_nulls().first()
    organization = metadata.get("organization") or {}
    competition = metadata.get("competition") or {}
    return {
        "series_id": (
            metadata.get("series_id")
            or metadata.get("simultaneousId")
            or competition.get("id")
        ),
        "group_id": group_id,
        "organization_id": metadata.get("organizationId") or organization.get("id"),
        "organization_name": (
            metadata.get("organization_name")
            or organization.get("name")
            or metadata.get("club")
        ),
        "source_updated_at": metadata.get("updatedAt") or metadata.get("date"),
    }


def _prepare_session_worker(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        session,
        source_dir_text,
        archive_dir_text,
        hierarchical_dir_text,
        metadata_entry,
        existing_flat,
    ) = task
    source_dir = pathlib.Path(source_dir_text)
    archive_dir = pathlib.Path(archive_dir_text)
    hierarchical_dir = (
        pathlib.Path(hierarchical_dir_text) if hierarchical_dir_text else None
    )
    pipeline = _load_quality_pipeline()
    session_id = str(session.session_id)
    try:
        if existing_flat is not None:
            augmented = pl.read_parquet(
                archive_dir / str(existing_flat["fragment_path"])
            )
            flat_record = None
            revision = str(existing_flat["revision"])
            series_id = existing_flat["series_id"]
        else:
            raw, _unmapped = pipeline.load_raw_session(source_dir, session)
            session_date, _metadata_path, metadata = metadata_entry
            prepared = _prepare_raw_session(raw, session_date)
            augmented = create.augment_and_cache_mldf(
                prepared,
                session_id,
                "archive",
                write_cache=False,
                hrs_cache_df=_session_hrs_cache(prepared),
            )
            context = _session_context(metadata, raw)
            flat_record = archive.prepare_archive_session(
                augmented,
                session_id,
                context=context,
                archive_dir=archive_dir,
            )
            revision = str(flat_record["revision"])
            series_id = context.get("series_id")
        hierarchical_record = None
        if hierarchical_dir is not None:
            hierarchical_record = normalized.prepare_hierarchical_session(
                augmented,
                session_id=session_id,
                revision=revision,
                output_dir=hierarchical_dir,
                series_id=str(series_id) if series_id is not None else None,
            )
        return {
            "status": "prepared",
            "session_id": session_id,
            "flat_record": flat_record,
            "hierarchical_record": hierarchical_record,
        }
    except (UnsupportedSessionError, pipeline.NoQualityRowsError) as exc:
        return {
            "status": "unsupported",
            "session_id": session_id,
            "reason": str(exc),
        }
    except ValueError as exc:
        if str(exc).startswith("Malformed score row in session "):
            return {
                "status": "unsupported",
                "session_id": session_id,
                "reason": str(exc),
            }
        raise


def _commit_prepared_batch(
    prepared: list[dict[str, Any]],
    archive_dir: pathlib.Path,
    hierarchical_dir: pathlib.Path | None,
) -> tuple[int, int]:
    flat_records = [
        item["flat_record"]
        for item in prepared
        if item["flat_record"] is not None
    ]
    hierarchical_records = [
        item["hierarchical_record"]
        for item in prepared
        if item["hierarchical_record"] is not None
    ]
    flat_created = archive.commit_archive_records(flat_records, archive_dir)
    hierarchical_created = (
        normalized.commit_hierarchical_records(
            hierarchical_records,
            hierarchical_dir,
        )
        if hierarchical_dir is not None
        else 0
    )
    return flat_created, hierarchical_created


def backfill_raw(
    source_dir: pathlib.Path,
    archive_dir: pathlib.Path,
    cutoff: date,
    *,
    since: date | None = None,
    limit: int | None = None,
    fetch_missing: bool = False,
    hierarchical_dir: pathlib.Path | None = None,
    workers: int = 1,
    commit_batch_size: int = 50,
) -> dict[str, int]:
    pipeline = _load_quality_pipeline()
    training_path = source_dir / "ffbridge_training_data_df.parquet"
    # Stage 1's audit manifest lists sessions already known complete, so their
    # ranking and team-score files are not read again.
    manifest_path = pipeline.default_audit_manifest_path(source_dir)
    metadata_by_session = pipeline.load_session_metadata(source_dir, cutoff)
    audit = pipeline.audit_historical_cache(
        source_dir,
        training_path,
        cutoff,
        manifest_path=manifest_path,
        metadata=metadata_by_session,
    )
    fetched = 0
    if fetch_missing:
        fetched = pipeline.fetch_missing_artifacts(audit)
        audit = pipeline.audit_historical_cache(
            source_dir,
            training_path,
            cutoff,
            manifest_path=manifest_path,
            metadata=metadata_by_session,
        )
    flat_manifest = archive.latest_manifest(archive_dir)
    flat_rows = {
        row["session_id"]: row for row in flat_manifest.iter_rows(named=True)
    }
    archived_ids = set(flat_rows)
    hierarchical_ids = (
        set(
            normalized.latest_hierarchical_manifest(hierarchical_dir)[
                "session_id"
            ].to_list()
        )
        if hierarchical_dir is not None
        else set()
    )
    candidates = [
        session
        for session in audit.sessions
        if session.complete
        and (since is None or date.fromisoformat(str(session.session_date)) >= since)
        and (
            session.session_id not in archived_ids
            or (
                hierarchical_dir is not None
                and session.session_id not in hierarchical_ids
            )
        )
    ]
    created = 0
    hierarchical_created = 0
    unsupported = 0
    processed = 0
    selected = candidates[:limit] if limit is not None else candidates
    tasks = [
        (
            session,
            str(source_dir),
            str(archive_dir),
            str(hierarchical_dir) if hierarchical_dir is not None else None,
            metadata_by_session.get(session.session_id),
            flat_rows.get(session.session_id),
        )
        for session in selected
    ]
    pending_batch: list[dict[str, Any]] = []
    hrs_cache_path = pipeline.default_hrs_cache_path(source_dir)
    load_dd_sd_cache(hrs_cache_path)  # fail before starting workers if missing
    # Workers inherit this at spawn; ddss reads it when it loads.
    if "DDSS_MAX_THREADS" not in os.environ:
        os.environ["DDSS_MAX_THREADS"] = str(max(1, (os.cpu_count() or 1) // workers))
    print(
        f"[archive-builder] DD/SD from {hrs_cache_path} (Par recomputed); "
        f"DDSS_MAX_THREADS={os.environ['DDSS_MAX_THREADS']} x {workers} workers",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(hrs_cache_path),),
    ) as executor:
        futures = [executor.submit(_prepare_session_worker, task) for task in tasks]
        with tqdm(total=len(futures), desc="Archiving FFBridge postmortems") as progress:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                progress.update()
                if result["status"] == "unsupported":
                    unsupported += 1
                    continue
                pending_batch.append(result)
                processed += 1
                if len(pending_batch) >= commit_batch_size:
                    flat_count, hierarchical_count = _commit_prepared_batch(
                        pending_batch,
                        archive_dir,
                        hierarchical_dir,
                    )
                    created += flat_count
                    hierarchical_created += hierarchical_count
                    pending_batch.clear()
    flat_count, hierarchical_count = _commit_prepared_batch(
        pending_batch,
        archive_dir,
        hierarchical_dir,
    )
    created += flat_count
    hierarchical_created += hierarchical_count
    return {
        "eligible": len(candidates),
        "created": created,
        "hierarchical_created": hierarchical_created,
        "processed": processed,
        "skipped_existing": len(archived_ids),
        "unsupported": unsupported,
        "fetched_missing_files": fetched,
    }


def _execute(args: argparse.Namespace, root: pathlib.Path) -> int:
    hierarchical_dir = (
        args.hierarchical_dir.resolve()
        if args.hierarchical_dir is not None
        else None
    )
    if args.audit_only:
        report = {
            "source_dir": str(args.source_dir.resolve()),
            "cache_files": len(list(args.cache_dir.resolve().glob("df-*.parquet"))),
            "archive": archive.archive_info(root),
            "hierarchical_dir": (
                str(hierarchical_dir) if hierarchical_dir is not None else None
            ),
        }
        training_path = args.source_dir.resolve() / "ffbridge_training_data_df.parquet"
        sessions_dir = args.source_dir.resolve() / "competitions" / "sessions"
        if training_path.is_file() and sessions_dir.is_dir():
            pipeline = _load_quality_pipeline()
            report["raw_source"] = pipeline.audit_historical_cache(
                args.source_dir.resolve(),
                training_path,
                args.cutoff,
            ).to_dict()["summary"]
        else:
            report["raw_source"] = {
                "available": False,
                "training_parquet": str(training_path),
                "sessions_dir": str(sessions_dir),
            }
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0
    if hierarchical_dir is not None:
        if args.hierarchical_layout_seed is None:
            raise ValueError(
                "--hierarchical-layout-seed is required with --hierarchical-dir"
            )
        normalized.initialize_hierarchical_layout(
            hierarchical_dir,
            args.hierarchical_layout_seed.resolve(),
        )
    if args.from_cache:
        result: dict[str, Any] = archive.copy_cache_to_archive(
            sorted(args.cache_dir.resolve().glob("df-*.parquet")),
            root,
        )
    else:
        result = backfill_raw(
            args.source_dir.resolve(),
            root,
            args.cutoff,
            since=args.since,
            limit=args.limit,
            fetch_missing=args.fetch_missing,
            hierarchical_dir=hierarchical_dir,
            workers=args.workers,
            commit_batch_size=args.commit_batch_size,
        )
    result["indexes"] = archive.rebuild_indexes(root)
    if args.skip_compact:
        result["compaction"] = "skipped"
    else:
        result["compaction"] = archive.compact_archive(
            root, force=args.force_compact
        )
    if hierarchical_dir is not None and not args.skip_compact:
        result["hierarchical_compaction"] = (
            normalized.compact_hierarchical_archive(
                hierarchical_dir,
                force=args.force_compact,
            )
        )
    result["archive"] = archive.archive_info(root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=pathlib.Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--archive-dir", type=pathlib.Path, default=None)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("cache"))
    parser.add_argument("--cutoff", type=_parse_date, default=date.today())
    parser.add_argument(
        "--since",
        type=_parse_date,
        default=None,
        help="Only archive sessions on or after this date (YYYY-MM-DD).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for session augmentation and fragment creation.",
    )
    parser.add_argument(
        "--commit-batch-size",
        type=int,
        default=50,
        help="Prepared sessions committed per atomic manifest update.",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Migrate existing player caches instead of rebuilding from raw JSON.",
    )
    parser.add_argument("--force-compact", action="store_true")
    parser.add_argument(
        "--skip-compact",
        action="store_true",
        help="Skip flat and hierarchical compaction after the backfill.",
    )
    parser.add_argument("--hierarchical-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--hierarchical-layout-seed",
        type=pathlib.Path,
        default=None,
        help="Validated normalized metadata.json used as the fixed layout.",
    )
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Fetch audit-reported missing ranking/team JSON before backfill.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.commit_batch_size < 1:
        raise ValueError("--commit-batch-size must be positive")
    if args.skip_compact and args.force_compact:
        raise ValueError("--skip-compact and --force-compact are mutually exclusive")
    if args.since is not None and args.since > args.cutoff:
        raise ValueError("--since must not be after --cutoff")
    if args.from_cache and args.hierarchical_dir is not None:
        raise ValueError(
            "--from-cache cannot build the authoritative hierarchical archive; "
            "use the raw source"
        )

    started = datetime.now()
    clock = time.perf_counter()
    print(f"[archive-builder] start {started.isoformat(timespec='seconds')}", flush=True)
    root = archive.resolve_archive_dir(args.archive_dir).resolve()
    try:
        return _execute(args, root)
    finally:
        elapsed = time.perf_counter() - clock
        print(
            f"[archive-builder] end {datetime.now().isoformat(timespec='seconds')} "
            f"(elapsed {elapsed:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
