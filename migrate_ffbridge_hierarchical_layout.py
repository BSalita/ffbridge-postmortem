"""Migrate the FFBridge hierarchical archive from layout version 3 to 4.

Adds the opening-lead columns to every results fragment: Lead from the raw
Lancelot team scores (the stage 4 source), DD_Lead_Cost / Lead_Is_DD_Best
recomputed with mlBridge add_dd_lead_cost, and the SD_Lead_* set as nulls
(stage 4 does not compute it). Boards fragments are hard-linked unchanged.

Run with the elo venv (needs endplay and the ddss DLL), with the postmortem
container and stage 4 stopped. The archive's metadata.json is also the stage 4
layout seed, so it is migrated in place. Afterwards rebuild the compacted
dataset with src/elo/compact_ffbridge_hierarchical_domains.py.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import time
from datetime import datetime
from typing import Any, Mapping

import polars as pl

import build_ffbridge_postmortem_archive as builder
import ffbridge_postmortem_normalized as normalized
from mlBridge.mlBridgeAugmentLib import add_dd_lead_cost


PLAYER_ID_COLUMNS = tuple(f"Player_ID_{seat}" for seat in normalized.SEATS)
JOIN_KEYS = ("Board", *PLAYER_ID_COLUMNS)


@functools.cache
def _complete_sessions(source_dir_text: str) -> dict[str, tuple[str, ...]]:
    """Stage 1's audit manifest: complete session id -> team ids."""
    pipeline = builder._load_quality_pipeline()
    path = pipeline.default_audit_manifest_path(pathlib.Path(source_dir_text))
    sessions = pipeline._load_audit_manifest(path)
    if not sessions:
        raise FileNotFoundError(f"Audit manifest missing or empty: {path}")
    return sessions


class RawLeadScorer:
    """Lead per stored result from the raw team scores, scored double dummy."""

    def __init__(self, source_dir: pathlib.Path) -> None:
        self.source_dir = str(source_dir)

    def __call__(self, row: Mapping[str, Any], inputs: pl.DataFrame) -> pl.DataFrame:
        pipeline = builder._load_quality_pipeline()
        session_id = str(row["session_id"])
        team_ids = _complete_sessions(self.source_dir).get(session_id)
        if team_ids is None:
            raise KeyError(f"Session {session_id} is not a complete raw session")
        session = pipeline.SessionAudit(
            session_id=session_id,
            session_date=str(row["Date"]),
            metadata_path="",
            in_training=False,
            ranking_present=True,
            expected_team_ids=team_ids,
            present_team_ids=team_ids,
            missing_team_ids=(),
        )
        raw, _unmapped = pipeline.load_raw_session(pathlib.Path(self.source_dir), session)
        # The same preparation stage 4 applies before augmenting.
        prepared = builder._prepare_raw_session(raw, row["Date"])
        leads = prepared.select(
            pl.col("Board").cast(pl.Int64),
            *[pl.col(column).cast(pl.String) for column in PLAYER_ID_COLUMNS],
            pl.col("Lead").cast(pl.String),
            pl.lit(True).alias("_matched"),
        )
        joined = inputs.with_columns(
            pl.col("Board").cast(pl.Int64),
            *[pl.col(column).cast(pl.String) for column in PLAYER_ID_COLUMNS],
        ).join(
            leads,
            on=list(JOIN_KEYS),
            how="left",
            validate="m:1",
            nulls_equal=True,
            maintain_order="left",
        )
        unmatched = int(joined["_matched"].is_null().sum())
        if unmatched:
            raise ValueError(
                f"Session {session_id}: {unmatched} of {joined.height} stored results "
                "have no raw score row"
            )
        scored = add_dd_lead_cost(
            joined.select("_result_row_id", "PBN", "BidSuit", "Declarer_Direction", "Lead")
        )
        return scored.select("_result_row_id", *normalized.LEAD_RESULT_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hierarchical-dir",
        type=pathlib.Path,
        required=True,
        help="Archive directory to migrate in place.",
    )
    parser.add_argument(
        "--source-dir",
        type=pathlib.Path,
        default=builder.DEFAULT_SOURCE_DIR,
        help="Raw FFBridge source (results/sessions, results/teams).",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--delete-old-fragments",
        action="store_true",
        help="Delete the layout_version=3 fragment files after success.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    directory = args.hierarchical_dir.resolve()
    source_dir = args.source_dir.resolve()
    started = datetime.now()
    clock = time.perf_counter()
    print(f"Started {started:%Y-%m-%d %H:%M:%S}: migrating {directory}", flush=True)

    manifest = normalized.read_hierarchical_manifest(directory)
    complete = _complete_sessions(str(source_dir))
    missing = sorted(set(manifest["session_id"].to_list()) - set(complete))
    if missing:
        raise KeyError(
            f"{len(missing)} archived sessions are not complete raw sessions in "
            f"{source_dir}; first: {missing[:10]}"
        )
    # Workers inherit this at spawn; ddss reads it when it loads.
    if "DDSS_MAX_THREADS" not in os.environ:
        os.environ["DDSS_MAX_THREADS"] = str(
            max(1, (os.cpu_count() or 1) // args.workers)
        )
    summary = normalized.migrate_hierarchical_layout(
        directory,
        RawLeadScorer(source_dir),
        workers=args.workers,
        delete_old_fragments=args.delete_old_fragments,
    )
    print(json.dumps(summary, indent=2, default=str))
    print(
        "Rebuild the compacted dataset: "
        "src/elo/compact_ffbridge_hierarchical_domains.py",
        flush=True,
    )
    print(
        f"Ended {datetime.now():%Y-%m-%d %H:%M:%S} "
        f"(elapsed {time.perf_counter() - clock:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
