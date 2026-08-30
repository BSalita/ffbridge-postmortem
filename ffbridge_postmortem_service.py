"""Headless access to cached FFBridge postmortem dataframes.

Postmortem parquets (cache/df-{session_id}-{player_id}.parquet) are written by
ffbridge_postmortem_create -- the same Lancelot + augment path Streamlit uses.
This module enumerates those parquets, re-derives the player personalization
columns (Boards_I_Played etc. -- same logic as filter_dataframe in
ffbridge_streamlit.py), and runs DuckDB SQL against the dataframe registered
as 'self'. Generation (list source sessions / write cache) lives in
ffbridge_postmortem_create.py.

Env:
  FFBRIDGE_POSTMORTEM_CACHE_DIR  cache directory (default ./cache next to this file)
"""

import json
import os
import pathlib
import re
import threading
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import polars as pl

import ffbridge_postmortem_archive as archive
import ffbridge_postmortem_create as create
import ffbridge_postmortem_normalized as normalized
import ffbridge_player_game_service as player_games

_APP_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = pathlib.Path(os.environ.get("FFBRIDGE_POSTMORTEM_CACHE_DIR", str(_APP_DIR / "cache")))
HIERARCHICAL_DIR = normalized.resolve_hierarchical_dir(CACHE_DIR)

CON_REGISTER_NAME = "self"
DEFAULT_SQL_ROW_LIMIT = 500
MAX_SQL_ROW_LIMIT = 2000
MAX_SCHEMA_COLUMNS = 1000

# df-{session_id}-{player_id}.parquet. FFBridge (Lancelot) player ids never
# contain dashes, so the player id is the trailing dash-free token.
_CACHE_FILE_RE = re.compile(r"^df-(?P<session_id>.+)-(?P<player_id>[^-]+)\.parquet$")

# (player_direction, pair_direction, partner_direction, opponent_pair_direction)
# ffbridge_streamlit.py keeps directions as seat letters (see filter_dataframe),
# so the SQL macros substitute seat letters here.
_SEAT_TUPLES = (
    ("N", "NS", "S", "EW"),
    ("S", "NS", "N", "EW"),
    ("E", "EW", "W", "NS"),
    ("W", "EW", "E", "NS"),
)

# Default column set for the per-board summary tool; intersected with the
# actual dataframe columns since older caches may predate some augmentations.
BOARD_SUMMARY_COLUMNS = [
    "Board", "Contract", "Declarer_Direction", "Declarer_ID", "Declarer_Name",
    "Result", "Tricks", "Score_NS", "Score_EW", "Pct_NS", "Pct_EW",
    "MP_NS", "MP_EW", "MP_Top", "Par_NS", "ParContract",
    "Pair_Number_NS", "Pair_Number_EW", "PBN",
]


def _parse_cache_filename(name: str) -> Optional[Dict[str, str]]:
    m = _CACHE_FILE_RE.match(name)
    if m is None:
        return None
    return {"session_id": m.group("session_id"), "player_id": m.group("player_id")}


def _player_id_aliases(player_id: str) -> List[str]:
    """Lancelot id and license number for the same person, when resolvable."""
    pid = str(player_id)
    aliases = {pid}
    try:
        resolved = create.resolve_player(pid)
        aliases.update(resolved.aliases())
    except Exception:
        pass
    return list(aliases)


def player_match_ids(*ids: Optional[Any]) -> List[str]:
    """Deduped identifier strings that all name the same player.

    Accepts a license number, Lancelot person id, Classic/migration id, or a
    mix. Resolution failures keep the raw values so matching still works when
    the dataframe already uses that namespace.
    """
    seen: List[str] = []
    for raw in ids:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.append(text)
    expanded: List[str] = []
    for text in seen:
        for alias in _player_id_aliases(text):
            if alias not in expanded:
                expanded.append(alias)
    return expanded


def list_cached_postmortems(player_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Cached postmortems (newest file first), optionally for one player.

    player_id may be a Lancelot person id or an FFBridge license number; both
    resolve to the same cache files when they name the same person.
    """
    out: List[Dict[str, Any]] = []
    wanted = None if player_id is None else set(_player_id_aliases(str(player_id)))
    if not CACHE_DIR.is_dir():
        return out
    for f in CACHE_DIR.glob("df-*.parquet"):
        parsed = _parse_cache_filename(f.name)
        if parsed is None:
            continue
        if wanted is not None and parsed["player_id"] not in wanted:
            continue
        stat = f.stat()
        out.append(
            {
                "player_id": parsed["player_id"],
                "session_id": parsed["session_id"],
                "file": f.name,
                "size_bytes": stat.st_size,
                "cached_at": stat.st_mtime,
            }
        )
    if player_id is None:
        archived = archive.latest_manifest()
    else:
        archived = archive.archived_sessions_for_player(
            player_match_ids(str(player_id))
        )
    existing = {(row["session_id"], row.get("player_id")) for row in out}
    for row in archived.iter_rows(named=True):
        archive_row = {
            "player_id": str(player_id) if player_id is not None else None,
            "session_id": row["session_id"],
            "file": row["fragment_path"],
            "size_bytes": row.get("size_bytes"),
            "cached_at": row.get("archived_at"),
            "source": "archive",
            "revision": row.get("revision"),
        }
        key = (archive_row["session_id"], archive_row["player_id"])
        if key not in existing:
            out.append(archive_row)
    if player_id is None and HIERARCHICAL_DIR is not None:
        hierarchical = normalized.latest_hierarchical_manifest(HIERARCHICAL_DIR)
        existing_sessions = {row["session_id"] for row in out}
        for row in hierarchical.iter_rows(named=True):
            session_id = str(row["session_id"])
            if session_id in existing_sessions:
                continue
            out.append(
                {
                    "player_id": None,
                    "session_id": session_id,
                    "file": row.get("results_path"),
                    "size_bytes": None,
                    "cached_at": row.get("archived_at"),
                    "source": "hierarchical_archive",
                    "revision": row.get("revision"),
                }
            )
            existing_sessions.add(session_id)
    out.sort(key=lambda d: str(d["cached_at"]), reverse=True)
    return out


def dataset_info() -> Dict[str, Any]:
    cached = list_cached_postmortems()
    return {
        "cache_dir": str(CACHE_DIR),
        "cached_postmortems": len(cached),
        "players": sorted(
            {c["player_id"] for c in cached if c["player_id"] is not None}
        ),
        "archive": archive.archive_info(),
        "hierarchical_archive": normalized.hierarchical_info(HIERARCHICAL_DIR),
        "generate": {
            "tool": "ffbridge_postmortem_generate",
            "list_source_sessions_tool": "ffbridge_postmortem_list_source_sessions",
            "status_tool": "ffbridge_postmortem_generate_status",
            "writer_health_tool": "ffbridge_postmortem_writer_health",
            "player_id": (
                "Lancelot person id, Classic/migration id, or FFBridge license. "
                "Optional explicit prefixes: lancelot:, classic:, license:."
            ),
        },
        "note": (
            "Postmortems are produced on demand by ffbridge_postmortem_generate "
            "(same Lancelot + augment path as Streamlit) and written to this "
            "cache. List playable sessions with "
            "ffbridge_postmortem_list_source_sessions. Check writer readiness "
            "without generating with ffbridge_postmortem_writer_health. Streamlit "
            "(https://ffbridge.postmortem.chat) reads and writes the same cache."
        ),
    }


def resolve_cache_file(player_id: str, session_id: Optional[str] = None) -> pathlib.Path:
    path, _is_archive, _cache_player_id = _resolve_postmortem_file(
        player_id, session_id
    )
    return path


def _resolve_postmortem_file(
    player_id: str,
    session_id: Optional[str] = None,
) -> Tuple[pathlib.Path, bool, Optional[str]]:
    if session_id is not None:
        try:
            return archive.resolve_archived_session(session_id), True, None
        except FileNotFoundError:
            pass
    else:
        indexed = archive.archived_sessions_for_player(
            player_match_ids(str(player_id))
        )
        if indexed.height:
            path = archive.resolve_archive_dir() / indexed["fragment_path"][0]
            if path.is_file():
                return path, True, None
    path = _resolve_cache_file(player_id, session_id)
    parsed = _parse_cache_filename(path.name)
    return path, False, parsed["player_id"] if parsed is not None else None


def _resolve_cache_file(player_id: str, session_id: Optional[str] = None) -> pathlib.Path:
    cached = list_cached_postmortems(player_id)
    if not cached:
        raise FileNotFoundError(
            f"No cached postmortem for player {player_id}. Generate one with "
            f"ffbridge_postmortem_generate(player_id={player_id!r}) "
            f"(optional session_id / date_from / date_to)."
        )
    if session_id is None:
        return CACHE_DIR / cached[0]["file"]  # newest cache file
    for c in cached:
        if c["session_id"] == str(session_id):
            return CACHE_DIR / c["file"]
    raise FileNotFoundError(
        f"No cached postmortem for player {player_id} session {session_id}. "
        f"Cached sessions: {[c['session_id'] for c in cached]}. "
        f"Generate with ffbridge_postmortem_generate(player_id={player_id!r}, "
        f"session_id={session_id!r})."
    )


# Small in-process cache: one postmortem parquet is ~10^4 rows x ~10^3 columns,
# cheap enough to keep a few resident keyed by (path, mtime).
_df_cache: Dict[Tuple[str, float], pl.DataFrame] = {}
_df_cache_lock = threading.Lock()
_DF_CACHE_MAX = 4


def _read_parquet_cached(path: pathlib.Path) -> pl.DataFrame:
    key = (str(path), path.stat().st_mtime)
    with _df_cache_lock:
        if key in _df_cache:
            return _df_cache[key]
    df = pl.read_parquet(path)
    with _df_cache_lock:
        if len(_df_cache) >= _DF_CACHE_MAX:
            _df_cache.pop(next(iter(_df_cache)))
        _df_cache[key] = df
    return df


def personalize(
    df: pl.DataFrame,
    player_id: str,
    extra_ids: Optional[List[Any]] = None,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Add the player-centric flag columns exactly as filter_dataframe does
    (the Pair_Direction branch: cached mldfs are reduced to the core column
    set before augmentation, so the lineup_* Lancelot columns are absent).

    ``player_id`` may be a license, Lancelot id, or Classic id. Matching uses
    every resolved alias plus any extra ids (e.g. the cache-filename id).
    """
    pid = str(player_id)
    match_ids = player_match_ids(pid, *(extra_ids or []))
    for player_direction, pair_direction, partner_direction, opponent_pair_direction in _SEAT_TUPLES:
        id_col = pl.col(f"Player_ID_{player_direction}").cast(pl.Utf8)
        rows = df.filter(id_col.is_in(match_ids))
        if rows.height == 0:
            continue
        partner_id = rows[f"Player_ID_{partner_direction}"][0]
        matched_player_id = str(rows[f"Player_ID_{player_direction}"][0])
        df = df.with_columns(
            id_col.is_in(match_ids).alias("Boards_I_Played"),
        )
        df = df.with_columns(
            pl.col("Boards_I_Played").and_(pl.col("Declarer_Direction").eq(player_direction)).alias("Boards_I_Declared"),
            pl.col("Boards_I_Played").and_(pl.col("Declarer_Direction").eq(partner_direction)).alias("Boards_Partner_Declared"),
        )
        df = df.with_columns(
            pl.col("Boards_I_Played").alias("Boards_We_Played"),
            pl.col("Boards_I_Played").alias("Our_Boards"),
            (pl.col("Boards_I_Declared") | pl.col("Boards_Partner_Declared")).alias("Boards_We_Declared"),
        )
        df = df.with_columns(
            (pl.col("Boards_I_Played") & ~pl.col("Boards_We_Declared") & pl.col("Contract").ne("PASS")).alias("Boards_Opponent_Declared"),
        )
        meta = {
            "player_id": pid,
            "matched_player_id": matched_player_id,
            "player_name": rows[f"Player_Name_{player_direction}"][0] if f"Player_Name_{player_direction}" in rows.columns else None,
            "player_direction": player_direction,
            "partner_id": str(partner_id),
            "partner_name": rows[f"Player_Name_{partner_direction}"][0] if f"Player_Name_{partner_direction}" in rows.columns else None,
            "partner_direction": partner_direction,
            "pair_direction": pair_direction,
            "opponent_pair_direction": opponent_pair_direction,
            "game_date": str(df["Date"].first()) if "Date" in df.columns else None,
        }
        return df, meta
    raise ValueError(f"Player {pid} not found in any Player_ID_[NESW] column of the cached postmortem.")


def _load_hierarchical_postmortem(
    player_id: str, session_id: str
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    if HIERARCHICAL_DIR is None:
        raise FileNotFoundError("Hierarchical postmortem archive is not configured")
    if not (HIERARCHICAL_DIR / "metadata.json").is_file():
        raise FileNotFoundError(
            f"Hierarchical postmortem archive is unavailable: {HIERARCHICAL_DIR}"
        )
    if not normalized.hierarchical_has_session(HIERARCHICAL_DIR, session_id):
        raise FileNotFoundError(
            f"Hierarchical archive has no session {session_id}"
        )
    metadata = json.loads(
        (HIERARCHICAL_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    mapping = metadata.get("column_mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise FileNotFoundError(
            f"Hierarchical archive metadata lacks column_mapping: {HIERARCHICAL_DIR}"
        )
    columns = [
        column
        for column in mapping
        if column not in {"session_id", "Board", "_result_row_id"}
    ]
    frame = normalized.normalized_player_report(
        HIERARCHICAL_DIR,
        session_id=str(session_id),
        player_ids=player_match_ids(str(player_id)),
        columns=columns,
        only_player_rows=False,
    )
    if frame.is_empty():
        raise FileNotFoundError(
            f"Hierarchical archive has no rows for session {session_id}"
        )
    frame, meta = personalize(frame, str(player_id))
    frame = archive.restore_pair_direction(frame, meta["pair_direction"])
    meta.update(
        {
            "session_id": str(session_id),
            "cache_file": None,
            "data_source": "hierarchical_archive",
            "archive_file": None,
            "requested_id": str(player_id),
            "cache_player_id": None,
            "hierarchical_dir": str(HIERARCHICAL_DIR),
        }
    )
    return frame, meta


def load_postmortem(player_id: str, session_id: Optional[str] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Load a cached postmortem (latest session when session_id is None) and
    personalize it for the player. Returns (df, meta)."""
    if session_id is not None:
        try:
            return _load_hierarchical_postmortem(str(player_id), str(session_id))
        except FileNotFoundError:
            pass
    path, is_archive, cache_player_id = _resolve_postmortem_file(
        str(player_id), session_id
    )
    df = _read_parquet_cached(path)
    # Cache files are named with the Lancelot person id. Personalize with the
    # requested id plus the filename id so license 9500754 and Lancelot 246273
    # both match whichever namespace Player_ID_* actually stores.
    df, meta = personalize(
        df,
        str(player_id),
        extra_ids=[cache_player_id] if cache_player_id is not None else None,
    )
    if is_archive:
        df = archive.restore_pair_direction(df, meta["pair_direction"])
        resolved_session_id = str(session_id) if session_id is not None else str(
            archive.archived_sessions_for_player(
                player_match_ids(str(player_id))
            )["session_id"][0]
        )
    else:
        parsed = _parse_cache_filename(path.name)
        if parsed is None:
            raise ValueError(f"Invalid postmortem cache filename: {path.name}")
        resolved_session_id = parsed["session_id"]
    meta["session_id"] = resolved_session_id
    meta["cache_file"] = path.name
    meta["data_source"] = "archive" if is_archive else "cache"
    meta["archive_file"] = str(path) if is_archive else None
    meta["requested_id"] = str(player_id)
    meta["cache_player_id"] = cache_player_id
    return df, meta


def process_sql_macros(sql: str, meta: Dict[str, Any]) -> str:
    """Same substitutions as PostmortemBase.process_prompt_macros."""
    for macro, key in (
        ("{Player_Direction}", "player_direction"),
        ("{Partner_Direction}", "partner_direction"),
        ("{Pair_Direction}", "pair_direction"),
        ("{Opponent_Pair_Direction}", "opponent_pair_direction"),
    ):
        value = meta.get(key)
        if value is not None:
            sql = sql.replace(macro, str(value))
    return sql


def run_sql(df: pl.DataFrame, sql: str, meta: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, Any]:
    """Run DuckDB SQL against the postmortem dataframe registered as 'self'."""
    limit = max(1, min(limit or DEFAULT_SQL_ROW_LIMIT, MAX_SQL_ROW_LIMIT))
    sql = process_sql_macros(sql.strip().rstrip(";"), meta)
    # Same convenience as the app's ShowDataFrameTable: allow DuckDB's
    # FROM-first syntax by prepending the table when it is not referenced.
    if f"from {CON_REGISTER_NAME}" not in sql.lower():
        sql = f"FROM {CON_REGISTER_NAME} " + sql
    # external access off: only the registered dataframe is queryable.
    con = duckdb.connect(config={"enable_external_access": "false"})
    try:
        con.register(CON_REGISTER_NAME, df)
        result = con.execute(sql).pl()
    finally:
        con.close()
    truncated = result.height > limit
    result = result.head(limit)
    return {
        "sql": sql,
        "columns": result.columns,
        "rows": result.to_dicts(),
        "row_count": result.height,
        "truncated": truncated,
    }


def board_results(
    df: pl.DataFrame,
    meta: Dict[str, Any],
    only_my_boards: bool = True,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Per-board rows, defaulting to the boards the player actually played."""
    limit = max(1, min(limit or DEFAULT_SQL_ROW_LIMIT, MAX_SQL_ROW_LIMIT))
    if only_my_boards and "Boards_I_Played" in df.columns:
        df = df.filter(pl.col("Boards_I_Played"))
    wanted = columns or BOARD_SUMMARY_COLUMNS
    missing = [c for c in wanted if c not in df.columns]
    selected = [c for c in wanted if c in df.columns]
    if not selected:
        raise ValueError(f"None of the requested columns exist. Missing: {missing}")
    if "Board" in df.columns:
        df = df.sort("Board")
    df = df.select(selected).head(limit)
    return {
        "meta": meta,
        "columns": selected,
        "missing_columns": missing,
        "rows": df.to_dicts(),
        "row_count": df.height,
    }


def hierarchical_board_results(
    player_id: str,
    session_id: str,
    *,
    only_my_boards: bool = True,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Produce a report from projected hierarchical board/result leaves."""
    if HIERARCHICAL_DIR is None:
        raise FileNotFoundError("Hierarchical postmortem archive is not configured")
    if not (HIERARCHICAL_DIR / "metadata.json").is_file():
        raise FileNotFoundError(
            f"Hierarchical postmortem archive is unavailable: {HIERARCHICAL_DIR}"
        )
    if not normalized.hierarchical_has_session(HIERARCHICAL_DIR, session_id):
        raise FileNotFoundError(
            f"Hierarchical archive has no session {session_id}"
        )
    match_ids = player_match_ids(str(player_id))
    wanted = columns or BOARD_SUMMARY_COLUMNS
    context_columns = [
        *[f"Player_ID_{seat}" for seat in "NESW"],
        *[f"Player_Name_{seat}" for seat in "NESW"],
        "Declarer_Direction",
        "Contract",
        "Date",
    ]
    selected = list(dict.fromkeys([*wanted, *context_columns]))
    frame = normalized.normalized_player_report(
        HIERARCHICAL_DIR,
        session_id=str(session_id),
        player_ids=match_ids,
        columns=selected,
        only_player_rows=only_my_boards,
    )
    frame, meta = personalize(frame, str(player_id))
    frame = archive.restore_pair_direction(frame, meta["pair_direction"])
    meta.update(
        {
            "session_id": str(session_id),
            "data_source": "hierarchical_archive",
            "requested_id": str(player_id),
            "hierarchical_dir": str(HIERARCHICAL_DIR),
        }
    )
    return board_results(
        frame,
        meta,
        only_my_boards=only_my_boards,
        columns=wanted,
        limit=limit,
    )


def schema_columns(df: pl.DataFrame, pattern: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Column names (with dtypes) of the augmented postmortem dataframe,
    optionally filtered by a case-insensitive regex. The frame has thousands
    of columns, hence the cap."""
    limit = max(1, min(limit or MAX_SCHEMA_COLUMNS, MAX_SCHEMA_COLUMNS))
    names = sorted(df.columns)
    if pattern:
        rx = re.compile(pattern, re.IGNORECASE)
        names = [c for c in names if rx.search(c)]
    truncated = len(names) > limit
    names = names[:limit]
    dtypes = dict(zip(df.columns, (str(t) for t in df.dtypes)))
    return {
        "total_columns": df.width,
        "matched_columns": len(names),
        "truncated": truncated,
        "columns": {c: dtypes[c] for c in names},
    }


def archive_rows(
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    series_id: Optional[str] = None,
    session_id: Optional[str] = None,
    player_id: Optional[str] = None,
    columns: Optional[List[str]] = None,
    limit: int = DEFAULT_SQL_ROW_LIMIT,
) -> Dict[str, Any]:
    """Read bounded historical rows with filters pushed into Parquet scans."""
    limit = max(1, min(limit, MAX_SQL_ROW_LIMIT))
    files = archive.dataset_files()
    if not files:
        raise FileNotFoundError(
            "The FFBridge postmortem analytics dataset has not been compacted"
        )
    requested = columns or BOARD_SUMMARY_COLUMNS
    mandatory = ["Date", "session_id"]
    if player_id is not None:
        mandatory.extend(f"Player_ID_{seat}" for seat in "NESW")
    wanted = list(dict.fromkeys([*requested, *mandatory]))
    lazy_frames: list[pl.LazyFrame] = []
    for path in files:
        schema = pl.scan_parquet(path).collect_schema()
        available = [column for column in wanted if column in schema]
        lazy_frames.append(pl.scan_parquet(path).select(available))
    query = pl.concat(lazy_frames, how="diagonal_relaxed")
    if date_from is not None:
        query = query.filter(pl.col("Date").cast(pl.Date) >= date.fromisoformat(date_from))
    if date_to is not None:
        query = query.filter(pl.col("Date").cast(pl.Date) <= date.fromisoformat(date_to))
    if series_id is not None:
        if "series_id" not in query.collect_schema():
            raise ValueError("Archive dataset does not contain series_id")
        query = query.filter(pl.col("series_id").cast(pl.String) == str(series_id))
    if session_id is not None:
        query = query.filter(pl.col("session_id").cast(pl.String) == str(session_id))
    if player_id is not None:
        match_ids = player_match_ids(player_id)
        query = query.filter(
            pl.any_horizontal(
                *(
                    pl.col(f"Player_ID_{seat}").cast(pl.String).is_in(match_ids)
                    for seat in "NESW"
                )
            )
        )
    result = query.select(
        [column for column in requested if column in query.collect_schema()]
    ).limit(limit + 1).collect(engine="streaming")
    truncated = result.height > limit
    result = result.head(limit)
    return {
        "columns": result.columns,
        "rows": result.to_dicts(),
        "row_count": result.height,
        "truncated": truncated,
        "filters": {
            "date_from": date_from,
            "date_to": date_to,
            "series_id": series_id,
            "session_id": session_id,
            "player_id": player_id,
        },
    }


def list_source_sessions(
    player_id: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    return create.list_source_sessions(
        player_id,
        date_from=date_from,
        date_to=date_to,
        cache_dir=CACHE_DIR,
    )


def last_game(
    player: str,
    clubs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return player_games.last_game(player, clubs)


def played_today(
    player: str,
    clubs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return player_games.played_today(player, clubs)


def generate_postmortems(
    player_id: str,
    session_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    force: bool = False,
    continue_on_error: Optional[bool] = None,
) -> Dict[str, Any]:
    return create.generate_postmortems(
        player_id,
        session_id=session_id,
        date_from=date_from,
        date_to=date_to,
        force=force,
        continue_on_error=continue_on_error,
        cache_dir=CACHE_DIR,
    )


def generate_status(job_id: str) -> Dict[str, Any]:
    return create.generate_status(job_id)
