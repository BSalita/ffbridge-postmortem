"""Headless access to cached FFBridge postmortem dataframes.

The Streamlit app (ffbridge_streamlit.py) persists each fully augmented
board-results dataframe to cache/df-{session_id}-{player_id}.parquet in
_finalize_mldf_for_report. This module is the shared, Streamlit-free core used
by ffbridge_postmortem_mcp_server.py: it enumerates those parquets, re-derives
the player personalization columns (Boards_I_Played etc. -- same logic as
filter_dataframe in ffbridge_streamlit.py), and runs DuckDB SQL against the
dataframe registered as 'self', mirroring how the app's SQL favorites work.

Env:
  FFBRIDGE_POSTMORTEM_CACHE_DIR  cache directory (default ./cache next to this file)
"""

import os
import pathlib
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import polars as pl

_APP_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = pathlib.Path(os.environ.get("FFBRIDGE_POSTMORTEM_CACHE_DIR", str(_APP_DIR / "cache")))

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


def list_cached_postmortems(player_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Cached postmortems (newest file first), optionally for one player."""
    out: List[Dict[str, Any]] = []
    if not CACHE_DIR.is_dir():
        return out
    for f in CACHE_DIR.glob("df-*.parquet"):
        parsed = _parse_cache_filename(f.name)
        if parsed is None:
            continue
        if player_id is not None and parsed["player_id"] != str(player_id):
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
    out.sort(key=lambda d: d["cached_at"], reverse=True)
    return out


def dataset_info() -> Dict[str, Any]:
    cached = list_cached_postmortems()
    return {
        "cache_dir": str(CACHE_DIR),
        "cached_postmortems": len(cached),
        "players": sorted({c["player_id"] for c in cached}),
        "note": (
            "Postmortems are produced on demand by the Streamlit app "
            "(https://ffbridge.postmortem.chat/?player_id=<FFBridge id>); this "
            "service reads its parquet cache."
        ),
    }


def _resolve_cache_file(player_id: str, session_id: Optional[str] = None) -> pathlib.Path:
    cached = list_cached_postmortems(player_id)
    if not cached:
        raise FileNotFoundError(
            f"No cached postmortem for player {player_id}. Generate one first by "
            f"loading https://ffbridge.postmortem.chat/?player_id={player_id} (add "
            f"&session_id=... for a specific game)."
        )
    if session_id is None:
        return CACHE_DIR / cached[0]["file"]  # newest cache file
    for c in cached:
        if c["session_id"] == str(session_id):
            return CACHE_DIR / c["file"]
    raise FileNotFoundError(
        f"No cached postmortem for player {player_id} session {session_id}. "
        f"Cached sessions: {[c['session_id'] for c in cached]}"
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


def personalize(df: pl.DataFrame, player_id: str) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Add the player-centric flag columns exactly as filter_dataframe does
    (the Pair_Direction branch: cached mldfs are reduced to the core column
    set before augmentation, so the lineup_* Lancelot columns are absent)."""
    pid = str(player_id)
    for player_direction, pair_direction, partner_direction, opponent_pair_direction in _SEAT_TUPLES:
        rows = df.filter(pl.col(f"Player_ID_{player_direction}").cast(pl.Utf8) == pid)
        if rows.height == 0:
            continue
        partner_id = rows[f"Player_ID_{partner_direction}"][0]
        df = df.with_columns(
            pl.col(f"Player_ID_{player_direction}").cast(pl.Utf8).eq(pl.lit(pid)).alias("Boards_I_Played"),
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


def load_postmortem(player_id: str, session_id: Optional[str] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Load a cached postmortem (latest session when session_id is None) and
    personalize it for the player. Returns (df, meta)."""
    path = _resolve_cache_file(str(player_id), session_id)
    parsed = _parse_cache_filename(path.name)
    df = _read_parquet_cached(path)
    df, meta = personalize(df, str(player_id))
    meta["session_id"] = parsed["session_id"]
    meta["cache_file"] = path.name
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
