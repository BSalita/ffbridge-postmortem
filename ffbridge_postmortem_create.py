"""Streamlit-free FFBridge postmortem creation (Lancelot + augment + cache).

Streamlit (ffbridge_streamlit.py) and the MCP server both call this module so
they write the same ``cache/df-{session_id}-{player_id}.parquet`` files. The
Lancelot fetch / convert / column-reduce / AllAugmentations path is the one
the Streamlit app uses; this file just drops ``st.session_state``.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

_APP_DIR = pathlib.Path(__file__).resolve().parent
_REQUIRED_LIBS = ("mlBridge",)


def _is_lib_dir(p: pathlib.Path) -> bool:
    return p.is_dir() and (p / "__init__.py").is_file()


def _ensure_mlbridge_on_path() -> None:
    candidates = (
        _APP_DIR / "mlBridge",
        _APP_DIR.parent / "mlBridge",
        _APP_DIR.parent.parent / "mlBridge",
    )
    found = next((p for p in candidates if _is_lib_dir(p)), None)
    if found is None:
        raise FileNotFoundError(
            "mlBridge not found at " + " or ".join(str(p) for p in candidates)
        )
    for root in (_APP_DIR, found.parent, found):
        s = str(root)
        if s not in sys.path:
            sys.path.append(s)


_ensure_mlbridge_on_path()

import polars as pl

import mlBridge.mlBridgeFFLib as mlBridgeFFLib
import mlBridge.mlBridgeFFIndexLib as mlBridgeFFIndexLib
from mlBridge.mlBridgeAugmentLib import AllAugmentations

DEFAULT_CACHE_DIR = pathlib.Path(
    os.environ.get("FFBRIDGE_POSTMORTEM_CACHE_DIR", str(_APP_DIR / "cache"))
)
DEFAULT_SD_PRODUCTIONS = 10

# Same column set _finalize_mldf_for_report keeps before AllAugmentations.
CORE_MLDF_COLUMNS = [
    "Date",
    "Section_Name",
    "Board",
    "PBN",
    "Pair_Direction",
    "Dealer",
    "Vul",
    "Declarer",
    "Contract",
    "Result",
    "Score_EW",
    "Score_NS",
    "Pct_NS",
    "Pct_EW",
    "MP_NS",
    "MP_EW",
    "MP_Top",
    "Pair_Number_NS",
    "Pair_Number_EW",
    "Player_ID_N",
    "Player_ID_E",
    "Player_ID_S",
    "Player_ID_W",
    "Player_Name_N",
    "Player_Name_E",
    "Player_Name_S",
    "Player_Name_W",
]

TeamProgress = Callable[[Sequence[Any]], Iterable[Any]]


def _log(msg: str) -> None:
    print(f"[ffbridge-postmortem-create] {msg}", flush=True)


def _elapsed_if_slow(label: str, started: float, threshold_s: float = 30.0) -> None:
    elapsed = time.time() - started
    if elapsed > threshold_s:
        _log(f"{label} elapsed {elapsed:.1f}s")


def cache_parquet_path(
    session_id: Any,
    player_id: str,
    cache_dir: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    directory = pathlib.Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return directory / f"df-{session_id}-{player_id}.parquet"


def json_normalize_api(json_data: Any, separator: str = "_") -> pl.DataFrame:
    """Flatten Lancelot ranking/score payloads the way Streamlit did via pandas."""
    return pl.json_normalize(json_data, separator=separator)


def _as_int_str(value: Any) -> Optional[str]:
    return None if value is None else str(int(value))


def _norm_digits(value: Any) -> str:
    return str(value or "").strip().lstrip("0") or "0"


# ---------------------------------------------------------------------------
# Auth + player resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LancelotAuth:
    token: str
    lancelot_id: str
    license_number: str
    classic_person_id: Optional[str] = None


@dataclass(frozen=True)
class ResolvedPlayer:
    lancelot_id: str
    license_number: Optional[str]
    requested_id: str
    classic_person_id: Optional[str] = None

    def aliases(self) -> List[str]:
        out = {self.lancelot_id, self.requested_id}
        if self.license_number:
            out.add(self.license_number)
        if self.classic_person_id:
            out.add(self.classic_person_id)
        return sorted(out)


_auth_lock = threading.Lock()
_auth: Optional[LancelotAuth] = None
_resolve_cache: Dict[str, ResolvedPlayer] = {}


def _firebase_refresh_token() -> str:
    load_dotenv()
    email = os.getenv("FFBRIDGE_EMAIL")
    password = os.getenv("FFBRIDGE_PASSWORD")
    if not email or not password:
        raise ValueError(
            "No Lancelot bearer token and no FFBRIDGE_EMAIL/FFBRIDGE_PASSWORD "
            "in .env; cannot authenticate."
        )
    token = mlBridgeFFLib.firebase_sign_in(email, password)
    try:
        from dotenv import set_key

        set_key(str(_APP_DIR / ".env"), "FFBRIDGE_BEARER_TOKEN_LANCELOT", token)
    except Exception as e:
        _log(f"could not persist refreshed token to .env: {e}")
    return token


def ensure_lancelot_auth(*, force: bool = False) -> LancelotAuth:
    """Return a validated Lancelot session for the .env credentials."""
    global _auth
    with _auth_lock:
        if _auth is not None and not force:
            return _auth
        load_dotenv()
        token = os.getenv("FFBRIDGE_BEARER_TOKEN_LANCELOT")
        persons_me = None
        if token:
            try:
                persons_me = mlBridgeFFLib.get_persons_me(token)
            except Exception as e:
                _log(f"env Lancelot token rejected ({e}); refreshing via Firebase")
                token = None
        if persons_me is None:
            token = _firebase_refresh_token()
            persons_me = mlBridgeFFLib.get_persons_me(token)
        auth = LancelotAuth(
            token=token,
            lancelot_id=str(persons_me["id"]),
            license_number=str(persons_me.get("ffbId") or ""),
            classic_person_id=(
                str(persons_me["migrationId"]) if persons_me.get("migrationId") is not None else None
            ),
        )
        _auth = auth
        _log(
            f"authenticated lancelot_id={auth.lancelot_id} "
            f"license={auth.license_number}"
        )
        return auth


def resolve_player(player_id: str, *, token: Optional[str] = None) -> ResolvedPlayer:
    """Map a Lancelot, Classic/migration, or license id to one player.

    ``9500754`` (license) and ``246273`` (Lancelot id) resolve to the same
    person when they belong to the same Lancelot record. Explicit ``license:``,
    ``classic:``, and ``lancelot:`` prefixes disambiguate numeric collisions.
    """
    requested = str(player_id).strip()
    if not requested:
        raise ValueError("player_id is required")

    try:
        indexed = _resolve_player_from_index(requested)
    except FileNotFoundError:
        indexed = None
    if indexed is not None:
        return indexed

    cached = _resolve_cache.get(requested)
    if cached is not None:
        return cached

    auth = None
    if token is None:
        auth = ensure_lancelot_auth()
        token = auth.token
    else:
        try:
            auth = ensure_lancelot_auth()
        except Exception:
            auth = None

    if auth is not None:
        if requested == auth.lancelot_id or _norm_digits(requested) == _norm_digits(auth.license_number):
            resolved = ResolvedPlayer(
                lancelot_id=auth.lancelot_id,
                license_number=auth.license_number or None,
                requested_id=requested,
                classic_person_id=auth.classic_person_id,
            )
            for key in resolved.aliases():
                _resolve_cache[key] = resolved
            return resolved

    if requested.isdigit():
        items = mlBridgeFFLib.search_persons(requested, token)
        if len(items) == 1:
            item = items[0]
            license_number = str(item.get("ffbId") or "")
            if license_number and _norm_digits(license_number) == _norm_digits(requested):
                resolved = ResolvedPlayer(
                    lancelot_id=str(item["id"]),
                    license_number=license_number,
                    requested_id=requested,
                    classic_person_id=(
                        str(item["migrationId"]) if item.get("migrationId") is not None else None
                    ),
                )
                for key in resolved.aliases():
                    _resolve_cache[key] = resolved
                return resolved

    resolved = ResolvedPlayer(
        lancelot_id=requested,
        license_number=auth.license_number if auth and requested == auth.lancelot_id else None,
        requested_id=requested,
        classic_person_id=auth.classic_person_id if auth and requested == auth.lancelot_id else None,
    )
    _resolve_cache[requested] = resolved
    return resolved


# ---------------------------------------------------------------------------
# Source session list
# ---------------------------------------------------------------------------


def _parse_session_date(value: Any) -> Optional[str]:
    if not value:
        return None
    return str(value)[:10]


def _in_date_window(date_str: Optional[str], date_from: Optional[str], date_to: Optional[str]) -> bool:
    if date_from is None and date_to is None:
        return True
    if not date_str:
        return False
    if date_from and date_str < date_from:
        return False
    if date_to and date_str > date_to:
        return False
    return True


def fetch_logged_in_source_sessions(token: str) -> List[Dict[str, Any]]:
    """All playable Lancelot sessions for the authenticated user (newest first)."""
    items: List[Dict[str, Any]] = []
    page = 1
    while True:
        response = mlBridgeFFLib.lancelot_get(f"results/search/me?currentPage={page}", token=token)
        items.extend(response.get("items", []))
        pagination = response.get("pagination", {})
        if not pagination.get("has_next_page") or page >= 20:
            break
        page += 1

    sessions: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        session = item.get("session") or {}
        session_id = session.get("id")
        if session_id is None or not session.get("hasResult"):
            continue
        sid = str(session_id)
        if sid in seen:
            continue
        seen.add(sid)
        group = item.get("group") or {}
        stade = (group.get("phase") or {}).get("stade") or {}
        organization = stade.get("organization") or {}
        competition = (stade.get("competitionDivision") or {}).get("label") or ""
        date_str = _parse_session_date(item.get("date"))
        session_label = session.get("label") or group.get("label") or competition
        organization_name = organization.get("name") or organization.get("label") or ""
        description = " ".join(part for part in [date_str, session_label, organization_name] if part)
        sessions.append(
            {
                "session_id": sid,
                "date": date_str,
                "club": organization_name,
                "organization_id": organization.get("ffbCode"),
                "organization_name": organization_name,
                "group_id": group.get("id"),
                "competition_label": competition or session_label,
                "session_label": session_label,
                "description": description,
                "raw_date": item.get("date"),
            }
        )
    return sessions


def _find_player_session_index_dir() -> pathlib.Path:
    configured = os.environ.get("FFBRIDGE_PLAYER_SESSION_INDEX_DIR", "").strip()
    if configured:
        directory = pathlib.Path(configured)
        required = mlBridgeFFIndexLib.index_paths(directory)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "FFBRIDGE_PLAYER_SESSION_INDEX_DIR is incomplete; missing: "
                + ", ".join(missing)
            )
        return directory

    candidates = [
        mlBridgeFFIndexLib.default_index_dir(),
        _APP_DIR.parent.parent
        / "Elo_Ratings"
        / "data"
        / "ffbridge"
        / "player_session_index",
    ]
    for directory in candidates:
        if all(path.is_file() for path in mlBridgeFFIndexLib.index_paths(directory)):
            return directory
    raise FileNotFoundError(
        "No shared Lancelot player-session index found. Set "
        "FFBRIDGE_PLAYER_SESSION_INDEX_DIR or build the index under "
        "/data/ffbridge/player_session_index."
    )


def _resolve_player_from_index(player_id: str) -> Optional[ResolvedPlayer]:
    index_dir = _find_player_session_index_dir()
    persons = mlBridgeFFIndexLib.load_persons(index_dir)
    person = mlBridgeFFIndexLib.lookup_person(persons, player_id)
    if person is None:
        return None
    return ResolvedPlayer(
        lancelot_id=str(person["lancelot_person_id"]),
        license_number=(
            str(person["license_number"])
            if person.get("license_number") is not None
            else None
        ),
        requested_id=str(player_id),
        classic_person_id=(
            str(person["classic_person_id"])
            if person.get("classic_person_id") is not None
            else None
        ),
    )


def fetch_other_player_source_sessions(
    lancelot_person_id: str,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Sessions for another player from the shared public-ranking index."""
    index_dir = _find_player_session_index_dir()
    rows = mlBridgeFFIndexLib.query_index_sessions(
        lancelot_person_id,
        date_from=date_from,
        date_to=date_to,
        index_dir=index_dir,
    ).to_dicts()

    sessions: List[Dict[str, Any]] = []
    for row in rows:
        sid = str(row["session_id"])
        date_str = _parse_session_date(row.get("session_date"))
        label = str(row.get("session_label") or "")
        club = str(row.get("club_name") or "")
        sessions.append(
            {
                "session_id": sid,
                "date": date_str,
                "club": club,
                "organization_id": row.get("club_id"),
                "organization_name": club,
                "group_id": None,
                "competition_label": label,
                "session_label": label,
                "description": " ".join(part for part in (date_str, label, club) if part),
                "raw_date": row.get("raw_date"),
                "series_id": row.get("series_id"),
                "team_id": row.get("team_id"),
                "listing_source": "shared Lancelot player-session index",
            }
        )
    return sessions


def list_source_sessions(
    player_id: str,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Playable Lancelot sessions for a player, with already_cached flags."""
    auth = ensure_lancelot_auth()
    token = token or auth.token
    resolved = resolve_player(player_id, token=token)
    directory = pathlib.Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    if resolved.lancelot_id == auth.lancelot_id:
        source_sessions = fetch_logged_in_source_sessions(token)
    else:
        source_sessions = fetch_other_player_source_sessions(
            resolved.lancelot_id,
            date_from=date_from,
            date_to=date_to,
        )
    sessions = []
    for entry in source_sessions:
        if not _in_date_window(entry.get("date"), date_from, date_to):
            continue
        cache_file = cache_parquet_path(entry["session_id"], resolved.lancelot_id, directory)
        row = dict(entry)
        row.pop("raw_date", None)
        row["already_cached"] = cache_file.is_file()
        row["cache_file"] = cache_file.name if cache_file.is_file() else None
        sessions.append(row)
    return {
        "player_id": resolved.lancelot_id,
        "player_license_number": resolved.license_number,
        "requested_id": resolved.requested_id,
        "sessions": sessions,
        "count": len(sessions),
    }


# ---------------------------------------------------------------------------
# Lancelot session -> mldf (same steps as Streamlit get_lancelot_session_mldf)
# ---------------------------------------------------------------------------


@dataclass
class LancelotSessionMeta:
    player_id: str
    session_id: str
    partner_id: Optional[str]
    player_license_number: Optional[str]
    partner_license_number: Optional[str]
    player_name: str
    partner_name: str
    group_id: Any
    org_id: Any
    tournament_date: Optional[str]
    organization_name: Any
    game_description: Any
    team_id: Any
    pair_direction: str
    opponent_pair_direction: str
    player_direction: str
    partner_direction: str
    section_name: Any
    team_number: Any
    game_url: str
    route_url: None = None


@dataclass
class LancelotSessionBuild:
    df: pl.DataFrame
    meta: LancelotSessionMeta
    ranking_df: pl.DataFrame
    scores_df: pl.DataFrame


def _fetch_lancelot_json_cached(
    path: str,
    cache_relpath: str,
    token: Optional[str],
    cache_dir: pathlib.Path,
    *,
    use_auth: bool = False,
) -> Any:
    file_path = cache_dir.joinpath(cache_relpath).with_suffix(".json")
    if file_path.exists():
        _log(f"loading Lancelot JSON cache {file_path}")
        return json.loads(file_path.read_text(encoding="utf-8"))
    data = mlBridgeFFLib.lancelot_get(path, token=token if use_auth else None)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def lancelot_session_meta(
    player_id: str,
    session_id: Any,
    game_entry: Dict[str, Any],
    *,
    token: Optional[str] = None,
    cache_dir: Optional[pathlib.Path] = None,
) -> Tuple[LancelotSessionMeta, pl.DataFrame, Dict[str, Any]]:
    """Ranking-only session metadata (no score download, no augment)."""
    directory = pathlib.Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    sid = int(session_id)
    resolved = resolve_player(str(player_id), token=token)
    pid = int(resolved.lancelot_id)
    ranking_json = _fetch_lancelot_json_cached(
        f"results/sessions/{sid}/ranking",
        f"rankings/{sid}",
        token,
        directory,
    )
    teams_df = json_normalize_api(ranking_json)
    match_df = teams_df.filter(
        pl.col("team_player1_id").cast(pl.Int64, strict=False).eq(pid)
        | pl.col("team_player2_id").cast(pl.Int64, strict=False).eq(pid)
    )
    if match_df.height == 0:
        raise ValueError(f"Player {resolved.lancelot_id} not found in ranking of session {sid}.")
    team_d = match_df.to_dicts()[0]
    is_player1 = int(team_d["team_player1_id"]) == pid
    pair_direction = team_d["orientation"]
    me, partner = ("team_player1", "team_player2") if is_player1 else ("team_player2", "team_player1")
    team_id = team_d["team_id"]
    group_id = game_entry.get("group_id")
    meta = LancelotSessionMeta(
        player_id=_as_int_str(team_d[f"{me}_id"]) or resolved.lancelot_id,
        session_id=str(sid),
        partner_id=_as_int_str(team_d[f"{partner}_id"]),
        player_license_number=_as_int_str(team_d[f"{me}_ffbId"]),
        partner_license_number=_as_int_str(team_d[f"{partner}_ffbId"]),
        player_name=f"{team_d[f'{me}_firstName']} {team_d[f'{me}_lastName']}",
        partner_name=f"{team_d[f'{partner}_firstName']} {team_d[f'{partner}_lastName']}",
        group_id=group_id,
        org_id=game_entry.get("organization_id"),
        tournament_date=_parse_session_date(game_entry.get("date") or game_entry.get("raw_date")),
        organization_name=game_entry.get("organization_name") or game_entry.get("club"),
        game_description=game_entry.get("competition_label"),
        team_id=team_id,
        pair_direction=pair_direction,
        opponent_pair_direction="EW" if pair_direction == "NS" else "NS",
        player_direction=pair_direction[0 if is_player1 else 1],
        partner_direction=pair_direction[1 if is_player1 else 0],
        section_name=team_d.get("section"),
        team_number=team_d.get("tableNumber"),
        game_url=(
            f"https://www.ffbridge.fr/competitions/results/groups/{group_id}"
            f"/sessions/{sid}/pairs/{team_id}"
        ),
    )
    return meta, teams_df, team_d


def build_lancelot_session_mldf(
    player_id: str,
    session_id: Any,
    game_entry: Dict[str, Any],
    *,
    token: Optional[str] = None,
    cache_dir: Optional[pathlib.Path] = None,
    team_progress: Optional[TeamProgress] = None,
) -> LancelotSessionBuild:
    """Download ranking + club scores and convert to the report mldf."""
    directory = pathlib.Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    sid = int(session_id)
    meta, teams_df, team_d = lancelot_session_meta(
        player_id, sid, game_entry, token=token, cache_dir=directory
    )
    _log(f"ranking teams_df shape: {teams_df.shape}")
    pair_direction = meta.pair_direction
    team_id = meta.team_id
    group_id = meta.group_id
    tournament_date = meta.tournament_date

    club_code = team_d.get("simultaneousId")
    if club_code is not None and "simultaneousId" in teams_df.columns:
        club_teams_df = teams_df.filter(pl.col("simultaneousId").eq(club_code))
    else:
        club_teams_df = teams_df
    team_ids = club_teams_df["team_id"].to_list()
    _log(f"Teams to fetch scores for: {len(team_ids)} (club_code={club_code})")

    t_scores = time.time()
    iterable: Iterable[Any] = team_ids
    if team_progress is not None:
        iterable = team_progress(team_ids)
    else:
        iterable = tqdm(team_ids, desc="Downloading team scores...")

    teams_jsons: List[Dict[str, Any]] = []
    for one_team_id in iterable:
        scores_json = _fetch_lancelot_json_cached(
            f"results/teams/{one_team_id}/session/{sid}/scores",
            f"scores/{one_team_id}_{sid}",
            token,
            directory,
        )
        if all(board.get("contract", "") == "" for board in scores_json):
            if one_team_id == team_id:
                raise ValueError(
                    f"Board contract data is missing for the selected team in session {sid}. "
                    "This tournament's detailed results may not be available through the Lancelot API "
                    "(e.g. some Roy René events)."
                )
            _log(f"Skipping team {one_team_id}: missing contract data.")
            continue
        teams_jsons.extend(scores_json)
    _elapsed_if_slow(f"score download session {sid}", t_scores)

    if not teams_jsons:
        raise ValueError(f"No board score data available for session {sid}.")

    ffdf = json_normalize_api(teams_jsons)
    ffdf = ffdf.with_columns(
        pl.lit(group_id).alias("group_id"),
        pl.lit(sid).alias("session_id"),
    )
    ffdf = ffdf.unique(subset=["board_id", "id"], keep="first")

    df = mlBridgeFFLib.convert_ffdf_lancelot_to_mldf(ffdf)
    df = df.with_columns(
        pl.lit(tournament_date).alias("Date"),
        pl.col("section_name").alias("Section_Name"),
        pl.lit(pair_direction).alias("Pair_Direction"),
    )
    table_number_by_team = dict(zip(teams_df["team_id"].to_list(), teams_df["tableNumber"].to_list()))
    df = df.with_columns(
        pl.when(pl.col("Pair_Direction_Home").eq("NS"))
        .then(pl.col("team_id_home"))
        .otherwise(pl.col("team_id_away"))
        .replace_strict(table_number_by_team, default=None)
        .cast(pl.Int64, strict=False)
        .alias("Pair_Number_NS"),
        pl.when(pl.col("Pair_Direction_Home").eq("EW"))
        .then(pl.col("team_id_home"))
        .otherwise(pl.col("team_id_away"))
        .replace_strict(table_number_by_team, default=None)
        .cast(pl.Int64, strict=False)
        .alias("Pair_Number_EW"),
    )
    return LancelotSessionBuild(df=df, meta=meta, ranking_df=teams_df, scores_df=ffdf)


def reduce_mldf_for_augment(df: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in CORE_MLDF_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"mldf is missing columns required for augmentation: {missing}")
    return df.select(CORE_MLDF_COLUMNS)


def augment_and_cache_mldf(
    df: pl.DataFrame,
    session_id: Any,
    player_id: str,
    *,
    cache_dir: Optional[pathlib.Path] = None,
    force: bool = False,
    sd_productions: int = DEFAULT_SD_PRODUCTIONS,
    progress: Any = None,
    lock_func: Optional[Callable[..., Any]] = None,
    write_cache: bool = True,
) -> pl.DataFrame:
    """Reduce to core columns, augment (unless cached), write parquet."""
    if df["Contract"].is_null().all():
        raise ValueError("No Contract data available. Unable to proceed.")
    df = reduce_mldf_for_augment(df)
    directory = pathlib.Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_file = cache_parquet_path(session_id, player_id, directory)
    if write_cache and cache_file.exists() and not force:
        loaded = pl.read_parquet(cache_file)
        _log(f"loaded {cache_file.name}: shape:{loaded.shape} size:{cache_file.stat().st_size}")
        return loaded

    t_aug = time.time()
    augmenter = AllAugmentations(
        df,
        None,
        sd_productions=sd_productions,
        progress=progress,
        lock_func=lock_func,
        output_progress=True,
    )
    df, _hrs = augmenter.perform_all_augmentations()
    _elapsed_if_slow(f"augment session {session_id}", t_aug)

    if write_cache:
        directory.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_file)
        _log(f"saved {cache_file.name}: shape:{df.shape} size:{cache_file.stat().st_size}")
    return df


def create_lancelot_postmortem(
    player_id: str,
    session_id: Any,
    game_entry: Dict[str, Any],
    *,
    token: Optional[str] = None,
    cache_dir: Optional[pathlib.Path] = None,
    force: bool = False,
    sd_productions: int = DEFAULT_SD_PRODUCTIONS,
    team_progress: Optional[TeamProgress] = None,
    augment_progress: Any = None,
    lock_func: Optional[Callable[..., Any]] = None,
    skip_build_if_cached: bool = False,
) -> Dict[str, Any]:
    """Build, augment, and cache one Lancelot session. Returns a result dict."""
    directory = pathlib.Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    resolved = resolve_player(str(player_id), token=token)
    cache_file = cache_parquet_path(session_id, resolved.lancelot_id, directory)
    started = datetime.now()
    _log(
        f"start create session={session_id} player={resolved.lancelot_id} "
        f"at {started.isoformat(timespec='seconds')}"
    )

    if skip_build_if_cached and cache_file.exists() and not force:
        meta, _, _ = lancelot_session_meta(
            resolved.lancelot_id, session_id, game_entry, token=token, cache_dir=directory
        )
        _log(f"end create session={session_id} status=cached")
        return {
            "session_id": str(session_id),
            "player_id": resolved.lancelot_id,
            "player_license_number": resolved.license_number,
            "status": "cached",
            "cache_file": cache_file.name,
            "meta": meta,
            "df": None,
        }

    built = build_lancelot_session_mldf(
        resolved.lancelot_id,
        session_id,
        game_entry,
        token=token,
        cache_dir=directory,
        team_progress=team_progress,
    )
    df = augment_and_cache_mldf(
        built.df,
        built.meta.session_id,
        built.meta.player_id,
        cache_dir=directory,
        force=force,
        sd_productions=sd_productions,
        progress=augment_progress,
        lock_func=lock_func,
    )
    ended = datetime.now()
    _log(
        f"end create session={built.meta.session_id} status=ok "
        f"elapsed {(ended - started).total_seconds():.1f}s"
    )
    return {
        "session_id": built.meta.session_id,
        "player_id": built.meta.player_id,
        "player_license_number": built.meta.player_license_number,
        "status": "ok",
        "cache_file": cache_parquet_path(built.meta.session_id, built.meta.player_id, directory).name,
        "meta": built.meta,
        "df": df,
        "ranking_df": built.ranking_df,
        "scores_df": built.scores_df,
    }


# ---------------------------------------------------------------------------
# MCP generate (single session or date window, with background jobs)
# ---------------------------------------------------------------------------


@dataclass
class GenerateJob:
    job_id: str
    status: str
    player_id: str
    player_license_number: Optional[str]
    requested_id: str
    session_ids: List[str]
    force: bool
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None
    continue_on_error: bool = True
    results: List[Dict[str, Any]] = field(default_factory=list)
    failed_session_ids: List[str] = field(default_factory=list)
    progress: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_jobs_lock = threading.Lock()
_jobs: Dict[str, GenerateJob] = {}


def generate_status(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise KeyError(f"Unknown generate job_id {job_id}")
    return job.as_dict()


def _select_sessions_to_generate(
    resolved: ResolvedPlayer,
    auth: LancelotAuth,
    session_id: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    cache_dir: pathlib.Path,
) -> List[Dict[str, Any]]:
    listed = list_source_sessions(
        resolved.lancelot_id,
        date_from=date_from,
        date_to=date_to,
        token=auth.token,
        cache_dir=cache_dir,
    )
    sessions: List[Dict[str, Any]] = listed["sessions"]
    if session_id is not None:
        sid = str(session_id).strip()
        if sid.lower() == "latest":
            sid = ""
            session_id = None
        else:
            match = [s for s in sessions if s["session_id"] == sid]
            if not match:
                raise ValueError(
                    f"Session {sid} not found in playable Lancelot games for "
                    f"player {resolved.lancelot_id}."
                )
            return match
    if session_id is None and date_from is None and date_to is None:
        if not sessions:
            raise ValueError(f"No playable Lancelot sessions for player {resolved.lancelot_id}.")
        return [sessions[0]]
    if not sessions:
        raise ValueError(
            f"No playable Lancelot sessions for player {resolved.lancelot_id} "
            f"in {date_from or '...'}..{date_to or '...'}."
        )
    return sessions


def _session_result_row(
    *,
    session_id: str,
    player_id: str,
    status: str,
    cache_file: Optional[str] = None,
    error: Optional[str] = None,
    meta: Any = None,
) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "player_id": player_id,
        "status": status,
        "cache_file": cache_file,
        "error": error,
        "meta": asdict(meta) if meta is not None and not isinstance(meta, dict) else meta,
    }


def _run_generate_job(job: GenerateJob, auth: LancelotAuth, cache_dir: pathlib.Path, force: bool) -> None:
    job.status = "running"
    already_done = len(job.results)
    job.progress = {
        "done": already_done,
        "total": already_done + len(job.session_ids),
        "current_session_id": None,
    }
    aborted = False
    try:
        listed = list_source_sessions(
            job.player_id,
            token=auth.token,
            cache_dir=cache_dir,
        )
        by_id = {s["session_id"]: s for s in listed["sessions"]}
        for sid in tqdm(job.session_ids, desc="Generating postmortems"):
            job.progress["current_session_id"] = sid
            try:
                try:
                    entry = by_id.get(sid)
                    if entry is None:
                        raise ValueError(f"Session {sid} disappeared from the Lancelot game list.")
                    result = create_lancelot_postmortem(
                        job.player_id,
                        sid,
                        entry,
                        token=auth.token,
                        cache_dir=cache_dir,
                        force=force,
                        skip_build_if_cached=True,
                    )
                    job.results.append(
                        _session_result_row(
                            session_id=result["session_id"],
                            player_id=result["player_id"],
                            status=result["status"],
                            cache_file=result["cache_file"],
                            meta=result.get("meta"),
                        )
                    )
                except Exception as e:
                    err = str(e)
                    _log(f"generate job {job.job_id} session {sid} failed: {err}")
                    job.failed_session_ids.append(sid)
                    job.results.append(
                        _session_result_row(
                            session_id=sid,
                            player_id=job.player_id,
                            status="error",
                            error=err,
                        )
                    )
                    if not job.continue_on_error:
                        job.status = "error"
                        job.error = err
                        aborted = True
                        break
            finally:
                job.progress["done"] = len(job.results)
        if not aborted:
            any_ok = any(r.get("status") in ("ok", "cached") for r in job.results)
            if job.failed_session_ids and any_ok:
                job.status = "completed"
            elif any_ok:
                job.status = "ok"
            else:
                job.status = "error"
                job.error = job.error or "All sessions failed."
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        _log(f"generate job {job.job_id} failed: {e}")
    finally:
        job.finished_at = datetime.now().isoformat(timespec="seconds")
        job.progress["current_session_id"] = None


def generate_postmortems(
    player_id: str,
    session_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    force: bool = False,
    continue_on_error: Optional[bool] = None,
    *,
    cache_dir: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Create postmortem parquet(s). Returns immediately with a job_id when work is needed.

    continue_on_error defaults to True for date-range jobs and False for a
    single session_id so one bad club result does not abort a 2025–2026 run.
    """
    started = datetime.now()
    _log(f"start generate player={player_id} session={session_id} at {started.isoformat(timespec='seconds')}")
    directory = pathlib.Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    auth = ensure_lancelot_auth()
    resolved = resolve_player(player_id, token=auth.token)
    if continue_on_error is None:
        continue_on_error = date_from is not None or date_to is not None
    sessions = _select_sessions_to_generate(
        resolved, auth, session_id, date_from, date_to, directory
    )

    cached_results = []
    to_build = []
    for entry in sessions:
        path = cache_parquet_path(entry["session_id"], resolved.lancelot_id, directory)
        if path.is_file() and not force:
            cached_results.append(
                {
                    "session_id": entry["session_id"],
                    "player_id": resolved.lancelot_id,
                    "status": "cached",
                    "cache_file": path.name,
                }
            )
        else:
            to_build.append(entry)

    if not to_build:
        _log(f"end generate player={resolved.lancelot_id} status=cached count={len(cached_results)}")
        first = cached_results[0]
        payload: Dict[str, Any] = {
            "player_id": resolved.lancelot_id,
            "player_license_number": resolved.license_number,
            "requested_id": resolved.requested_id,
            "status": "cached",
            "session_id": first["session_id"] if len(cached_results) == 1 else None,
            "cache_file": first["cache_file"] if len(cached_results) == 1 else None,
            "sessions": cached_results,
            "count": len(cached_results),
        }
        return payload

    job_id = uuid.uuid4().hex
    job = GenerateJob(
        job_id=job_id,
        status="started",
        player_id=resolved.lancelot_id,
        player_license_number=resolved.license_number,
        requested_id=resolved.requested_id,
        session_ids=[s["session_id"] for s in to_build],
        force=force,
        started_at=started.isoformat(timespec="seconds"),
        continue_on_error=bool(continue_on_error),
        results=list(cached_results),
        progress={"done": len(cached_results), "total": len(cached_results) + len(to_build)},
    )
    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_generate_job,
        args=(job, auth, directory, force),
        name=f"ffbridge-generate-{job_id[:8]}",
        daemon=True,
    )
    thread.start()
    _log(f"started generate job {job_id} sessions={job.session_ids}")
    return {
        "player_id": resolved.lancelot_id,
        "player_license_number": resolved.license_number,
        "requested_id": resolved.requested_id,
        "status": "started",
        "job_id": job_id,
        "session_id": job.session_ids[0] if len(job.session_ids) == 1 else None,
        "cache_file": None,
        "sessions": cached_results,
        "pending_session_ids": job.session_ids,
        "count": len(cached_results) + len(to_build),
    }
