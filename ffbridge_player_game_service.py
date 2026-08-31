"""Fast live/indexed lookup of an FFBridge player's latest game."""
from __future__ import annotations

import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

import requests

import ffbridge_postmortem_create as create


_CLUB_CATALOG_LOCK = threading.Lock()
_CLUB_CATALOG: Optional[List[Dict[str, Any]]] = None
_GROUP_URL_RE = re.compile(r"/groups/(\d+)")
_MOMENTS = {
    "A": "Après-midi",
    "M": "Matin",
    "S": "Soir",
    "E": "Soir",
}


def _text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return " ".join(
        "".join(c for c in normalized if not unicodedata.combining(c))
        .casefold()
        .split()
    )


def _date(value: Any) -> Optional[str]:
    return str(value)[:10] if value else None


def _person_name(person: Dict[str, Any]) -> str:
    last = str(person.get("lastName") or "").strip().upper()
    first = str(person.get("firstName") or "").strip()
    return " ".join(part for part in (last, first) if part)


def _player_matches(person: Dict[str, Any], resolved: create.ResolvedPlayer) -> bool:
    identifiers = {
        str(value)
        for value in (
            person.get("id"),
            person.get("migrationId"),
            person.get("ffbId"),
        )
        if value is not None
    }
    return bool(identifiers.intersection(resolved.aliases()))


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _percentage(value: Any) -> str:
    return f"{float(value):.2f}".replace(".", ",")


def _club_catalog() -> List[Dict[str, Any]]:
    global _CLUB_CATALOG
    with _CLUB_CATALOG_LOCK:
        if _CLUB_CATALOG is not None:
            return _CLUB_CATALOG
        def load_page(page: int) -> Dict[str, Any]:
            return create.mlBridgeFFLib.lancelot_get(
                "results/search/",
                params={
                    "competitionType": "club",
                    "searchSeason": "current",
                    "currentPage": page,
                },
            )

        first = load_page(1)
        total_pages = int((first.get("pagination") or {}).get("total_pages") or 1)
        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = [
                first,
                *executor.map(load_page, range(2, min(total_pages, 100) + 1)),
            ]
        rows: List[Dict[str, Any]] = []
        for response in responses:
            for stade in response.get("items", []):
                organization = stade.get("organization") or {}
                for phase in stade.get("phases") or []:
                    for group in phase.get("groups") or []:
                        rows.append(
                            {
                                "group_id": str(group["id"]),
                                "club_id": (
                                    str(organization["id"])
                                    if organization.get("id") is not None
                                    else None
                                ),
                                "club_migration_id": (
                                    str(organization["migrationId"])
                                    if organization.get("migrationId") is not None
                                    else None
                                ),
                                "club_code": str(organization.get("ffbCode") or ""),
                                "club_name": str(
                                    organization.get("name")
                                    or organization.get("label")
                                    or ""
                                ),
                            }
                        )
        _CLUB_CATALOG = rows
        return rows


@lru_cache(maxsize=4096)
def _direct_group(token: str) -> Optional[Dict[str, Any]]:
    match = _GROUP_URL_RE.search(token)
    group_id = match.group(1) if match else token if token.isdigit() else None
    if group_id is None or len(group_id) > 6:
        return None
    try:
        group = create.mlBridgeFFLib.lancelot_get(
            f"competitions/groups/{group_id}"
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise
    stade = ((group.get("phase") or {}).get("stade") or {})
    organization = stade.get("organization") or {}
    season = (
        (stade.get("competitionDivision") or {}).get("season")
        or stade.get("season")
        or {}
    )
    return {
        "group_id": group_id,
        "season_id": (
            int(season["id"]) if season.get("id") is not None else None
        ),
        "season_label": season.get("label"),
        "club_id": (
            str(organization["id"]) if organization.get("id") is not None else None
        ),
        "club_migration_id": (
            str(organization["migrationId"])
            if organization.get("migrationId") is not None
            else None
        ),
        "club_code": str(organization.get("ffbCode") or ""),
        "club_name": str(
            organization.get("name") or organization.get("label") or ""
        ),
    }


def resolve_clubs(clubs: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Resolve group URLs/IDs, club codes, internal IDs, or exact club names."""
    resolved: List[Dict[str, Any]] = []
    for raw in clubs or []:
        token = str(raw).strip()
        if not token:
            continue
        direct = _direct_group(token)
        if direct is not None:
            resolved.append(direct)
            continue
        normalized = _text(token)
        matches = [
            row
            for row in _club_catalog()
            if normalized
            in {
                _text(row["club_name"]),
                _text(row["club_code"]),
                _text(row["club_id"]),
                _text(row["club_migration_id"]),
            }
        ]
        if not matches:
            raise ValueError(f"Unknown current-season FFBridge club {token!r}")
        # results/search omits season metadata and may return the club's groups
        # from several seasons despite searchSeason=current. Resolve the small
        # matched set to full group records before selecting the newest season.
        with ThreadPoolExecutor(max_workers=min(8, len(matches))) as executor:
            detailed_matches = list(
                executor.map(
                    lambda row: _direct_group(str(row["group_id"])),
                    matches,
                )
            )
        matches = [row for row in detailed_matches if row is not None]
        if not matches:
            raise ValueError(
                f"No usable current-season FFBridge groups for club {token!r}"
            )
        season_ids = [
            int(row["season_id"])
            for row in matches
            if row.get("season_id") is not None
        ]
        if season_ids:
            latest_season_id = max(season_ids)
            matches = [
                row
                for row in matches
                if row.get("season_id") == latest_season_id
            ]
        club_keys = {(row["club_code"], row["club_name"]) for row in matches}
        if len(club_keys) > 1:
            raise ValueError(
                f"Ambiguous FFBridge club {token!r}; provide its club code or group URL"
            )
        resolved.extend(matches)
    return list({row["group_id"]: row for row in resolved}.values())


def _simultaneous_sessions(
    date_from: str,
    date_to: str,
) -> List[Dict[str, Any]]:
    sessions: Dict[str, Dict[str, Any]] = {}
    for lancelot_series_id, migration_series_id in (
        create.mlBridgeFFLib.LANCELOT_TO_MIGRATION.items()
    ):
        for page in range(1, 21):
            response = create.mlBridgeFFLib.get_simultaneous_sessions_page(
                lancelot_series_id,
                page=page,
                per_page=80,
            )
            items = response.get("items") or []
            dates = [parsed for item in items if (parsed := _date(item.get("date")))]
            for item in items:
                session_date = _date(item.get("date"))
                if not session_date or not date_from <= session_date <= date_to:
                    continue
                session_id = item.get("id")
                if session_id is None:
                    continue
                sessions[str(session_id)] = {
                    "session_id": str(session_id),
                    "date": session_date,
                    "raw_date": item.get("date"),
                    "session_label": item.get("label") or "",
                    "moment": item.get("moment"),
                    "series_id": migration_series_id,
                    "lancelot_series_id": lancelot_series_id,
                    "group_id": None,
                    "club_code": None,
                    "club_name": None,
                    "scope": "simultaneous",
                }
            if dates and max(dates) < date_from:
                break
            if not (response.get("pagination") or {}).get("has_next_page"):
                break
    return list(sessions.values())


def _http_status(exc: requests.RequestException) -> Optional[int]:
    return exc.response.status_code if exc.response is not None else None


def _club_sessions(
    clubs: List[Dict[str, Any]],
    errors: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    sessions: Dict[str, Dict[str, Any]] = {}
    for club in clubs:
        path = f"competitions/groups/{club['group_id']}/groupSessions"
        try:
            rows = create.mlBridgeFFLib.lancelot_get(
                path,
                params={"context[]": "result_status"},
            )
        except requests.HTTPError as exc:
            # Lancelot occasionally fails while expanding result_status for an
            # otherwise valid group. The unexpanded endpoint still provides
            # the session list; personal-ranking lookup filters unpublished
            # sessions later.
            if (_http_status(exc) or 0) < 500:
                if errors is None:
                    raise
                errors.append(
                    {
                        "error": "upstream_group_sessions_error",
                        "club_code": club.get("club_code"),
                        "club_name": club.get("club_name"),
                        "group_id": club.get("group_id"),
                        "http_status": _http_status(exc),
                        "detail": str(exc),
                    }
                )
                continue
            try:
                rows = create.mlBridgeFFLib.lancelot_get(path)
            except requests.RequestException as retry_exc:
                if errors is None:
                    raise
                errors.append(
                    {
                        "error": "upstream_group_sessions_error",
                        "club_code": club.get("club_code"),
                        "club_name": club.get("club_name"),
                        "group_id": club.get("group_id"),
                        "http_status": _http_status(retry_exc),
                        "detail": str(retry_exc),
                    }
                )
                continue
        for row in rows:
            session = row.get("session") or {}
            if session.get("hasResult") is False:
                continue
            session_id = session.get("id")
            session_date = _date(row.get("date"))
            if session_id is None or session_date is None:
                continue
            sessions[str(session_id)] = {
                "session_id": str(session_id),
                "date": session_date,
                "raw_date": row.get("date"),
                "session_label": session.get("label") or "",
                "moment": row.get("moment"),
                "series_id": (
                    (session.get("simultaneous") or {}).get("migrationId")
                ),
                "lancelot_series_id": (
                    (session.get("simultaneous") or {}).get("id")
                ),
                "group_id": club["group_id"],
                "club_code": club["club_code"],
                "club_name": club["club_name"],
                "scope": "specified_club",
            }
    return list(sessions.values())


def _indexed_baseline(resolved: create.ResolvedPlayer) -> Dict[str, Any]:
    rows = create.fetch_other_player_source_sessions(resolved.lancelot_id)
    if not rows:
        raise FileNotFoundError(
            f"No simultaneous-session index entries for player {resolved.requested_id!r}"
        )
    return rows[0]


def _candidate_sessions(
    resolved: create.ResolvedPlayer,
    *,
    target_date: Optional[str],
    clubs: Optional[List[str]],
    club_errors: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    today = date.today().isoformat()
    club_rows = resolve_clubs(clubs)
    club_sessions = (
        _club_sessions(club_rows, errors=club_errors) if club_rows else []
    )
    if target_date is not None:
        date.fromisoformat(target_date)
        date_from = target_date
        date_to = target_date
        baseline = None
    else:
        try:
            baseline = _indexed_baseline(resolved)
        except FileNotFoundError:
            if not club_sessions:
                raise
            baseline = None
        if baseline is not None:
            date_from = str(baseline["date"])
        else:
            current = date.today()
            season_year = current.year if current.month >= 7 else current.year - 1
            date_from = f"{season_year}-07-01"
        date_to = today
    candidates = _simultaneous_sessions(date_from, date_to)
    if baseline is not None and not any(
        row["session_id"] == str(baseline["session_id"]) for row in candidates
    ):
        candidates.append(
            {
                **baseline,
                "session_id": str(baseline["session_id"]),
                "scope": "simultaneous_index",
            }
        )
    if club_sessions:
        candidates.extend(
            row
            for row in club_sessions
            if date_from <= row["date"] <= date_to
        )
    deduplicated = {row["session_id"]: row for row in candidates}
    return sorted(
        deduplicated.values(),
        key=lambda row: (row["date"], row["session_id"]),
        reverse=True,
    )


def _personal_ranking(session_id: str, license_number: str) -> Optional[Dict[str, Any]]:
    try:
        return create.mlBridgeFFLib.lancelot_get(
            f"results/sessions/{session_id}/ranking/{license_number}"
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise


def _local_rank(rows: Iterable[Dict[str, Any]], team_id: Any) -> Optional[int]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row.get("sessionScore") or float("-inf")),
            str((row.get("team") or {}).get("id") or ""),
        ),
        reverse=True,
    )
    for index, row in enumerate(ordered, start=1):
        if str((row.get("team") or {}).get("id")) == str(team_id):
            return index
    return None


def _session_club_context(session_id: str, club_code: str) -> Dict[str, Any]:
    session = create.mlBridgeFFLib.lancelot_get(
        f"competitions/sessions/{session_id}",
        params={"context[]": ["result_status", "result_data"]},
    )
    for group_session in session.get("groupSessions") or []:
        group = group_session.get("group") or {}
        organization = (
            (((group.get("phase") or {}).get("stade") or {}).get("organization"))
            or {}
        )
        if str(organization.get("ffbCode") or "") != club_code:
            continue
        return {
            "group_id": (
                str(group["id"]) if group.get("id") is not None else None
            ),
            "club_name": (
                organization.get("name") or organization.get("label")
            ),
            "raw_date": group_session.get("date"),
            "moment": group_session.get("moment"),
        }
    return {}


def _game(
    candidate: Dict[str, Any],
    ranking: Dict[str, Any],
    resolved: create.ResolvedPlayer,
) -> Dict[str, Any]:
    all_rows = create.mlBridgeFFLib.get_session_ranking(int(candidate["session_id"]))
    club_code = str(ranking.get("simultaneousId") or candidate.get("club_code") or "")
    candidate = {
        **candidate,
        **_session_club_context(candidate["session_id"], club_code),
    }
    local_rows = [
        row for row in all_rows if str(row.get("simultaneousId") or "") == club_code
    ]
    team = ranking.get("team") or {}
    players = [
        player
        for number in range(1, 9)
        if isinstance((player := team.get(f"player{number}")), dict)
    ]
    target_index = next(
        (index for index, person in enumerate(players) if _player_matches(person, resolved)),
        None,
    )
    if target_index is None:
        raise ValueError(
            f"Personal ranking did not contain player {resolved.requested_id!r}"
        )
    target = players[target_index]
    partner = next(
        (person for index, person in enumerate(players) if index != target_index),
        {},
    )
    orientation = str(ranking.get("orientation") or team.get("orientation") or "")
    seats = ("North", "South") if orientation == "NS" else ("East", "West")
    target_seat = seats[target_index] if target_index < 2 else None
    partner_seat = (
        seats[1 - target_index] if target_index in (0, 1) and partner else None
    )
    local_position = _local_rank(local_rows, team.get("id"))
    raw_date = str(candidate.get("raw_date") or "")
    moment = _MOMENTS.get(str(candidate.get("moment") or "").upper())
    if moment is None and "T" in raw_date:
        try:
            hour = int(raw_date[11:13])
            moment = (
                "Après-midi" if hour >= 12 else "Matin" if hour > 0 else None
            )
        except ValueError:
            moment = None
    game = {
        "session_id": candidate["session_id"],
        "group_id": candidate.get("group_id"),
        "series_id": candidate.get("series_id"),
        "competition": candidate.get("session_label") or "FFBridge",
        "date": candidate["date"],
        "moment": moment,
        "club_code": club_code or None,
        "club_name": candidate.get("club_name"),
        "team_count": len(local_rows),
        "local_rank": local_position,
        "general_rank": ranking.get("rank"),
        "theoretical_rank": ranking.get("theoreticalRank"),
        "section": ranking.get("section") or team.get("section"),
        "table_number": ranking.get("tableNumber") or team.get("startTableNumber"),
        "team_id": team.get("id"),
        "player_name": _person_name(target),
        "player_seat": target_seat,
        "partner_name": _person_name(partner),
        "partner_seat": partner_seat,
        "percentage": ranking.get("sessionScore"),
        "scope": candidate.get("scope"),
        "results_url": (
            f"https://www.ffbridge.fr/competitions/results/groups/"
            f"{candidate['group_id']}/sessions/{candidate['session_id']}/pairs/"
            f"{team.get('id')}"
            if candidate.get("group_id") and team.get("id")
            else None
        ),
    }
    game["summary"] = format_game_summary(game)
    return game


def format_game_summary(game: Dict[str, Any]) -> str:
    day, month, year = game["date"].split("-")[::-1]
    teams = f"{game['team_count']} équipes"
    finish = (
        f"{game['player_name']} finished {_ordinal(game['local_rank'])} "
        f"of {game['team_count']}"
    )
    ranks = (
        f"(G. {game['general_rank']}, TH. {game['theoretical_rank']})"
    )
    section_table = (
        f"Série {game['section']}, table {game['table_number']}"
    )
    partnership = (
        f"{game['player_name']} {game['player_seat']}, "
        f"{game['partner_name']} {game['partner_seat']}"
    )
    parts = [
        game["competition"],
        f"{day}/{month}/{year}",
        game.get("moment"),
        teams,
        f"{finish} {ranks}",
        section_table,
        partnership,
        f"{_percentage(game['percentage'])} %",
    ]
    return " — ".join(str(part) for part in parts[:2]) + " — " + " · ".join(
        str(part) for part in parts[2:] if part
    )


def _lookup(
    player: str,
    *,
    target_date: Optional[str],
    clubs: Optional[List[str]],
    first_only: bool,
) -> Dict[str, Any]:
    query = str(player or "").strip()
    if not query:
        raise ValueError("player is required; provide a player name or number")
    auth = create.ensure_lancelot_auth()
    resolved, display_name = create.resolve_player_query(query, auth=auth)
    if not resolved.license_number:
        raise ValueError(f"Could not determine a license number for {query!r}")
    games: List[Dict[str, Any]] = []
    club_errors: List[Dict[str, Any]] = []
    for candidate in _candidate_sessions(
        resolved,
        target_date=target_date,
        clubs=clubs,
        club_errors=club_errors,
    ):
        ranking = _personal_ranking(
            candidate["session_id"],
            resolved.license_number,
        )
        if not ranking:
            continue
        games.append(_game(candidate, ranking, resolved))
        if first_only:
            break
    return {
        "player_query": query,
        "player_id": resolved.lancelot_id,
        "player_license_number": resolved.license_number,
        "player_name": games[0]["player_name"] if games else display_name,
        "found": bool(games),
        "game": games[0] if games else None,
        "games": games,
        "summary": games[0]["summary"] if games else None,
        "coverage": (
            "configured simultaneous series plus specified current-season clubs"
            if clubs
            else "configured simultaneous series"
        ),
        "coverage_complete": not club_errors,
        "clubs": clubs or [],
        "club_errors": club_errors,
    }


def last_game(player: str, clubs: Optional[List[str]] = None) -> Dict[str, Any]:
    return _lookup(player, target_date=None, clubs=clubs, first_only=True)


def played_today(player: str, clubs: Optional[List[str]] = None) -> Dict[str, Any]:
    result = _lookup(
        player,
        target_date=date.today().isoformat(),
        clubs=clubs,
        first_only=False,
    )
    result["played"] = result["found"]
    result["date"] = date.today().isoformat()
    return result
