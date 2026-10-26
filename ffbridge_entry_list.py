"""Read a session's entry list from FFBridge, before play.

The list is what ffbridge.fr shows as "Liste des équipes inscrites". It comes
from the Lancelot API with the service's own sign-in, so the forecast needs
no pasted names: ``forecast_group_session`` hands each pair's Lancelot person
ids to ``ffbridge_entries_forecast.forecast_entries``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import requests

import ffbridge_entries_forecast as entries_forecast
import ffbridge_postmortem_create as create

LANCELOT_API = "https://api-lancelot.ffbridge.fr"
PAGE_SIZE = 100
# How far ahead the club-schedule scan looks for a player whose own entry list is hidden.
CLUB_SCAN_DAYS = 14


def _api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """GET from the Lancelot API as the service account; sign in again once on 401."""
    for attempt in (0, 1):
        auth = create.ensure_lancelot_auth(force=attempt == 1)
        response = requests.get(
            f"{LANCELOT_API}{path}",
            params=params,
            headers={"Authorization": f"Bearer {auth.token}", "Accept": "application/json"},
            timeout=30,
        )
        if response.status_code == 401 and attempt == 0:
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable")


def _person_name(person: Dict[str, Any]) -> str:
    return f"{person.get('lastName') or ''} {person.get('firstName') or ''}".strip()


def session_meta(group_session_id: str) -> Dict[str, Any]:
    raw = _api_get(f"/competitions/groupSessions/{group_session_id}")
    organization = (
        ((raw.get("group") or {}).get("phase") or {}).get("stade") or {}
    ).get("organization") or {}
    return {
        "group_session_id": str(group_session_id),
        "date": raw.get("date"),
        "label": ((raw.get("session") or {}).get("label")) or raw.get("description"),
        "expected_boards": raw.get("expectedBoardCount"),
        "club_code": str(organization["ffbCode"]) if organization.get("ffbCode") is not None else None,
        "club_name": organization.get("label") or organization.get("name"),
    }


def session_entrants(group_session_id: str) -> List[Dict[str, Any]]:
    """Every registered pair, all pages: players, team IV, registration time."""
    pairs: List[Dict[str, Any]] = []
    page = 1
    while True:
        raw = _api_get(
            "/entries/teams/search",
            {"groupSessionId": group_session_id, "currentPage": page, "maxPerPage": PAGE_SIZE},
        )
        items = raw.get("items") or []
        for team in items:
            pairs.append(
                {
                    "team_id": str(team.get("id")),
                    "players": [
                        {
                            "lancelot_id": str(player.get("id")),
                            "licence": str(player.get("ffbId") or ""),
                            "name": _person_name(player),
                            "iv": ((player.get("season") or {}).get("ranking") or {}).get("iv"),
                        }
                        for player in team.get("players") or []
                    ],
                    "team_iv": team.get("iv"),
                    "registered_at": (team.get("currentTeamEntry") or {}).get("createdAt"),
                }
            )
        if len(items) < PAGE_SIZE:
            return pairs
        page += 1


class EntriesPrivate(PermissionError):
    """FFBridge does not show this player's entries to the service account."""


def upcoming_entries(date_from: Optional[str] = None, person_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """A player's upcoming entries, soonest first (default: the service account).

    FFBridge answers 403 for some players; that raises EntriesPrivate.
    """
    auth = create.ensure_lancelot_auth()
    person = str(person_id or auth.lancelot_id)
    try:
        raw = _api_get(
            f"/persons/{person}/teams",
            {
                "dateFrom": date_from or date.today().isoformat(),
                "includePreEntries": 1,
                "currentPage": 1,
                "maxPerPage": 80,
            },
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            raise EntriesPrivate(f"FFBridge does not show the entries of player {person}") from exc
        raise
    sessions: List[Dict[str, Any]] = []
    for team in raw.get("items") or []:
        partners = [
            _person_name(member)
            for member in team.get("players") or []
            if str(member.get("id")) != person
        ]
        for entry in team.get("teamEntries") or []:
            group_session = (entry.get("groupSession") or {}).get("id")
            if group_session is None:
                continue
            meta = session_meta(str(group_session))
            meta["partner"] = ", ".join(partners) or None
            sessions.append(meta)
    return sorted(sessions, key=lambda row: row.get("date") or "")


def entries_from_club_schedules(
    person_id: str,
    group_ids: Iterable[str],
    date_from: Optional[str] = None,
    days: int = CLUB_SCAN_DAYS,
) -> List[Dict[str, Any]]:
    """A player's upcoming entries, found in the entry lists of the given club groups.

    For players whose own entry list FFBridge hides (``EntriesPrivate``): every
    session of those groups in the next ``days`` days is read, and the ones
    the player is entered in are returned like ``upcoming_entries`` rows.
    Games at clubs outside ``group_ids`` are not found.
    """
    person = str(person_id)
    first_day = date.fromisoformat(date_from) if date_from else date.today()
    last_day = first_day + timedelta(days=days)
    candidates: Dict[str, str] = {}
    for group_id in sorted({str(g) for g in group_ids if g}):
        for row in _api_get(f"/competitions/groups/{group_id}/groupSessions") or []:
            day = str(row.get("date") or "")[:10]
            if row.get("id") is not None and first_day.isoformat() <= day <= last_day.isoformat():
                candidates[str(row["id"])] = day

    def entered(group_session_id: str) -> Optional[Dict[str, Any]]:
        for pair in session_entrants(group_session_id):
            ids = [player["lancelot_id"] for player in pair["players"]]
            if person in ids:
                meta = session_meta(group_session_id)
                partners = [p["name"] for p in pair["players"] if p["lancelot_id"] != person]
                meta["partner"] = ", ".join(partners) or None
                return meta
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        found = [row for row in executor.map(entered, candidates) if row is not None]
    return sorted(found, key=lambda row: row.get("date") or "")


def forecast_group_session(
    group_session_id: Optional[str] = None,
    sims: int = entries_forecast.DEFAULT_SIMS,
) -> Dict[str, Any]:
    """Forecast for a group session's current entry list.

    Without ``group_session_id``, the service account's next entered session.
    """
    if group_session_id is None:
        upcoming = upcoming_entries()
        if not upcoming:
            raise ValueError("The service account has no upcoming entries")
        group_session_id = upcoming[0]["group_session_id"]
    read_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = session_meta(str(group_session_id))
    entrants = session_entrants(str(group_session_id))
    if len(entrants) < 2:
        raise ValueError(f"Group session {group_session_id} has {len(entrants)} registered pairs")
    entries = [
        {
            "players": [player["lancelot_id"] for player in pair["players"]],
            "label": " - ".join(player["name"] for player in pair["players"]),
        }
        for pair in entrants
    ]
    result = entries_forecast.forecast_entries(entries, str(meta["date"] or "")[:10] or None, sims=sims)
    result["session"] = meta
    result["entry_list_read_at"] = read_at
    result["team_iv"] = {
        " - ".join(player["name"] for player in pair["players"]): pair["team_iv"] for pair in entrants
    }
    # Pair label for each entered player, to find a player's own row.
    result["pair_of_player"] = {
        player["lancelot_id"]: " - ".join(p["name"] for p in pair["players"])
        for pair in entrants
        for player in pair["players"]
    }
    return result


def player_row(forecast: Dict[str, Any], person_id: str) -> Optional[Dict[str, Any]]:
    """The forecast row of the pair that ``person_id`` entered with, if any."""
    label = (forecast.get("pair_of_player") or {}).get(str(person_id))
    if label is None:
        return None
    return next((row for row in forecast.get("rows") or [] if row.get("Pair") == label), None)


def entries_with_forecast(
    person_id: Optional[str] = None,
    date_from: Optional[str] = None,
    sims: int = 5000,
    club_group_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """A player's upcoming entries, each with that player's forecast rank and pct.

    A session with fewer than two registered pairs gets ``forecast_error``.
    When FFBridge keeps the player's entries private, the entry lists of
    ``club_group_ids`` are scanned instead and each row gets
    ``found_via_club_scan``; without club groups, EntriesPrivate is raised.
    """
    person = str(person_id or create.ensure_lancelot_auth().lancelot_id)
    try:
        rows = upcoming_entries(date_from, person_id=person)
    except EntriesPrivate:
        if not club_group_ids:
            raise
        rows = entries_from_club_schedules(person, club_group_ids, date_from)
        for row in rows:
            row["found_via_club_scan"] = True
    for row in rows:
        try:
            forecast = forecast_group_session(row["group_session_id"], sims=sims)
        except ValueError as exc:
            row["forecast_error"] = str(exc)
            continue
        own = player_row(forecast, person)
        row["pairs"] = forecast.get("pairs")
        row["pair"] = own.get("Pair") if own else None
        row["orientation"] = own.get("Orientation") if own else None
        row["forecast_rank"] = own.get("Rank_Forecast") if own else None
        row["forecast_pct"] = own.get("Pct_Forecast") if own else None
        row["entry_list_read_at"] = forecast.get("entry_list_read_at")
    return rows
