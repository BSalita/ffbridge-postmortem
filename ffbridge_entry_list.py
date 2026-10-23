"""Read a session's entry list from FFBridge, before play.

The list is what ffbridge.fr shows as "Liste des équipes inscrites". It comes
from the Lancelot API with the service's own sign-in, so the forecast needs
no pasted names: ``predict_group_session`` hands each pair's Lancelot person
ids to ``ffbridge_entries_predictions.predict_entries``.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import requests

import ffbridge_entries_predictions as entries_predictions
import ffbridge_postmortem_create as create

LANCELOT_API = "https://api-lancelot.ffbridge.fr"
PAGE_SIZE = 100


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


def upcoming_entries(date_from: Optional[str] = None) -> List[Dict[str, Any]]:
    """The service account's own upcoming entries, soonest first."""
    auth = create.ensure_lancelot_auth()
    raw = _api_get(
        f"/persons/{auth.lancelot_id}/teams",
        {
            "dateFrom": date_from or date.today().isoformat(),
            "includePreEntries": 1,
            "currentPage": 1,
            "maxPerPage": 80,
        },
    )
    sessions: List[Dict[str, Any]] = []
    for team in raw.get("items") or []:
        partners = [
            _person_name(person)
            for person in team.get("players") or []
            if str(person.get("id")) != auth.lancelot_id
        ]
        for entry in team.get("teamEntries") or []:
            group_session = (entry.get("groupSession") or {}).get("id")
            if group_session is None:
                continue
            meta = session_meta(str(group_session))
            meta["partner"] = ", ".join(partners) or None
            sessions.append(meta)
    return sorted(sessions, key=lambda row: row.get("date") or "")


def predict_group_session(
    group_session_id: Optional[str] = None,
    sims: int = entries_predictions.DEFAULT_SIMS,
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
    result = entries_predictions.predict_entries(entries, str(meta["date"] or "")[:10] or None, sims=sims)
    result["session"] = meta
    result["entry_list_read_at"] = read_at
    result["team_iv"] = {
        " - ".join(player["name"] for player in pair["players"]): pair["team_iv"] for pair in entrants
    }
    return result
