"""BridgeInterNet (bridgeinter.net) results while ffbridge.fr still points at the organizer.

Octopus and Simultanet sessions are run by Simultanés Bridge International.
FFBridge lists ``www.bridgeinter.net`` as their website with a short
``nonDisclosureDays`` window; during it Lancelot has no session for the date,
so the organizer's pages are the only board record. Same pattern as
``ffbridge_bridgeplus_postmortem`` (Roy René on Bridge+).

Pages used, all public:

* ``resseance*.php``  national ranking, which links each club's page
* ``restotal.php``    a club's pairs, each linking that pair's scorecard
* ``feuilleroute.php`` the pair's scorecard: contract, lead, score, % per board
* ``donnes.php``      each deal, with the whole field's score frequencies
"""

from __future__ import annotations

import pathlib
import re
import sys
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import polars as pl
import requests

import ffbridge_bridgeplus_postmortem as bridgeplus

_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
import mlBridge.mlBridgeBILib as bi  # noqa: E402

BRIDGEINTER_HOST = "bridgeinter.net"
# Above Bridge+ ids (900_000_000 + tournament number); yymmdd * 10 + game.
_SESSION_ID_BASE = 920_000_000
_GAME_INDEX = {"lo": 0, "jo": 1, "vi": 2}

Fetch = Callable[[str], str]


def default_fetch(url: str) -> str:
    return bi.default_get_text(url)


def bridgeinter_session_id(session_code: str) -> str:
    """Numeric id that cannot collide with Lancelot or Bridge+ ids, e.g. lo261026 -> 922610260."""
    prefix, yymmdd = session_code[:2].lower(), session_code[2:]
    if prefix not in _GAME_INDEX or not yymmdd.isdigit():
        raise ValueError(f"Unrecognized BridgeInterNet session code {session_code!r}")
    return str(_SESSION_ID_BASE + int(yymmdd) * 10 + _GAME_INDEX[prefix])


def _host_is_bridgeinter(url: str) -> bool:
    host = urlparse(url).netloc.casefold()
    return host == BRIDGEINTER_HOST or host.endswith("." + BRIDGEINTER_HOST)


def _organizers() -> List[Dict[str, Any]]:
    return bridgeplus._cached_lookup("bridgeinter-series", _load_organizers)


def _load_organizers() -> List[Dict[str, Any]]:
    organizers = []
    for series_id in bridgeplus._fflib().LANCELOT_TO_MIGRATION:
        payload = bridgeplus.fetch_json(f"competitions/simultaneous/{series_id}")
        website = payload.get("website")
        days = payload.get("nonDisclosureDays")
        if not website or not days:
            continue
        absolute = website if str(website).startswith("http") else f"https://{website}"
        if not _host_is_bridgeinter(absolute):
            continue
        organizers.append(
            {
                "lancelot_series_id": series_id,
                "series_id": payload.get("migrationId"),
                "label": payload.get("label") or "BridgeInterNet",
                "website": website,
                "non_disclosure_days": int(days),
            }
        )
    return organizers


def _tokens(text: str) -> List[str]:
    return re.sub(r"[^A-Z0-9 ]", " ", bi.normalize_text(text)).split()


def club_matches(ffb_club_name: str, bi_club_name: str) -> bool:
    """``Bridge Club Levallois Perret`` (FFBridge) matches ``Levallois-Perret`` (BridgeInterNet)."""
    ffb = set(_tokens(ffb_club_name))
    wanted = _tokens(bi_club_name)
    return bool(wanted) and all(token in ffb for token in wanted)


def name_matches(cell: str, last_name: str, first_name: str) -> bool:
    """``COLLINS Gilbert`` matches last ``Collins``, first ``Gilbert``."""
    tokens = set(_tokens(cell))
    last = _tokens(last_name)
    first = _tokens(first_name)
    return bool(last) and all(token in tokens for token in last) and (not first or first[0] in tokens)


def find_pair(rows: List[Dict[str, Any]], last_name: str, first_name: str) -> Optional[Dict[str, Any]]:
    """The club row with the player, and which of the two names is theirs. None unless unique."""
    found = []
    for row in rows:
        for index, key in enumerate(("player1_name", "player2_name")):
            if name_matches(str(row.get(key) or ""), last_name, first_name):
                found.append((row, index))
    if len(found) != 1:
        return None
    row, index = found[0]
    return {**row, "player_index": index}


def _club_names(sessions: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for row in sessions:
        name = str(row.get("organization_name") or row.get("club") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def gap_session_entries(
    *,
    player_id: str,
    license_number: Optional[str],
    last_name: str,
    first_name: str,
    sessions: List[Dict[str, Any]],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    today: Optional[date] = None,
    fetch: Fetch = default_fetch,
) -> List[Dict[str, Any]]:
    """BridgeInterNet sessions inside the ffbridge.fr disclosure window that Lancelot has not published.

    The player is found by name on the page of a club they have played at.
    """
    if not last_name:
        return []
    today = today or date.today()
    if date_to and date_to < today.isoformat():
        return []
    clubs = _club_names(sessions)
    if not clubs:
        return []
    entries: List[Dict[str, Any]] = []
    for organizer in _organizers():
        published = bridgeplus._lancelot_session_dates(organizer["lancelot_series_id"])
        for offset in range(organizer["non_disclosure_days"]):
            session_date = today - timedelta(days=offset)
            day = session_date.isoformat()
            if (date_from and day < date_from) or (date_to and day > date_to) or day in published:
                continue
            spec = bi.game_for_date(session_date)
            if spec is None or spec.series_id != organizer["series_id"]:
                continue
            entry = _find_entry(
                session_date,
                organizer=organizer,
                clubs=clubs,
                player_id=player_id,
                license_number=license_number,
                last_name=last_name,
                first_name=first_name,
                fetch=fetch,
            )
            if entry is not None:
                entries.append(entry)
    return entries


def _find_entry(
    session_date: date,
    *,
    organizer: Dict[str, Any],
    clubs: List[str],
    player_id: str,
    license_number: Optional[str],
    last_name: str,
    first_name: str,
    fetch: Fetch,
) -> Optional[Dict[str, Any]]:
    session_code = bi.session_code_for_date(session_date)
    ranking_url = bi.ranking_url(session_date, "scratch")
    try:
        ranking_html = fetch(ranking_url)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None  # results not posted yet
        raise
    for bi_club in bi.parse_club_links(ranking_html, ranking_url):
        ffb_club = next((name for name in clubs if club_matches(name, bi_club["club_name"])), None)
        if ffb_club is None:
            continue
        club_page_url = bi.club_url(session_date, bi_club["club_code"], "scratch")
        rows = bi.parse_result_rows(
            fetch(club_page_url),
            page_kind="club",
            club_code=bi_club["club_code"],
            club_name=bi_club["club_name"],
        )
        row = find_pair(rows, last_name, first_name)
        if row is None or not row.get("pair_id"):
            continue
        pair = bi.parse_pair_code(row["pair_id"])
        names = [row["player1_name"], row["player2_name"]]
        player_name = names[row["player_index"]]
        partner_name = names[1 - row["player_index"]]
        first_seat = row["player_index"] == 0
        if pair["pair_direction"] == "EW":
            player_direction, partner_direction = ("E", "W") if first_seat else ("W", "E")
        else:
            player_direction, partner_direction = ("N", "S") if first_seat else ("S", "N")
        day = session_date.isoformat()
        return {
            "session_id": bridgeinter_session_id(session_code),
            "date": day,
            "club": ffb_club,
            "organization_id": None,
            "organization_name": ffb_club,
            "group_id": None,
            "competition_label": organizer["label"],
            "session_label": organizer["label"],
            "description": " ".join(part for part in (day, organizer["label"], ffb_club, "BridgeInterNet") if part),
            "series_id": organizer["series_id"],
            "listing_source": "BridgeInterNet organizer pages",
            "source": "bridgeinter",
            "session_code": session_code,
            "bi_club_code": bi_club["club_code"],
            "results_url": club_page_url,
            "route_url": bi.scorecard_url(session_code, row["pair_id"]),
            "bi_pair_id": row["pair_id"],
            "player_id": player_id,
            "player_license_number": bridgeplus._license_key(license_number) if license_number else None,
            "partner_license_number": None,
            "player_name": player_name,
            "partner_name": partner_name,
            "pair_direction": pair["pair_direction"],
            "player_direction": player_direction,
            "partner_direction": partner_direction,
            "section_name": pair["section"],
            "pair_number": str(pair["table"]),
        }
    return None


def _opponent_tables(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """``SININGE Catherine - ZABOTTO Guy`` -> that pair's table number, from the club page."""
    tables = {}
    for row in rows:
        if not row.get("pair_id"):
            continue
        key = " ".join(_tokens(f"{row['player1_name']} {row['player2_name']}"))
        tables[key] = bi.parse_pair_code(row["pair_id"])["table"]
    return tables


def build_mldf(
    card: Dict[str, Any],
    deals: Dict[int, Dict[str, Any]],
    *,
    entry: Dict[str, Any],
    opponent_tables: Dict[str, int],
) -> pl.DataFrame:
    """One row per board for the entry's pair, in the shape Lancelot sessions are built to."""
    direction = entry["pair_direction"]
    our_number = int(entry["pair_number"])
    seats = {
        entry["player_direction"]: (str(entry["player_id"]), entry["player_name"]),
        entry["partner_direction"]: (entry.get("partner_license_number"), entry["partner_name"]),
    }
    rows = []
    for board in card["boards"]:
        board_no = board["board"]
        if board_no not in deals:
            raise ValueError(f"BridgeInterNet deal page is missing board {board_no}")
        deal = deals[board_no]
        top = deal["played"] - 1
        ours = board["pct"] / 100.0
        theirs = 1.0 - ours
        opponent = opponent_tables.get(" ".join(_tokens(board["opponents"] or "")))
        if direction == "EW":
            pct_ns, pct_ew = theirs, ours
            pair_ns, pair_ew = opponent, our_number
        else:
            pct_ns, pct_ew = ours, theirs
            pair_ns, pair_ew = our_number, opponent
        row = {
            "Date": entry["date"],
            "Section_Name": entry.get("section_name"),
            "Board": board_no,
            "PBN": deal["pbn"],
            "Pair_Direction": direction,
            "Dealer": deal["dealer"],
            "Vul": deal["vul"],
            "Declarer": board["Declarer"],
            "Contract": board["Contract"],
            "Result": board["Result"],
            "Lead": board["lead"],
            "Score_NS": board["score_ns"],
            "Score_EW": -board["score_ns"],
            "Pct_NS": pct_ns,
            "Pct_EW": pct_ew,
            "MP_NS": round(pct_ns * top, 2),
            "MP_EW": round(pct_ew * top, 2),
            "MP_Top": top,
            "Pair_Number_NS": pair_ns,
            "Pair_Number_EW": pair_ew,
        }
        for seat in "NESW":
            player_id, name = seats.get(seat, (None, None))
            row[f"Player_ID_{seat}"] = player_id
            row[f"Player_Name_{seat}"] = name
        rows.append(row)
    # Same dtypes as a Lancelot archive fragment; unused seats stay null but typed.
    return pl.DataFrame(rows).with_columns(
        pl.col("Board").cast(pl.Int64),
        pl.col("Result").cast(pl.Int16),
        pl.col("Lead").cast(pl.String),
        pl.col("Declarer").cast(pl.String),
        pl.col("Score_NS").cast(pl.Int16),
        pl.col("Score_EW").cast(pl.Int16),
        pl.col("Pair_Number_NS").cast(pl.Int64),
        pl.col("Pair_Number_EW").cast(pl.Int64),
        pl.col("MP_Top").cast(pl.Int64),
        pl.col("MP_NS").cast(pl.Float64),
        pl.col("MP_EW").cast(pl.Float64),
        *[pl.col(f"Player_ID_{seat}").cast(pl.String) for seat in "NESW"],
        *[pl.col(f"Player_Name_{seat}").cast(pl.String) for seat in "NESW"],
    )


def scrape_mldf(
    entry: Dict[str, Any],
    *,
    fetch: Fetch = default_fetch,
    cache_dir: Optional[pathlib.Path] = None,
) -> pl.DataFrame:
    """Read the pair's scorecard and each board's deal. Deals are cached; results are provisional, so not."""
    session_code = str(entry["session_code"])
    card = bi.parse_scorecard(fetch(str(entry["route_url"])))
    club_rows = bi.parse_result_rows(fetch(str(entry["results_url"])), page_kind="club")
    deals = {}
    for board in card["boards"]:
        board_no = board["board"]
        deals[board_no] = bi.parse_deal_page(
            bridgeplus._page_cache(
                cache_dir,
                f"bridgeinter/{session_code}/donne_{board_no:02d}.html",
                bi.deal_url(session_code, board_no),
                fetch,
            )
        )
    return build_mldf(card, deals, entry=entry, opponent_tables=_opponent_tables(club_rows))
