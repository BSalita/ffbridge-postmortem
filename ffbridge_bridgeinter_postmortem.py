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

import json
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor
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
                sessions=sessions,
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
    sessions: List[Dict[str, Any]],
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
            # FFBridge groups of the player's games at this club, to find the
            # entry list (player ids) of the same game.
            "club_group_ids": sorted({
                str(s["group_id"]) for s in sessions
                if s.get("group_id") and str(s.get("organization_name") or s.get("club") or "").strip() == ffb_club
            }),
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


def _person_key(name: str) -> str:
    return " ".join(sorted(_tokens(name)))


def _pair_key(text: str) -> str:
    """Order-free key for a pair: club page ``name1 name2`` or a scorecard's ``A - B`` opponents."""
    return " ".join(sorted(_tokens(text)))


def build_club_mldf(
    cards: Dict[str, Dict[str, Any]],
    deals: Dict[int, Dict[str, Any]],
    *,
    entry: Dict[str, Any],
    club_rows: List[Dict[str, Any]],
    player_ids: Dict[str, str],
) -> pl.DataFrame:
    """One row per board and table at the club, built from the North-South pairs' scorecards.

    Like a Lancelot club session, every pair's results are present, so every
    pair can be predicted and ranked. ``player_ids`` maps ``_person_key(name)``
    to a Lancelot id; players it lacks keep a null id.
    """
    by_pair_id = {row["pair_id"]: row for row in club_rows if row.get("pair_id")}
    by_names = {_pair_key(f"{row['player1_name']} {row['player2_name']}"): row for row in by_pair_id.values()}
    ids = dict(player_ids)
    ids[_person_key(entry["player_name"])] = str(entry["player_id"])

    rows = []
    for ns_pair_id, card in cards.items():
        ns_row = by_pair_id[ns_pair_id]
        ns_code = bi.parse_pair_code(ns_pair_id)
        for board in card["boards"]:
            board_no = board["board"]
            if board_no not in deals:
                raise ValueError(f"BridgeInterNet deal page is missing board {board_no}")
            ew_row = by_names.get(_pair_key(board["opponents"] or ""))
            if ew_row is None:
                raise ValueError(
                    f"Board {board_no}: opponents {board['opponents']!r} of {ns_pair_id} are not on the club page"
                )
            deal = deals[board_no]
            top = deal["played"] - 1
            pct_ns = board["pct"] / 100.0
            row = {
                "Date": entry["date"],
                "Section_Name": ns_code["section"],
                "Board": board_no,
                "PBN": deal["pbn"],
                "Pair_Direction": "NS",
                "Dealer": deal["dealer"],
                "Vul": deal["vul"],
                "Declarer": board["Declarer"],
                "Contract": board["Contract"],
                "Result": board["Result"],
                "Lead": board["lead"],
                "Score_NS": board["score_ns"],
                "Score_EW": -board["score_ns"],
                "Pct_NS": pct_ns,
                "Pct_EW": 1.0 - pct_ns,
                "MP_NS": round(pct_ns * top, 2),
                "MP_EW": round((1.0 - pct_ns) * top, 2),
                "MP_Top": top,
                "Pair_Number_NS": ns_code["table"],
                "Pair_Number_EW": bi.parse_pair_code(ew_row["pair_id"])["table"],
            }
            for seat, name in (
                ("N", ns_row["player1_name"]), ("S", ns_row["player2_name"]),
                ("E", ew_row["player1_name"]), ("W", ew_row["player2_name"]),
            ):
                row[f"Player_ID_{seat}"] = ids.get(_person_key(name))
                row[f"Player_Name_{seat}"] = name
            rows.append(row)
    if not rows:
        raise ValueError(f"No BridgeInterNet scorecards for club {entry.get('bi_club_code')}")
    # Same dtypes as a Lancelot archive fragment; unknown ids stay null but typed.
    return pl.DataFrame(rows).sort(["Board", "Pair_Number_NS"]).with_columns(
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


def ranking_payload(
    session: Dict[str, Any],
    *,
    home_club_code: str,
    player_ids: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Every pair of every club, shaped like Lancelot ``results/sessions/{id}/ranking``.

    Ranks are national, over every pair at every club, as on the national
    ranking page; the club pages' rank columns are not that rank. With a
    handicap ranking, sessionScore and rank are handicapped and the scratch
    figures go in totalScoreWithoutHandicap / rankWithoutHandicap, as Lancelot
    does. Pairs at other clubs get placeholder ids so they can never be taken
    for a home pair with the same table number.
    """
    handicap = {row["pair_id"]: row for row in session["club_handicap"] if row.get("pair_id")}
    scratch_field = [row["percentage"] for row in session["club_scratch"] if row.get("pair_id")]
    handicap_field = [row["percentage"] for row in handicap.values()]

    def national_rank(pct: float, field: List[float]) -> int:
        return 1 + sum(1 for other in field if other > pct)

    items = []
    for row in session["club_scratch"]:
        if not row.get("pair_id"):
            continue
        code = bi.parse_pair_code(row["pair_id"])
        home = row["club_code"] == home_club_code
        team = {}
        for index, name in ((1, row["player1_name"]), (2, row["player2_name"])):
            player_id = player_ids.get(_person_key(name)) if home else f"bi:{row['pair_id']}:{index}"
            team[f"player{index}"] = {"id": player_id, "lastName": name, "firstName": ""}
        item = {
            "orientation": code["pair_direction"],
            "section": code["section"],
            "tableNumber": code["table"],
            "simultaneousId": row["club_code"],
            "team": team,
        }
        handicap_row = handicap.get(row["pair_id"])
        if handicap_row is not None:
            item.update(
                sessionScore=handicap_row["percentage"],
                rank=national_rank(handicap_row["percentage"], handicap_field),
                totalScoreWithoutHandicap=row["percentage"],
                rankWithoutHandicap=national_rank(row["percentage"], scratch_field),
                totalBonus=round(handicap_row["percentage"] - row["percentage"], 2),
            )
        else:
            item.update(sessionScore=row["percentage"], rank=national_rank(row["percentage"], scratch_field))
        items.append(item)
    return items


def ranking_path(cache_dir: pathlib.Path, session_id: Any) -> pathlib.Path:
    """Where load_ranking reads an organizer session's ranking."""
    return pathlib.Path(cache_dir) / "rankings" / f"{session_id}.json"


def _player_ids(entry: Dict[str, Any]) -> Dict[str, str]:
    """``_person_key(name)`` -> Lancelot id, from FFBridge's entry list of the same club game."""
    import ffbridge_entry_list as entry_list

    pairs = entry_list.entrants_of_player_session(
        str(entry["player_id"]), entry.get("club_group_ids") or [], str(entry["date"])
    )
    return {_person_key(p["name"]): p["lancelot_id"] for pair in pairs for p in pair["players"]}


def scrape_mldf(
    entry: Dict[str, Any],
    *,
    fetch: Fetch = default_fetch,
    cache_dir: Optional[pathlib.Path] = None,
) -> pl.DataFrame:
    """The whole club's boards from every North-South scorecard, plus the national ranking.

    Deal pages are cached; results are provisional, so scorecards and rankings
    are read fresh. The ranking is written where load_ranking finds it.
    """
    session_code = str(entry["session_code"])
    session = bi.fetch_session_tables(entry["date"], get_text=fetch)
    club_rows = [row for row in session["club_scratch"] if row["club_code"] == entry["bi_club_code"]]
    ns_pair_ids = [
        row["pair_id"] for row in club_rows
        if row.get("pair_id") and bi.parse_pair_code(row["pair_id"])["pair_direction"] == "NS"
    ]

    def card(pair_id: str) -> Dict[str, Any]:
        return bi.parse_scorecard(fetch(bi.scorecard_url(session_code, pair_id)))

    def deal(board_no: int) -> Dict[str, Any]:
        return bi.parse_deal_page(
            bridgeplus._page_cache(
                cache_dir,
                f"bridgeinter/{session_code}/donne_{board_no:02d}.html",
                bi.deal_url(session_code, board_no),
                fetch,
            )
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        cards = dict(zip(ns_pair_ids, executor.map(card, ns_pair_ids)))
        boards = sorted({board["board"] for c in cards.values() for board in c["boards"]})
        deals = dict(zip(boards, executor.map(deal, boards)))
    player_ids = _player_ids(entry)
    frame = build_club_mldf(cards, deals, entry=entry, club_rows=club_rows, player_ids=player_ids)
    if cache_dir is not None:
        path = ranking_path(cache_dir, entry["session_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = ranking_payload(session, home_club_code=entry["bi_club_code"], player_ids=player_ids)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame
