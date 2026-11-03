"""One-time Bridge+ scrape when ffbridge.fr still points at the organizer.

The public results site redirects a simultaneous to ``simultaneous.website``
while ``today`` is inside ``nonDisclosureDays`` of the session date. Roy René's
website is Bridge+. During that window the Lancelot session list often has no
session for the date, so the pair's route sheet is the only board record.
"""

from __future__ import annotations

import pathlib
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from html import unescape
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import polars as pl
import requests

BRIDGEPLUS_HOST = "bridgeplus.com"
BRIDGEPLUS_FORM_URL = (
    "https://www.bridgeplus.com/nos-simultanes/espace-club/?p=joueur&res=sim"
)
# Marks a frame whose percentages are Bridge+'s, not FFBridge's published ones.
PCT_SOURCE_COLUMN = "Pct_Source"
PCT_SOURCE = "Bridge+"
BRIDGEPLUS_RESULTS_URL ="https://www.bridgeplus.com/nos-simultanes/resultats/"
BRIDGEPLUS_CLUB_URL = (
    "https://www.bridgeplus.com/nos-simultanes/resultats/?p=club/{tournament}/{club}"
)
_USER_AGENT = {"User-Agent": "Mozilla/5.0"}
_SESSION_ID_BASE = 900_000_000
_SUIT = {"P": "S", "C": "H", "K": "D", "T": "C", "SA": "N"}
_SEAT = {"N": "N", "E": "E", "S": "S", "O": "W"}
_RANK = {
    "A": "A",
    "R": "K",
    "D": "Q",
    "V": "J",
    "10": "T",
    "9": "9",
    "8": "8",
    "7": "7",
    "6": "6",
    "5": "5",
    "4": "4",
    "3": "3",
    "2": "2",
}
_HAND_SUIT = {"pique": "S", "coeur": "H", "carreau": "D", "trefle": "C"}
_LEAD_SUIT = {
    "♥": "H",
    "♠": "S",
    "♦": "D",
    "♣": "C",
    "P": "S",
    "C": "H",
    "K": "D",
    "T": "C",
    "H": "H",
    "S": "S",
    "D": "D",
}
_CONTRACT_RE = re.compile(
    r"^([NESO])\s*:\s*(\d)(SA|P|C|K|T)(XX|X)?(=|[+-]\d+)$"
)
_OPTION_RE = re.compile(
    r"<option[^>]*value='(S\d+)'[^>]*>.*?du\s+(\d{2})/(\d{2})/(\d{4})",
    re.S,
)
_ROUTE_RE = re.compile(
    r"(\d{6,8})\s*-\s*(\d{6,8}).{0,500}?href=(['\"])([^'\"]*p=route/[^'\"]+)\3",
    re.S,
)
_HOLDING_RE = re.compile(
    r'class="(pique|coeur|carreau|trefle)"[^>]*>.*?</span>(?:&nbsp;|\s)*'
    r'<span class="gros">([^<]*)</span>',
    re.S,
)

Fetch = Callable[[str], str]
_LOOKUP_TTL_S = 300.0
_lookup_cache: Dict[str, tuple[float, Any]] = {}


def _cached_lookup(key: str, produce):
    now = time.monotonic()
    hit = _lookup_cache.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    value = produce()
    _lookup_cache[key] = (now + _LOOKUP_TTL_S, value)
    return value


def bridgeplus_session_id(tournament_id: str) -> str:
    """Numeric id that cannot collide with a Lancelot session id."""
    digits = str(tournament_id).lstrip("S")
    if not digits.isdigit():
        raise ValueError(f"Bridge+ tournament id {tournament_id!r} is not numeric")
    return str(_SESSION_ID_BASE + int(digits))


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    ).casefold()


def _license_key(value: Any) -> str:
    text = str(value or "").strip().lstrip("0")
    return text or "0"


def disclosure_redirect(
    website: Optional[str],
    non_disclosure_days: Optional[int],
    session_date: date,
    today: date,
) -> Optional[str]:
    """Same window as ffbridge.fr ``getSimultaneousResultsRedirection``.

    The public site sends the player to the organizer while today is on or
    after the session date and still inside ``nonDisclosureDays``.
    """
    if not website or not non_disclosure_days or non_disclosure_days < 1:
        return None
    if today < session_date or today >= session_date + timedelta(days=int(non_disclosure_days)):
        return None
    if website.startswith("http://") or website.startswith("https://"):
        return website
    return f"https://{website}"


def _host_is_bridgeplus(url: str) -> bool:
    host = urlparse(url).netloc.casefold()
    return host == BRIDGEPLUS_HOST or host.endswith("." + BRIDGEPLUS_HOST)


def parse_tournament_options(html: str) -> List[Dict[str, str]]:
    rows = []
    for tournament_id, day, month, year in _OPTION_RE.findall(html):
        rows.append(
            {
                "tournament_id": tournament_id,
                "date": f"{year}-{month}-{day}",
            }
        )
    return rows


_FRENCH_MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
}
_RESULTS_ROW_RE = re.compile(
    r"(\d{1,2})\s+([^\s<]+)\s+(\d{4})[^<]*</div>\s*<div[^>]*>\s*"
    r"<a href='[^']*p=clubs/(S\d+)'",
)


def parse_results_tournaments(html: str) -> List[Dict[str, str]]:
    """Simultaneous ids and dates from the public results list.

    The player form (BRIDGEPLUS_FORM_URL) adds a simultaneous only some days
    after it is played; this list has it the same day.
    """
    rows = []
    for day, month, year, tournament_id in _RESULTS_ROW_RE.findall(html):
        number = _FRENCH_MONTHS.get(_fold(month))
        if number is None:
            raise ValueError(f"Bridge+ results list has an unknown month {month!r}")
        rows.append(
            {"tournament_id": tournament_id, "date": f"{year}-{number:02d}-{int(day):02d}"}
        )
    return rows


def bridgeplus_tournaments(fetch: Fetch) -> List[Dict[str, str]]:
    """Every Bridge+ simultaneous on the results list and the player form, newest id first."""
    found = {
        row["tournament_id"]: row
        for row in [
            *parse_tournament_options(fetch(BRIDGEPLUS_FORM_URL)),
            *parse_results_tournaments(fetch(BRIDGEPLUS_RESULTS_URL)),
        ]
    }
    return [found[key] for key in sorted(found, reverse=True)]


def parse_player_route(html: str, license_number: str) -> Optional[Dict[str, str]]:
    """Club sheet row: padded licenses, then the pair's route link."""
    wanted = _license_key(license_number)
    for first, second, _quote, href in _ROUTE_RE.findall(html):
        if wanted not in {_license_key(first), _license_key(second)}:
            continue
        return {
            "route_url": href,
            "license_1": _license_key(first),
            "license_2": _license_key(second),
        }
    return None


def _lines(html: str) -> List[str]:
    text = unescape(re.sub(r"<[^>]+>", "\n", html))
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


def _pair_header(lines: List[str]) -> Dict[str, str]:
    for line in lines:
        match = re.search(r"\((NS|EO)(\d+)\s+([A-Z])\)", line)
        if match is None:
            continue
        direction, pair_number, section = match.groups()
        names = line[: match.start()].strip(" -")
        return {
            "pair_direction": "NS" if direction == "NS" else "EW",
            "pair_number": pair_number,
            "section": section,
            "pair_names": names,
        }
    raise ValueError("Bridge+ route sheet has no pair header")


def _lead_code(text: str) -> Optional[str]:
    compact = text.replace(" ", "")
    if not compact:
        return None
    if compact[0] in _LEAD_SUIT:
        suit = _LEAD_SUIT[compact[0]]
        rank_text = compact[1:]
    elif compact[-1] in _LEAD_SUIT:
        suit = _LEAD_SUIT[compact[-1]]
        rank_text = compact[:-1]
    else:
        return None
    rank = _RANK.get(rank_text)
    if rank is None:
        return None
    # Suit then rank ("HA"), the Lancelot Lead format the lead scoring reads.
    return suit + rank


_MP_RE = re.compile(r"^-?\d+(?:,\d+)?$")
_PCT_RE = re.compile(r"^(\d+(?:,\d+)?) %$")
_TOP_RE = re.compile(r"^Top int\S*\s*:\s*(\d+)")
_LEAD_SYMBOLS = {"♠", "♥", "♦", "♣"}


def parse_route_boards(html: str) -> Dict[str, Any]:
    """A pair's route sheet: one row per board, grouped under each opponent.

    A board row is number, score, matchpoints, percentage, contract, then the
    lead's suit symbol and rank (absent when no lead was recorded). The score
    is shown from the opponents' side: an EW sheet lists NS's score, an NS
    sheet lists EW's. Matchpoints and percentage are the sheet pair's own.
    """
    lines = _lines(html)
    header = _pair_header(lines)
    top = next(
        (int(match.group(1)) for line in lines if (match := _TOP_RE.match(line))),
        None,
    )
    if top is None:
        raise ValueError("Bridge+ route sheet has no 'Top intégral'")
    sign = 1 if header["pair_direction"] == "EW" else -1
    boards: Dict[int, Dict[str, Any]] = {}
    opponent = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if re.match(r"^(NS|EO)\s+\d+", line):
            opponent = line
            index += 1
            continue
        if (
            re.fullmatch(r"\d{1,2}", line)
            and index + 4 < len(lines)
            and _MP_RE.match(lines[index + 2])
            and _PCT_RE.match(lines[index + 3])
            and (_CONTRACT_RE.match(lines[index + 4]) or lines[index + 4] == "-")
        ):
            board = int(line)
            lead = None
            step = 5
            if index + 6 < len(lines) and lines[index + 5] in _LEAD_SYMBOLS:
                lead = _lead_code(lines[index + 5] + lines[index + 6])
                step = 7
            boards[board] = {
                "board": board,
                "ns_score": sign * int(lines[index + 1]),
                "mp": float(lines[index + 2].replace(",", ".")),
                "pct": float(_PCT_RE.match(lines[index + 3]).group(1).replace(",", ".")),
                # "-": the club recorded no contract for this table.
                "contract": None if lines[index + 4] == "-" else lines[index + 4],
                "lead": lead,
                "opponent": opponent,
            }
            index += step
            continue
        index += 1
    if not boards:
        raise ValueError("Bridge+ route sheet has no board results")
    return {"header": header, "top": top, "boards": boards}


def _suit_holding(text: str) -> str:
    cards = []
    for token in unescape(text).replace("\xa0", " ").split():
        if token in {"-", "–", "—"}:
            continue
        if token not in _RANK:
            raise ValueError(f"Unrecognized card {token!r} on a Bridge+ deal")
        cards.append(_RANK[token])
    return "".join(cards)


def _vulnerability(name: str) -> str:
    folded = _fold(name)
    if folded.startswith("personne") or folded.startswith("aucun"):
        return "None"
    if folded.startswith("ns") or "nord-sud" in folded:
        return "N_S"
    if folded.startswith("eo") or "est-ouest" in folded:
        return "E_W"
    if folded.startswith("tous"):
        return "Both"
    raise ValueError(f"Unrecognized vulnerability {name!r}")


def _dealer(name: str) -> str:
    folded = _fold(name)
    return {"nord": "N", "est": "E", "sud": "S", "ouest": "W"}[folded]


def parse_donne_boards(html: str) -> Dict[int, Dict[str, Any]]:
    parts = re.split(r'<div id="donne-board-(\d+)"', html)
    boards: Dict[int, Dict[str, Any]] = {}
    for index in range(1, len(parts), 2):
        board_no = int(parts[index])
        block = parts[index + 1]
        header = re.search(
            r"Donneur\s*:\s*([^<]+)</div>\s*<div[^>]*>\s*Vuln[^:]*:\s*([^<]+)</div>",
            block,
        )
        if header is None:
            raise ValueError(f"Board {board_no} is missing dealer or vulnerability")
        marker = block.find("Votre résultat")
        if marker < 0:
            marker = block.find("Votre resultat")
        holdings = _HOLDING_RE.findall(block[:marker] if marker >= 0 else block)
        if len(holdings) != 16:
            raise ValueError(
                f"Board {board_no} has {len(holdings)} suit holdings, expected 16"
            )
        seats = {}
        for seat, chunk in zip(("N", "W", "E", "S"), (holdings[i:i + 4] for i in range(0, 16, 4))):
            suits = {code: "" for code in "SHDC"}
            for suit_class, raw in chunk:
                suits[_HAND_SUIT[suit_class]] = _suit_holding(raw)
            hand = ".".join(suits[code] for code in "SHDC")
            if sum(len(part) for part in hand.split(".")) != 13:
                raise ValueError(f"Board {board_no} {seat} does not have 13 cards: {hand}")
            seats[seat] = hand
        top_match = re.search(r"Top\s*=\s*(\d+)", block)
        boards[board_no] = {
            "dealer": _dealer(unescape(header.group(1)).strip()),
            "vul": _vulnerability(unescape(header.group(2)).strip()),
            "pbn": "N:" + " ".join(seats[seat] for seat in "NESW"),
            "mp_top": int(top_match.group(1)) if top_match else None,
        }
    if not boards:
        raise ValueError("Bridge+ deal sheet has no boards")
    return boards


def _contract_fields(contract: Optional[str]) -> Dict[str, Any]:
    if contract is None:
        raise ValueError("Bridge+ route sheet has a board with no recorded contract")
    seat, level, suit, penalty, result = _CONTRACT_RE.match(contract).groups()
    declarer = _SEAT[seat]
    encoded = f"{level}{_SUIT[suit]}{penalty or ''}{declarer}"
    if result == "=":
        tricks = 0
    else:
        tricks = int(result)
    return {"Contract": encoded, "Declarer": declarer, "Result": tricks}


def _opponent_number(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    match = re.match(r"(?:NS|EO)\s+(\d+)", text)
    return int(match.group(1)) if match else None


def build_mldf(
    route: Dict[str, Any],
    deals: Dict[int, Dict[str, Any]],
    *,
    session_date: str,
    player_id: str,
    player_license: str,
    partner_license: str,
    license_1: str,
    license_2: str,
    player_name: str,
    partner_name: str,
) -> pl.DataFrame:
    """One row per board for the route pair. Seats follow Bridge+ name order."""
    header = route["header"]
    direction = header["pair_direction"]
    our_number = int(header["pair_number"])
    first_is_player = _license_key(license_1) == _license_key(player_license)
    if direction == "EW":
        east_id, west_id = (player_id, partner_license) if first_is_player else (partner_license, player_id)
        east_name, west_name = (player_name, partner_name) if first_is_player else (partner_name, player_name)
        north_id = south_id = None
        north_name = south_name = None
    else:
        north_id, south_id = (player_id, partner_license) if first_is_player else (partner_license, player_id)
        north_name, south_name = (player_name, partner_name) if first_is_player else (partner_name, player_name)
        east_id = west_id = None
        east_name = west_name = None
    rows = []
    for board_no in sorted(route["boards"]):
        board = route["boards"][board_no]
        if board_no not in deals:
            raise ValueError(f"Bridge+ deal sheet is missing board {board_no}")
        deal = deals[board_no]
        contract = _contract_fields(board["contract"])
        top = deal["mp_top"]
        pct = board["pct"] / 100.0
        if direction == "EW":
            mp_ew, mp_ns = board["mp"], (None if top is None else top - board["mp"])
            pct_ew, pct_ns = pct, (None if top is None else 1.0 - pct)
            pair_ns, pair_ew = _opponent_number(board["opponent"]), our_number
        else:
            mp_ns, mp_ew = board["mp"], (None if top is None else top - board["mp"])
            pct_ns, pct_ew = pct, (None if top is None else 1.0 - pct)
            pair_ns, pair_ew = our_number, _opponent_number(board["opponent"])
        rows.append(
            {
                "Date": session_date,
                "Section_Name": header["section"],
                "Board": board_no,
                "PBN": deal["pbn"],
                "Pair_Direction": direction,
                "Dealer": deal["dealer"],
                "Vul": deal["vul"],
                "Declarer": contract["Declarer"],
                "Contract": contract["Contract"],
                "Result": contract["Result"],
                "Lead": board["lead"],
                "Score_NS": board["ns_score"],
                "Score_EW": -board["ns_score"],
                "Pct_NS": pct_ns,
                "Pct_EW": pct_ew,
                "MP_NS": mp_ns,
                "MP_EW": mp_ew,
                "MP_Top": top,
                "Pair_Number_NS": pair_ns,
                "Pair_Number_EW": pair_ew,
                "Player_ID_N": north_id,
                "Player_ID_E": east_id,
                "Player_ID_S": south_id,
                "Player_ID_W": west_id,
                "Player_Name_N": north_name,
                "Player_Name_E": east_name,
                "Player_Name_S": south_name,
                "Player_Name_W": west_name,
            }
        )
    # Match a Lancelot archive fragment. Unused seats are null, but an all-null
    # column would otherwise be Null and make Pair_IDs/Pair_Names List(Null).
    return pl.DataFrame(rows).with_columns(
        pl.col("Board").cast(pl.Int64),
        pl.col("Result").cast(pl.Int16),
        pl.col("Lead").cast(pl.String),
        pl.col("Score_NS").cast(pl.Int16),
        pl.col("Score_EW").cast(pl.Int16),
        pl.col("Pair_Number_NS").cast(pl.Int64),
        pl.col("Pair_Number_EW").cast(pl.Int64),
        pl.col("MP_Top").cast(pl.Int64),
        pl.col("MP_NS").cast(pl.Float64),
        pl.col("MP_EW").cast(pl.Float64),
        pl.col("Player_ID_N").cast(pl.String),
        pl.col("Player_ID_E").cast(pl.String),
        pl.col("Player_ID_S").cast(pl.String),
        pl.col("Player_ID_W").cast(pl.String),
        pl.col("Player_Name_N").cast(pl.String),
        pl.col("Player_Name_E").cast(pl.String),
        pl.col("Player_Name_S").cast(pl.String),
        pl.col("Player_Name_W").cast(pl.String),
    )


def default_fetch(url: str) -> str:
    response = requests.get(url, headers=_USER_AGENT, timeout=60)
    response.raise_for_status()
    return response.content.decode("utf-8", "replace")


def _fflib():
    import sys

    root = str(pathlib.Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.append(root)
    import mlBridge.mlBridgeFFLib as lib

    return lib


def _series_organizers() -> List[Dict[str, Any]]:
    return _cached_lookup("bridgeplus-series", _load_series_organizers)


def _load_series_organizers() -> List[Dict[str, Any]]:
    organizers = []
    for series_id in _fflib().LANCELOT_TO_MIGRATION:
        payload = fetch_json(f"competitions/simultaneous/{series_id}")
        website = payload.get("website")
        days = payload.get("nonDisclosureDays")
        if not website or not days:
            continue
        absolute = website if str(website).startswith("http") else f"https://{website}"
        if not _host_is_bridgeplus(absolute):
            continue
        organizers.append(
            {
                "lancelot_series_id": series_id,
                "series_id": payload.get("migrationId"),
                "label": payload.get("label") or "Bridge+",
                "website": website,
                "non_disclosure_days": int(days),
            }
        )
    return organizers


def fetch_json(path: str) -> Dict[str, Any]:
    return _fflib().lancelot_get(path)


def _lancelot_session_dates(series_id: int) -> set[str]:
    return _cached_lookup(f"series-dates-{series_id}", lambda: _load_session_dates(series_id))


def _load_session_dates(series_id: int) -> set[str]:
    page = _fflib().get_simultaneous_sessions_page(series_id, page=1, per_page=80)
    dates = set()
    for item in page.get("items") or []:
        raw = str(item.get("date") or "")
        if len(raw) >= 10:
            dates.add(raw[:10])
    return dates


def _club_codes(sessions: List[Dict[str, Any]]) -> List[str]:
    codes = []
    for row in sessions:
        code = str(row.get("organization_id") or "").strip()
        if code.isdigit() and len(code) >= 6 and code not in codes:
            codes.append(code)
    return codes


def _page_cache(cache_dir: Optional[pathlib.Path], relative: str, url: str, fetch: Fetch) -> str:
    if cache_dir is None:
        return fetch(url)
    path = pathlib.Path(cache_dir) / "organizer" / relative
    if path.is_file():
        return path.read_text(encoding="utf-8")
    html = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return html


def gap_session_entries(
    *,
    player_id: str,
    license_number: Optional[str],
    sessions: List[Dict[str, Any]],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    today: Optional[date] = None,
    fetch: Fetch = default_fetch,
    cache_dir: Optional[pathlib.Path] = None,
) -> List[Dict[str, Any]]:
    """Sessions on Bridge+ whose date is inside the ffbridge.fr disclosure window and absent from Lancelot."""
    if not license_number:
        return []
    today = today or date.today()
    if date_to and date_to < today.isoformat():
        return []
    clubs = _club_codes(sessions)
    if not clubs:
        return []
    club_name = next(
        (str(row.get("organization_name") or row.get("club") or "") for row in sessions if row.get("organization_name") or row.get("club")),
        "",
    )
    entries: List[Dict[str, Any]] = []
    tournaments = bridgeplus_tournaments(fetch)
    for organizer in _series_organizers():
        # Lancelot's series date list can lag; a session already listed for
        # this player is the same game, so it is never added again.
        listed = {
            str(row.get("date") or "")[:10]
            for row in sessions
            if str(row.get("series_id")) == str(organizer["series_id"])
        }
        published = _lancelot_session_dates(organizer["lancelot_series_id"]) | listed
        for tournament in tournaments:
            session_date = date.fromisoformat(tournament["date"])
            if date_from and tournament["date"] < date_from:
                continue
            if date_to and tournament["date"] > date_to:
                continue
            if disclosure_redirect(
                organizer["website"],
                organizer["non_disclosure_days"],
                session_date,
                today,
            ) is None:
                continue
            if tournament["date"] in published:
                continue
            for club in clubs:
                club_html = _page_cache(
                    cache_dir,
                    f"bridgeplus/{tournament['tournament_id']}/{club}/club.html",
                    BRIDGEPLUS_CLUB_URL.format(
                        tournament=tournament["tournament_id"],
                        club=club,
                    ),
                    fetch,
                )
                found = parse_player_route(club_html, license_number)
                if found is None:
                    continue
                route_html = _page_cache(
                    cache_dir,
                    f"bridgeplus/{tournament['tournament_id']}/{club}/route.html",
                    found["route_url"],
                    fetch,
                )
                header = parse_route_boards(route_html)["header"]
                names = [part.strip() for part in re.split(r"\s+-\s+", header["pair_names"])]
                player_name = names[0] if names else ""
                partner_name = names[1] if len(names) > 1 else ""
                first_is_player = found["license_1"] == _license_key(license_number)
                if not first_is_player and found["license_2"] == _license_key(license_number):
                    player_name, partner_name = partner_name, player_name
                partner_license = (
                    found["license_2"] if first_is_player else found["license_1"]
                )
                if header["pair_direction"] == "EW":
                    player_direction, partner_direction = ("E", "W") if first_is_player else ("W", "E")
                else:
                    player_direction, partner_direction = ("N", "S") if first_is_player else ("S", "N")
                entries.append(
                    {
                        "session_id": bridgeplus_session_id(tournament["tournament_id"]),
                        "date": tournament["date"],
                        "club": club_name,
                        "organization_id": club,
                        "organization_name": club_name,
                        "group_id": None,
                        "competition_label": organizer["label"],
                        "session_label": organizer["label"],
                        "description": " ".join(
                            part
                            for part in (
                                tournament["date"],
                                organizer["label"],
                                club_name,
                                "Bridge+",
                            )
                            if part
                        ),
                        "series_id": organizer["series_id"],
                        "listing_source": "Bridge+ organizer sheet",
                        "source": "bridgeplus",
                        "results_url": found["route_url"],
                        "route_url": found["route_url"],
                        "tournament_id": tournament["tournament_id"],
                        "player_id": player_id,
                        "player_license_number": _license_key(license_number),
                        "partner_license_number": partner_license,
                        "license_1": found["license_1"],
                        "license_2": found["license_2"],
                        "player_name": player_name,
                        "partner_name": partner_name,
                        "pair_direction": header["pair_direction"],
                        "player_direction": player_direction,
                        "partner_direction": partner_direction,
                        "section_name": header["section"],
                        "pair_number": header["pair_number"],
                    }
                )
                break
    return entries


def scrape_mldf(
    entry: Dict[str, Any],
    *,
    fetch: Fetch = default_fetch,
    cache_dir: Optional[pathlib.Path] = None,
    route_html: Optional[str] = None,
    donne_html: Optional[str] = None,
) -> pl.DataFrame:
    """Read the route and deal sheets into a provisional board frame.

    The deal sheet is cached. The route sheet is downloaded every time: its
    percentages change while other clubs play the deals. The frame carries
    Pct_Source="Bridge+" so it is rebuilt on every load and never archived.
    """
    route_url = str(entry["route_url"])
    tournament = str(entry.get("tournament_id") or "event")
    club = str(entry.get("organization_id") or "club")
    if route_html is None:
        route_html = fetch(route_url)
    route = parse_route_boards(route_html)
    donne_url = route_url.replace("p=route/", "p=donne/")
    if donne_html is None:
        donne_html = _page_cache(
            cache_dir,
            f"bridgeplus/{tournament}/{club}/donne.html",
            donne_url,
            fetch,
        )
    deals = parse_donne_boards(donne_html)
    return build_mldf(
        route,
        deals,
        session_date=str(entry["date"]),
        player_id=str(entry["player_id"]),
        player_license=str(entry["player_license_number"]),
        partner_license=str(entry.get("partner_license_number") or ""),
        license_1=str(entry.get("license_1") or entry["player_license_number"]),
        license_2=str(entry.get("license_2") or entry.get("partner_license_number") or ""),
        player_name=str(entry.get("player_name") or ""),
        partner_name=str(entry.get("partner_name") or ""),
    ).with_columns(pl.lit(PCT_SOURCE).alias(PCT_SOURCE_COLUMN))


def organizer_series_ids() -> set[str]:
    """FFBridge migration series ids whose simultaneous is scored on Bridge+."""
    return {str(row["series_id"]) for row in _series_organizers() if row.get("series_id") is not None}


def club_session_percentages(
    *,
    session_date: str,
    club: str,
    license_number: str,
    fetch: Fetch = default_fetch,
    workers: int = 8,
) -> Optional[pl.DataFrame]:
    """Bridge+ matchpoints for every table a club played in one simultaneous.

    Returns one row per (Section_Name, Board, Pair_Number_NS, Pair_Number_EW)
    with Score_NS, MP_NS, MP_EW, MP_Top, Pct_NS and Pct_EW, or None when no
    Bridge+ simultaneous that day lists the player at this club. Pair numbers
    drop Bridge+'s 100 (NS) / 200 (EW) prefix to match Lancelot table numbers.

    Pages are never cached: the same deals are played at other clubs over
    several days, so the field and the percentages keep changing.
    """
    tournaments = [row for row in bridgeplus_tournaments(fetch) if row["date"] == session_date]
    for tournament in tournaments:
        club_html = fetch(
            BRIDGEPLUS_CLUB_URL.format(tournament=tournament["tournament_id"], club=club)
        )
        if parse_player_route(club_html, license_number) is None:
            continue
        hrefs = [href for _first, _second, _quote, href in _ROUTE_RE.findall(club_html)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            routes = list(pool.map(lambda href: parse_route_boards(fetch(href)), hrefs))
        rows = []
        for route in routes:
            header = route["header"]
            own_ns = header["pair_direction"] == "NS"
            pair = int(header["pair_number"]) % 100
            top = route["top"]
            for board in route["boards"].values():
                opponent = _opponent_number(board["opponent"])
                if opponent is None:
                    raise ValueError(f"Bridge+ board {board['board']} has no opponent pair")
                own_pct = board["pct"] / 100.0
                rows.append(
                    {
                        "Section_Name": header["section"],
                        "Board": board["board"],
                        "Pair_Number_NS": pair if own_ns else opponent % 100,
                        "Pair_Number_EW": opponent % 100 if own_ns else pair,
                        "Score_NS": board["ns_score"],
                        "MP_NS": board["mp"] if own_ns else top - board["mp"],
                        "MP_EW": top - board["mp"] if own_ns else board["mp"],
                        "MP_Top": top,
                        "Pct_NS": own_pct if own_ns else 1.0 - own_pct,
                        "Pct_EW": 1.0 - own_pct if own_ns else own_pct,
                    }
                )
        key = ["Section_Name", "Board", "Pair_Number_NS", "Pair_Number_EW"]
        frame = pl.DataFrame(rows)
        # Both pairs' sheets list each table: their scores must agree.
        clash = frame.group_by(key).agg(pl.col("Score_NS").n_unique().alias("scores")).filter(
            pl.col("scores") > 1
        )
        if clash.height:
            raise ValueError(f"Bridge+ sheets disagree on {clash.height} table scores: {clash.head(3).to_dicts()}")
        return frame.unique(subset=key, keep="first", maintain_order=True)
    return None
