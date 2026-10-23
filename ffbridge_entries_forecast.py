"""Pre-game forecast: a session's ranking from its entries, before anyone plays.

Forecast (this module, ``*_Forecast`` fields) means before seating, from pair Elo
against the entered field. Prediction (``ffbridge_session_predictions``,
``*_Pred`` fields) means the board model, after the session is seated and played.

The board-level models in ``ffbridge_session_predictions`` need the deals, so
they only work once a session exists. This module needs only the field: the
pairs registered for the session (the "inscriptions" on ffbridge.fr).

For each pair:

* Each player's rating is their latest scratch Elo after their last session
  before ``on_date`` (the pre-session rating the next build would use).
* The pair's rating is the mean of the two, as in ``_attach_elo_features``.
* A player with no earlier session gets the field's median player rating and
  the pair is flagged ``Unrated_Players`` and ``Low_Confidence``.
* The predicted percentage is ``PCT_INTERCEPT + PCT_PER_ELO * (pair Elo -
  field mean pair Elo)``, clipped to 0-100.
* ``PCT_RESIDUAL_SD`` is the spread of actual session percentages around
  that line. The 95% range is the prediction plus or minus 1.96 of it.

Ranks come from ``sims`` simulated sessions in which every pair's percentage
is drawn independently around its prediction. ``Rank_Forecast`` is the place of
the predicted percentage in the field; ``Expected_Rank``, ``Rank_Forecast_Low``
and ``Rank_Forecast_High`` (the 5th and 95th percentile places), ``P_Win`` and
``P_Top3`` come from the simulation. Ties in the predicted percentage share a
place. With ``orientation`` set on every pair, places are within NS and EW.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

import ffbridge_session_predictions as sp

# Calibrated on the FFBridge Elo build of 2026-10-23, last 12 months, every
# club field of 6 or more pairs with both players rated: 325,185 pair
# sessions. Least squares of the session percentage on (pair Elo - club field
# mean pair Elo): slope 0.010028 percentage points per Elo point, intercept
# 50.14, R^2 0.319, residual SD 5.954. Recompute with ``calibration_sql``.
PCT_PER_ELO = 0.010028
PCT_INTERCEPT = 50.0
PCT_RESIDUAL_SD = 5.954
Z95 = sp.Z95
DEFAULT_SIMS = 20000
MAX_SIMS = 200000
MAX_PAIRS = 400
RANK_LOW_Q = 0.05
RANK_HIGH_Q = 0.95
ORIENTATIONS = ("NS", "EW")

ENTRY_COLUMNS = (
    "Pair",
    "Orientation",
    "Rank_Forecast",
    "Expected_Rank",
    "Rank_Forecast_Low",
    "Rank_Forecast_High",
    "P_Win",
    "P_Top3",
    "Pct_Forecast",
    "Pct_Forecast_Low",
    "Pct_Forecast_High",
    "Pair_Elo",
    "Player1",
    "Player2",
    "Player1_Elo",
    "Player2_Elo",
    "Player1_License",
    "Player2_License",
    "Low_Confidence",
    "Unrated_Players",
    "Min_Prior_Sessions",
    "Pair_Prior_Sessions",
    "Ratings_Age_Days",
)

calibration_sql = """
WITH r AS (
  SELECT tournament_id, club_code, Pct_Used AS y,
    (player1_scratch_elo_before + player2_scratch_elo_before) / 2 AS pe
  FROM self
  WHERE Pct_Used IS NOT NULL
    AND player1_scratch_elo_before IS NOT NULL
    AND player2_scratch_elo_before IS NOT NULL
), g AS (
  SELECT tournament_id, club_code, avg(pe) AS m, count(*) AS n FROM r GROUP BY 1, 2
), d AS (
  SELECT r.y, r.pe - g.m AS x FROM r JOIN g USING (tournament_id, club_code) WHERE g.n >= 6
), f AS (
  SELECT regr_slope(y, x) AS b, regr_intercept(y, x) AS a, regr_r2(y, x) AS r2, count(*) AS n FROM d
)
SELECT f.*, (SELECT stddev_samp(d.y - (f.a + f.b * d.x)) FROM d) AS resid_sd FROM f
"""


# ---------------------------------------------------------------------------
# Player identity
# ---------------------------------------------------------------------------


def normalize_name(value: Any) -> str:
    """Accent-free, case-free, order-free name key: 'SALITA Robert' == 'Robert Salita'."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
    return " ".join(sorted(tokens))


def _digits(value: Any) -> str:
    token = sp._id_token_value(value)
    return token if token.isdigit() else ""


@dataclass
class PlayerRecord:
    key: str
    name: Optional[str]
    license: Optional[str]
    lancelot_id: Optional[str]
    elo: Optional[float]
    last_day: Optional[str]
    sessions: int


_SIDE_COLUMNS = {
    "id": "{p}_id",
    "lancelot": "{p}_lancelot_id",
    "license": "{p}_license_number",
    "name": "{p}_name",
    "elo": "{p}_scratch_elo_after",
}


def _player_long(results: pl.LazyFrame, before_day: str) -> pl.DataFrame:
    """One row per player per earlier session, from both seats of each pair row."""
    names = set(results.collect_schema().names())
    frames = []
    for prefix in ("player1", "player2"):
        cols = {key: template.format(p=prefix) for key, template in _SIDE_COLUMNS.items()}
        if cols["id"] not in names or cols["elo"] not in names:
            raise ValueError(f"FFBridge Elo results lack {cols['id']} or {cols['elo']}")

        def optional(key: str) -> pl.Expr:
            column = cols[key]
            if column in names:
                return sp._id_token(column) if key != "name" else pl.col(column).cast(pl.Utf8)
            return pl.lit(None, dtype=pl.Utf8)

        frames.append(
            results.filter(sp._ELO_DAY < before_day).select(
                sp._id_token(cols["id"]).alias("key"),
                optional("lancelot").alias("lancelot"),
                optional("license").alias("license"),
                optional("name").alias("name"),
                pl.col(cols["elo"]).cast(pl.Float64).alias("elo"),
                sp._ELO_DAY.alias("day"),
                pl.col("tournament_id").cast(pl.Utf8).alias("session"),
            )
        )
    return pl.concat(frames).filter(pl.col("key") != "").collect()


def _player_table(long: pl.DataFrame) -> pl.DataFrame:
    rated = long.filter(pl.col("elo").is_not_null())
    latest = (
        rated.sort("day")
        .group_by("key")
        .agg(
            pl.col("elo").last(),
            pl.col("day").max().alias("last_day"),
            pl.col("name").drop_nulls().last(),
            pl.col("license").filter(pl.col("license") != "").drop_nulls().last(),
            pl.col("lancelot").filter(pl.col("lancelot") != "").drop_nulls().last(),
        )
    )
    counts = long.group_by("key").agg(pl.col("session").n_unique().cast(pl.Int64).alias("sessions"))
    return latest.join(counts, on="key", how="left").with_columns(
        pl.col("license").map_elements(_digits, return_dtype=pl.Utf8).alias("license_key"),
        pl.col("lancelot").map_elements(_digits, return_dtype=pl.Utf8).alias("lancelot_key"),
        pl.col("name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("name_key"),
    )


class PlayerIndex:
    """Players from the Elo results, findable by licence, Lancelot id, person id, or name."""

    def __init__(self, table: pl.DataFrame):
        self.records: Dict[str, PlayerRecord] = {}
        self.by_license: Dict[str, str] = {}
        self.by_lancelot: Dict[str, str] = {}
        self.by_name: Dict[str, List[str]] = {}
        for row in table.iter_rows(named=True):
            record = PlayerRecord(
                key=row["key"],
                name=row.get("name"),
                license=row.get("license_key") or None,
                lancelot_id=row.get("lancelot_key") or None,
                elo=row.get("elo"),
                last_day=row.get("last_day"),
                sessions=int(row.get("sessions") or 0),
            )
            self.records[record.key] = record
            if record.license:
                self.by_license[record.license] = record.key
            if record.lancelot_id:
                self.by_lancelot[record.lancelot_id] = record.key
            if row.get("name_key"):
                self.by_name.setdefault(row["name_key"], []).append(record.key)

    def resolve(self, query: Any) -> Tuple[Optional[PlayerRecord], Optional[str]]:
        """The player and a warning, or (None, reason) when nobody matches."""
        text = str(query or "").strip()
        if not text:
            return None, "empty player"
        digits = _digits(text)
        if digits:
            for lookup in (self.by_license, self.by_lancelot):
                if digits in lookup:
                    return self.records[lookup[digits]], None
            if digits in self.records:
                return self.records[digits], None
            return None, f"no rated player with licence or id {text}"
        keys = self.by_name.get(normalize_name(text), [])
        if not keys:
            return None, f"no rated player named {text}"
        if len(keys) == 1:
            return self.records[keys[0]], None
        chosen = max(keys, key=lambda key: (self.records[key].last_day or "", self.records[key].sessions))
        others = ", ".join(
            f"{self.records[key].license or key}" for key in keys if key != chosen
        )
        return self.records[chosen], f"{text} is ambiguous; used the most recently active one (others: {others}). Pass a licence to choose."


def _pair_sessions(long: pl.DataFrame, pairs: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
    wanted = {tuple(sorted(pair)) for pair in pairs if pair[0] and pair[1]}
    if not wanted:
        return {}
    players = {key for pair in wanted for key in pair}
    sessions = (
        long.filter(pl.col("key").is_in(list(players)))
        .group_by("session")
        .agg(pl.col("key").unique().alias("keys"))
    )
    counts: Dict[Tuple[str, str], int] = {pair: 0 for pair in wanted}
    for keys in sessions["keys"].to_list():
        present = set(keys)
        for pair in wanted:
            if pair[0] in present and pair[1] in present:
                counts[pair] += 1
    return counts


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _parse_day(on_date: Optional[str]) -> str:
    day = sp._parse_iso_date(on_date)
    return day or Date.today().isoformat()


def _entry_players(entry: Any) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Two player queries, an orientation, and a label from one entry."""
    orientation = None
    label = None
    if isinstance(entry, dict):
        players = entry.get("players")
        if players is None:
            players = [entry.get("player1"), entry.get("player2")]
        orientation = entry.get("orientation")
        label = entry.get("label")
    elif isinstance(entry, str):
        players = [part for part in re.split(r"\s*(?:/|\+|;|\s-\s|&)\s*", entry) if part.strip()]
    else:
        players = list(entry or [])
    players = [str(player).strip() for player in players if str(player or "").strip()]
    if len(players) != 2:
        raise ValueError(f"Each entry needs exactly two players: {entry!r}")
    if orientation is not None:
        orientation = str(orientation).strip().upper().replace("-", "")
        if orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be NS or EW, not {orientation!r}")
    return players, orientation, label


def predicted_pct(pair_elo: np.ndarray) -> np.ndarray:
    centred = pair_elo - pair_elo.mean()
    return np.clip(PCT_INTERCEPT + PCT_PER_ELO * centred, 0.0, 100.0)


def _ranks_desc(values: np.ndarray) -> np.ndarray:
    """Place of each value, 1 = highest; equal values share the better place."""
    return np.array([1 + int(np.sum(values > value)) for value in values], dtype=int)


def simulate_ranks(
    mean_pct: np.ndarray,
    sd: float,
    sims: int,
    seed: Optional[int] = 0,
) -> Dict[str, np.ndarray]:
    """Rank distribution from ``sims`` sessions with independent normal percentages."""
    n = len(mean_pct)
    if n == 0:
        empty = np.zeros(0)
        return {"expected": empty, "low": empty, "high": empty, "p_win": empty, "p_top3": empty}
    rng = np.random.default_rng(seed)
    draws = mean_pct[None, :] + rng.standard_normal((sims, n)) * sd
    order = np.argsort(-draws, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(sims)[:, None]
    ranks[rows, order] = np.arange(1, n + 1)[None, :]
    return {
        "expected": ranks.mean(axis=0),
        "low": np.quantile(ranks, RANK_LOW_Q, axis=0, method="lower"),
        "high": np.quantile(ranks, RANK_HIGH_Q, axis=0, method="higher"),
        "p_win": (ranks == 1).mean(axis=0),
        "p_top3": (ranks <= 3).mean(axis=0),
    }


def _round(value: Optional[float], digits: int) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def forecast_entries(
    entries: Sequence[Any],
    on_date: Optional[str] = None,
    *,
    sims: int = DEFAULT_SIMS,
    seed: Optional[int] = 0,
    results: Optional[pl.LazyFrame] = None,
) -> Dict[str, Any]:
    """Predicted percentages, places and win chances for a list of entered pairs.

    ``entries`` items are two players each: ``["01353276", "SALITA Robert"]``,
    ``"JACOUPY Christian / SALITA Robert"``, or ``{"players": [...],
    "orientation": "NS", "label": "..."}``. A player is a licence number,
    Lancelot id, FFBridge person id, or name.
    """
    if not entries:
        raise ValueError("entries is empty: pass the pairs registered for the session")
    if len(entries) > MAX_PAIRS:
        raise ValueError(f"At most {MAX_PAIRS} pairs")
    sims = int(sims)
    if not 100 <= sims <= MAX_SIMS:
        raise ValueError(f"sims must be between 100 and {MAX_SIMS}")
    day = _parse_day(on_date)
    if results is None:
        results = sp._lancelot_results()
    long = _player_long(results, day)
    index = PlayerIndex(_player_table(long))

    parsed = [_entry_players(entry) for entry in entries]
    orientations = [orientation for _, orientation, _ in parsed]
    if any(orientations) and not all(orientations):
        raise ValueError("Give an orientation for every pair or for none")

    warnings: List[str] = []
    resolved: List[List[Optional[PlayerRecord]]] = []
    for players, _, _ in parsed:
        row: List[Optional[PlayerRecord]] = []
        for query in players:
            record, note = index.resolve(query)
            if note:
                warnings.append(note)
            row.append(record)
        resolved.append(row)

    seen: Dict[str, int] = {}
    for position, row in enumerate(resolved):
        for record in row:
            if record is None:
                continue
            if record.key in seen and seen[record.key] != position:
                raise ValueError(f"{record.name or record.key} is entered in two pairs")
            seen[record.key] = position

    known = [record.elo for row in resolved for record in row if record is not None and record.elo is not None]
    if not known:
        raise ValueError("None of the entered players has an FFBridge Elo rating before the session")
    median_elo = float(np.median(known))

    player_elos = np.array(
        [
            [record.elo if record is not None and record.elo is not None else median_elo for record in row]
            for row in resolved
        ],
        dtype=float,
    )
    pair_elo = player_elos.mean(axis=1)
    pair_counts = _pair_sessions(
        long,
        [tuple(record.key if record else "" for record in row) for row in resolved],
    )

    groups: Dict[Optional[str], List[int]] = {}
    for position, orientation in enumerate(orientations):
        groups.setdefault(orientation, []).append(position)

    pct = np.zeros(len(resolved))
    expected = np.zeros(len(resolved))
    rank_low = np.zeros(len(resolved), dtype=int)
    rank_high = np.zeros(len(resolved), dtype=int)
    p_win = np.zeros(len(resolved))
    p_top3 = np.zeros(len(resolved))
    rank_pred = np.zeros(len(resolved), dtype=int)
    for offset, positions in enumerate(groups.values()):
        idx = np.array(positions)
        group_pct = predicted_pct(pair_elo[idx])
        pct[idx] = group_pct
        rank_pred[idx] = _ranks_desc(np.round(group_pct, 2))
        sim = simulate_ranks(group_pct, PCT_RESIDUAL_SD, sims, None if seed is None else seed + offset)
        expected[idx] = sim["expected"]
        rank_low[idx] = sim["low"]
        rank_high[idx] = sim["high"]
        p_win[idx] = sim["p_win"]
        p_top3[idx] = sim["p_top3"]

    day_date = datetime.strptime(day, "%Y-%m-%d").date()
    rows: List[Dict[str, Any]] = []
    for position, ((players, orientation, label), row) in enumerate(zip(parsed, resolved)):
        names = [
            (record.name if record and record.name else query)
            for query, record in zip(players, row)
        ]
        unrated = [query for query, record in zip(players, row) if record is None or record.elo is None]
        prior = [record.sessions if record else 0 for record in row]
        key = tuple(sorted(record.key if record else "" for record in row))
        pair_prior = pair_counts.get(key, 0) if all(row) else 0
        ages = [
            (day_date - datetime.strptime(record.last_day, "%Y-%m-%d").date()).days
            for record in row
            if record is not None and record.last_day
        ]
        age = max(ages) if ages else None
        spread = sp._prediction_spread(float(pct[position]), None, min(prior), pair_prior, age)
        low_confidence = bool(spread["Low_Confidence"]) or bool(unrated)
        rows.append(
            {
                "Pair": label or f"{names[0]} - {names[1]}",
                "Orientation": orientation,
                "Rank_Forecast": int(rank_pred[position]),
                "Expected_Rank": _round(expected[position], 1),
                "Rank_Forecast_Low": int(rank_low[position]),
                "Rank_Forecast_High": int(rank_high[position]),
                "P_Win": _round(100.0 * p_win[position], 1),
                "P_Top3": _round(100.0 * p_top3[position], 1),
                "Pct_Forecast": _round(pct[position], 2),
                "Pct_Forecast_Low": _round(max(0.0, pct[position] - Z95 * PCT_RESIDUAL_SD), 2),
                "Pct_Forecast_High": _round(min(100.0, pct[position] + Z95 * PCT_RESIDUAL_SD), 2),
                "Pair_Elo": _round(pair_elo[position], 1),
                "Player1": names[0],
                "Player2": names[1],
                "Player1_Elo": _round(row[0].elo if row[0] else None, 1),
                "Player2_Elo": _round(row[1].elo if row[1] else None, 1),
                "Player1_License": row[0].license if row[0] else None,
                "Player2_License": row[1].license if row[1] else None,
                "Low_Confidence": low_confidence,
                "Unrated_Players": unrated or None,
                "Min_Prior_Sessions": int(min(prior)),
                "Pair_Prior_Sessions": int(pair_prior),
                "Ratings_Age_Days": age,
            }
        )
    rows.sort(key=lambda item: (item["Orientation"] or "", item["Rank_Forecast"], -(item["P_Win"] or 0.0), item["Pair"]))
    path = sp._elo_results_path()
    return {
        "date": day,
        "pairs": len(rows),
        "columns": list(ENTRY_COLUMNS),
        "rows": rows,
        "warnings": warnings,
        "provenance": {
            "prediction_source": sp.FALLBACK_SOURCE,
            "ratings_as_of": max((r.last_day for row in resolved for r in row if r and r.last_day), default=None),
            "elo_results_built_at": sp._file_built_at(path),
            "model": "pair Elo vs field mean, linear",
            "pct_per_elo": PCT_PER_ELO,
            "pct_residual_sd": PCT_RESIDUAL_SD,
            "unrated_player_elo": round(median_elo, 1),
            "sims": sims,
        },
    }
