"""Handicap bonuses of FFBridge handicap simultaneous sessions, from player IVs.

A handicap percentage is the scratch (matchpoint) percentage plus a bonus
set by the pair's IVs. The rules below reproduce every bonus FFBridge
published in the sessions they were checked against (November 2026):

* Octopus (series 386) and Simultanet (series 384): each player earns points
  from their own IV (``PLAYER_POINTS``) and the pair's bonus is the sum.
  Checked on 581 pairs from 5 sessions (Octopus 19, 26 and 29 Oct;
  Simultanet 23 and 30 Oct 2026), no exceptions.
* Roy René on the first Tuesday of the month (series 5): the bonus is
  ``(160 - team IV) / 8``, where team IV is the two players' IVs added.
  Checked on 94 pairs from 6 clubs (6 Oct 2026). Every pair matched except
  at one club (4600071), which gave every pair 0 that day. Team IV 160 got
  0; a team above 160 is given 0 too (none was seen).

The club's session record has no series id, only the club's own label,
so the rule is chosen from the label (and, for Roy René, the date).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date as Date
from typing import Iterable, Optional

PER_PLAYER = "per_player"
ROY_RENE = "roy_rene"

# (lowest IV of the band, points). IV 60 and up earns nothing. The lowest
# IV seen was 20, which earns 8 like 22 and 24.
PLAYER_POINTS = ((60, 0.0), (50, 3.0), (40, 4.0), (36, 5.0), (30, 6.0), (26, 7.0), (0, 8.0))
ROY_RENE_TEAM_IV_ZERO = 160
ROY_RENE_IV_PER_POINT = 8.0


def _tokens(label: Optional[str]) -> set:
    text = unicodedata.normalize("NFKD", str(label or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return {token for token in re.split(r"[^a-z0-9]+", text) if token}


def is_first_tuesday(day: str) -> bool:
    parsed = Date.fromisoformat(str(day)[:10])
    return parsed.weekday() == 1 and parsed.day <= 7


def handicap_rule(label: Optional[str], day: Optional[str]) -> Optional[str]:
    """``PER_PLAYER``, ``ROY_RENE``, or None for a scratch session."""
    tokens = _tokens(label)
    if "octopus" in tokens or "simultanet" in tokens:
        return PER_PLAYER
    if {"roy", "rene"} <= tokens and day and is_first_tuesday(day):
        return ROY_RENE
    return None


def player_points(iv: float) -> float:
    for lowest, points in PLAYER_POINTS:
        if iv >= lowest:
            return points
    raise ValueError(f"IV {iv} is below every band")


def pair_bonus(rule: str, ivs: Iterable[float]) -> float:
    """The pair's bonus in percentage points under ``rule``."""
    values = [float(iv) for iv in ivs]
    if len(values) != 2:
        raise ValueError(f"A pair needs two IVs, got {values}")
    if rule == PER_PLAYER:
        return sum(player_points(iv) for iv in values)
    if rule == ROY_RENE:
        return max(0.0, (ROY_RENE_TEAM_IV_ZERO - sum(values)) / ROY_RENE_IV_PER_POINT)
    raise ValueError(f"Unknown handicap rule {rule!r}")
