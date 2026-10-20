import polars as pl
import pytest

import build_ffbridge_postmortem_archive as builder


def test_adjusted_and_passed_out_boards_are_accepted():
    raw = pl.DataFrame(
        {
            "nsScore": ["420", "", "%Tournoi", "60%+", "PASSE", None],
            "ewScore": ["", "100", "%Tournoi", "40%-", "PASSE", "50"],
        }
    )
    builder._reject_ambiguous_scores(raw)


def test_two_trick_scores_on_one_row_are_rejected():
    raw = pl.DataFrame({"nsScore": ["420", " 100 "], "ewScore": ["", "-100"]})
    with pytest.raises(builder.UnsupportedSessionError):
        builder._reject_ambiguous_scores(raw)
