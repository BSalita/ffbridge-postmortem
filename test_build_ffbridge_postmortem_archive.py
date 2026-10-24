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


def test_dd_sd_cache_clears_par_and_keeps_dd_sd(tmp_path):
    pipeline = builder._load_quality_pipeline()
    schema = pipeline._official_hrs_schema()
    row = {name: None for name in schema}
    row.update({"PBN": "N:x", "Dealer": "N", "Vul": "None", "DD_N_C": 7, "ParScore": 420, "ParNumber": 2, "Probs_Trials": 10})
    path = tmp_path / "hrs.parquet"
    pl.DataFrame([row], schema=schema).write_parquet(path)
    cache = builder.load_dd_sd_cache(path)
    assert cache.schema == pl.Schema(schema)
    for column in builder.PAR_COLUMNS:
        assert cache[column].null_count() == 1
    assert cache["DD_N_C"].to_list() == [7]
    assert cache["Probs_Trials"].to_list() == [10]


def test_dd_sd_cache_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        builder.load_dd_sd_cache(tmp_path / "missing.parquet")


def test_session_hrs_cache_keeps_only_session_deals(monkeypatch):
    cache = pl.DataFrame({"PBN": ["a", "a", "b", "c"], "Dealer": ["N", "E", "S", "W"]})
    monkeypatch.setattr(builder, "_WORKER_HRS_CACHE", cache)
    got = builder._session_hrs_cache(pl.DataFrame({"PBN": ["a", "c", "c", "z"]})).sort("PBN", "Dealer")
    assert got["PBN"].to_list() == ["a", "a", "c"]
    monkeypatch.setattr(builder, "_WORKER_HRS_CACHE", None)
    with pytest.raises(RuntimeError):
        builder._session_hrs_cache(pl.DataFrame({"PBN": ["a"]}))
