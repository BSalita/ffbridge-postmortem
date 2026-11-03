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


def test_same_section_rows_are_accepted():
    raw = pl.DataFrame(
        {
            "lineup_segment_game_homeTeam_section": ["A", "B", None],
            "lineup_segment_game_awayTeam_section": ["A", "B", None],
        }
    )
    builder._reject_cross_section(raw)


def test_cross_section_rows_are_rejected():
    raw = pl.DataFrame(
        {
            "lineup_segment_game_homeTeam_section": ["A", "A"],
            "lineup_segment_game_awayTeam_section": ["A", "B"],
        }
    )
    with pytest.raises(builder.UnsupportedSessionError):
        builder._reject_cross_section(raw)


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


def test_sd_refresh_picks_sessions_the_cache_can_improve(tmp_path):
    # s1: stored at 10, cache has 40 for both deals -> refresh.
    # s2: stored at 10, one deal still at 10 in the cache -> wait.
    # s3: stored at 40 -> nothing to do.  s4: a deal missing from the cache -> wait.
    stored = {
        "s1": (["a", "b"], [10, 10]),
        "s2": (["c", "d"], [10, 10]),
        "s3": (["e"], [40]),
        "s4": (["f"], [10]),
    }
    flat_rows = {}
    for session_id, (pbns, trials) in stored.items():
        relative = f"fragments/{session_id}.parquet"
        (tmp_path / "fragments").mkdir(exist_ok=True)
        pl.DataFrame({"PBN": pbns, "Probs_Trials": trials}).write_parquet(tmp_path / relative)
        flat_rows[session_id] = {"fragment_path": relative}
    cache = tmp_path / "hrs.parquet"
    pl.DataFrame(
        {"PBN": ["a", "b", "c", "d", "e"], "Probs_Trials": [40, 40, 40, 10, 40]}
    ).write_parquet(cache)
    ids, counts = builder.sd_refresh_session_ids(tmp_path, flat_rows, list(stored), cache, target=40)
    assert ids == {"s1"}
    assert counts == {"below_target": 3, "ready": 1, "waiting_for_cache": 2}


def test_session_hrs_cache_keeps_only_session_deals(monkeypatch):
    cache = pl.DataFrame({"PBN": ["a", "a", "b", "c"], "Dealer": ["N", "E", "S", "W"]})
    monkeypatch.setattr(builder, "_WORKER_HRS_CACHE", cache)
    got = builder._session_hrs_cache(pl.DataFrame({"PBN": ["a", "c", "c", "z"]})).sort("PBN", "Dealer")
    assert got["PBN"].to_list() == ["a", "a", "c"]
    monkeypatch.setattr(builder, "_WORKER_HRS_CACHE", None)
    with pytest.raises(RuntimeError):
        builder._session_hrs_cache(pl.DataFrame({"PBN": ["a"]}))
