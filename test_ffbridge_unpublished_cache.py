import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

import ffbridge_postmortem_create as create

UNPUBLISHED = [
    {"contract": "4S", "nsNote": 0.0, "ewNote": 0.0, "nsScore": ""},
    {"contract": "3NT", "nsNote": 0.0, "ewNote": 0.0, "nsScore": "690"},
]
PUBLISHED = [
    {"contract": "4S", "nsNote": 54.92, "ewNote": 45.08, "nsScore": ""},
    {"contract": "3NT", "nsNote": 52.46, "ewNote": 47.54, "nsScore": "690"},
]


class UnpublishedResultsCacheTests(unittest.TestCase):
    def test_scores_and_rankings_at_zero_percent_are_unpublished(self):
        self.assertTrue(create.scores_unpublished(UNPUBLISHED))
        self.assertFalse(create.scores_unpublished(PUBLISHED))
        self.assertFalse(create.scores_unpublished([]))
        self.assertTrue(create.ranking_unpublished([{"sessionScore": 0.0}, {"sessionScore": 0}]))
        self.assertFalse(create.ranking_unpublished([{"sessionScore": 0.0}, {"sessionScore": 61.2}]))

    def test_unpublished_scores_are_not_cached_and_old_ones_are_downloaded_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = pathlib.Path(tmp)
            with patch.object(create.mlBridgeFFLib, "lancelot_get", return_value=UNPUBLISHED):
                create._fetch_lancelot_json_cached(
                    "results/teams/1/session/300760/scores", "scores/1_300760", None, cache,
                    unpublished=create.scores_unpublished,
                )
            path = cache / "scores" / "1_300760.json"
            self.assertFalse(path.exists())
            # A file cached before this rule existed is replaced once FFBridge has the percentages.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(UNPUBLISHED), encoding="utf-8")
            with patch.object(create.mlBridgeFFLib, "lancelot_get", return_value=PUBLISHED) as get:
                data = create._fetch_lancelot_json_cached(
                    "results/teams/1/session/300760/scores", "scores/1_300760", None, cache,
                    unpublished=create.scores_unpublished,
                )
            get.assert_called_once()
            self.assertEqual(data, PUBLISHED)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), PUBLISHED)

    def test_a_game_built_before_publication_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale = pathlib.Path(tmp) / "df-300760-56959.parquet"
            good = pathlib.Path(tmp) / "df-300759-56959.parquet"
            pl.DataFrame({"Board": [1, 2], "Pct_NS": [0.0, 0.0], "DD_Lead_Cost": [0, 1]}).write_parquet(stale)
            pl.DataFrame({"Board": [1, 2], "Pct_NS": [0.0, 0.55], "DD_Lead_Cost": [0, 1]}).write_parquet(good)
            self.assertFalse(create.cached_frame_usable(stale))
            self.assertTrue(create.cached_frame_usable(good))
            self.assertFalse(create.cached_frame_usable(pathlib.Path(tmp) / "missing.parquet"))

    def test_a_game_built_before_leads_were_scored_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = pathlib.Path(tmp) / "df-271195-56959.parquet"
            pl.DataFrame({"Board": [1, 2], "Pct_NS": [0.45, 0.55], "Lead": ["SA", "H2"]}).write_parquet(old)
            self.assertFalse(create.cached_frame_usable(old))


def _lancelot_tables(pct: float = 0.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Section_Name": ["A", "A"],
            "Board": [1, 2],
            "Pair_Number_NS": [13, 13],
            "Pair_Number_EW": [12, 12],
            "Score_NS": pl.Series([-140, 420], dtype=pl.Int16),
            "Contract": ["3SE", "4HN"],
            "MP_NS": [pct, pct],
            "MP_EW": [pct, pct],
            "MP_Top": pl.Series([0, 0], dtype=pl.Int64),
            "Pct_NS": [pct, pct],
            "Pct_EW": [pct, pct],
        }
    )


def _bridgeplus_tables(scores=(-140, 420)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Section_Name": ["A", "A"],
            "Board": [1, 2],
            "Pair_Number_NS": [13, 13],
            "Pair_Number_EW": [12, 12],
            "Score_NS": list(scores),
            "MP_NS": [203.0, 400.0],
            "MP_EW": [376.0, 179.0],
            "MP_Top": [579, 579],
            "Pct_NS": [0.3507, 0.6908],
            "Pct_EW": [0.6493, 0.3092],
        }
    )


class OrganizerPercentagesTests(unittest.TestCase):
    META = create.LancelotSessionMeta(
        player_id="246273", session_id="288738", partner_id=None,
        player_license_number="9500754", partner_license_number=None,
        player_name="", partner_name="", group_id=None, org_id=None,
        tournament_date="2026-11-03T00:00:00", organization_name=None,
        game_description=None, team_id=None, pair_direction="EW",
        opponent_pair_direction="NS", player_direction="E", partner_direction="W",
        section_name="A", team_number=None, game_url="",
    )
    ENTRY = {"series_id": 5, "organization_id": "5802079"}

    def _fill(self, bridgeplus_tables, series=frozenset({"5"})):
        with (
            patch.object(create.bridgeplus, "organizer_series_ids", return_value=set(series)),
            patch.object(
                create.bridgeplus, "club_session_percentages", return_value=bridgeplus_tables
            ) as fetched,
        ):
            frame = create._fill_organizer_percentages(_lancelot_tables(), self.META, self.ENTRY)
        return frame, fetched

    def test_unscored_simultaneous_takes_bridgeplus_percentages(self):
        frame, fetched = self._fill(_bridgeplus_tables())
        fetched.assert_called_once_with(
            session_date="2026-11-03", club="5802079", license_number="9500754"
        )
        self.assertEqual(frame["Pct_EW"].to_list(), [0.6493, 0.3092])
        self.assertEqual(frame["MP_Top"].to_list(), [579, 579])
        self.assertEqual(frame.schema["MP_Top"], pl.Int64)
        self.assertEqual(frame[create.PCT_SOURCE_COLUMN].to_list(), ["Bridge+", "Bridge+"])
        self.assertEqual(frame.columns[:-1], _lancelot_tables().columns)

    def test_other_series_and_missing_sheets_keep_lancelot_zeros(self):
        frame, fetched = self._fill(_bridgeplus_tables(), series={"3"})
        fetched.assert_not_called()
        self.assertTrue(frame.equals(_lancelot_tables()))
        frame, _ = self._fill(None)
        self.assertTrue(frame.equals(_lancelot_tables()))

    def test_bridgeplus_scores_must_match_lancelot(self):
        with self.assertRaisesRegex(ValueError, "scores differ on 1 tables"):
            self._fill(_bridgeplus_tables(scores=(-140, 450)))
        with self.assertRaisesRegex(ValueError, "no result for 1 of 2 tables"):
            self._fill(_bridgeplus_tables().head(1))


class ArchiveOnlyFinalResultsTests(unittest.TestCase):
    def test_unscored_and_provisional_frames_are_not_archived(self):
        self.assertIsNone(create.unarchivable_reason(_lancelot_tables(pct=0.5)))
        self.assertIn("not published", create.unarchivable_reason(_lancelot_tables()))
        provisional = _lancelot_tables(pct=0.5).with_columns(
            pl.lit("Bridge+").alias(create.PCT_SOURCE_COLUMN)
        )
        self.assertIn("provisional", create.unarchivable_reason(provisional))
        with (
            patch.object(create.postmortem_archive, "archive_session") as archived,
            patch.object(create, "_write_hierarchical_if_configured") as hierarchical,
        ):
            result, tree = create._archive_frame(
                provisional, "288738", context={}, series_id="5"
            )
        archived.assert_not_called()
        hierarchical.assert_not_called()
        self.assertEqual(result, {"archive_file": None, "revision": None})
        self.assertIsNone(tree)

    def test_a_game_with_provisional_percentages_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "df-288738-246273.parquet"
            _lancelot_tables(pct=0.5).with_columns(
                pl.lit(1, dtype=pl.UInt8).alias("DD_Lead_Cost"),
                pl.lit("Bridge+").alias(create.PCT_SOURCE_COLUMN),
            ).write_parquet(path)
            self.assertFalse(create.cached_frame_usable(path))


if __name__ == "__main__":
    unittest.main()
