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
            pl.DataFrame({"Board": [1, 2], "Pct_NS": [0.0, 0.0]}).write_parquet(stale)
            pl.DataFrame({"Board": [1, 2], "Pct_NS": [0.0, 0.55]}).write_parquet(good)
            self.assertFalse(create.cached_frame_usable(stale))
            self.assertTrue(create.cached_frame_usable(good))
            self.assertFalse(create.cached_frame_usable(pathlib.Path(tmp) / "missing.parquet"))


if __name__ == "__main__":
    unittest.main()
