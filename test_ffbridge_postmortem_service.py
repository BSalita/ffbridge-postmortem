import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

import ffbridge_postmortem_create as create
import ffbridge_postmortem_service as svc


def _board_df(north_id: str, south_id: str = "1") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Player_ID_N": [north_id],
            "Player_ID_S": [south_id],
            "Player_ID_E": ["2"],
            "Player_ID_W": ["3"],
            "Player_Name_N": ["Robert SALITA"],
            "Player_Name_S": ["Partner"],
            "Player_Name_E": ["Opp E"],
            "Player_Name_W": ["Opp W"],
            "Declarer_Direction": ["N"],
            "Contract": ["3NT"],
            "Date": ["2026-08-26"],
        }
    )


class PlayerMatchIdsTests(unittest.TestCase):
    def test_expands_license_and_lancelot(self):
        resolved = create.ResolvedPlayer(
            lancelot_id="246273",
            license_number="9500754",
            requested_id="9500754",
            classic_person_id="597539",
        )
        with patch.object(create, "resolve_player", return_value=resolved):
            ids = svc.player_match_ids("9500754")
        self.assertEqual(set(ids), {"246273", "9500754", "597539"})


class PersonalizeAliasTests(unittest.TestCase):
    def test_license_matches_lancelot_player_id_column(self):
        resolved = create.ResolvedPlayer(
            lancelot_id="246273",
            license_number="9500754",
            requested_id="9500754",
            classic_person_id="597539",
        )
        df = _board_df("246273")
        with patch.object(create, "resolve_player", return_value=resolved):
            out, meta = svc.personalize(df, "9500754")
        self.assertTrue(out["Boards_I_Played"][0])
        self.assertEqual(meta["player_id"], "9500754")
        self.assertEqual(meta["matched_player_id"], "246273")
        self.assertEqual(meta["player_direction"], "N")

    def test_lancelot_matches_license_player_id_column(self):
        resolved = create.ResolvedPlayer(
            lancelot_id="246273",
            license_number="9500754",
            requested_id="246273",
            classic_person_id="597539",
        )
        df = _board_df("9500754")
        with patch.object(create, "resolve_player", return_value=resolved):
            out, meta = svc.personalize(df, "246273")
        self.assertTrue(out["Boards_I_Played"][0])
        self.assertEqual(meta["matched_player_id"], "9500754")

    def test_filename_id_is_enough_without_resolve(self):
        df = _board_df("246273")
        with patch.object(create, "resolve_player", side_effect=FileNotFoundError("no index")):
            out, meta = svc.personalize(df, "9500754", extra_ids=["246273"])
        self.assertTrue(out["Boards_I_Played"][0])
        self.assertEqual(meta["player_id"], "9500754")


class LiveLatestGameTests(unittest.TestCase):
    def setUp(self):
        with svc._live_session_entries_lock:
            svc._live_session_entries.clear()

    def test_live_game_metadata_reaches_generation(self):
        live = {
            "player_id": "246273",
            "player_license_number": "9500754",
            "found": True,
            "game": {
                "session_id": "300753",
                "date": "2026-08-31",
                "competition": "Simultané Octopus",
                "group_id": "21333",
                "series_id": 42,
                "club_code": "5802079",
                "club_name": "Bridge Club Levallois Perret",
                "team_id": 15158159,
                "results_url": "https://www.ffbridge.fr/result/300753",
            },
        }
        with (
            patch.object(svc.player_games, "last_game", return_value=live),
            patch.object(
                svc.create,
                "generate_postmortems",
                return_value={"status": "started"},
            ) as generate,
        ):
            svc.last_game("9500754")
            result = svc.generate_postmortems("9500754", session_id="300753")

        self.assertEqual(result["status"], "started")
        entry = generate.call_args.kwargs["session_entry"]
        self.assertEqual(entry["session_id"], "300753")
        self.assertEqual(entry["date"], "2026-08-31")
        self.assertEqual(entry["group_id"], "21333")
        self.assertEqual(entry["team_id"], 15158159)

    def test_direct_generation_discovers_latest_when_index_is_stale(self):
        live = {
            "player_id": "246273",
            "player_license_number": "9500754",
            "found": True,
            "game": {
                "session_id": "300753",
                "date": "2026-08-31",
                "competition": "Simultané Octopus",
                "group_id": "21333",
                "club_code": "5802079",
                "club_name": "Bridge Club Levallois Perret",
                "team_id": 15158159,
            },
        }
        with (
            patch.object(
                svc.create,
                "list_source_sessions",
                return_value={"sessions": [{"session_id": "282839"}]},
            ),
            patch.object(svc.player_games, "last_game", return_value=live),
            patch.object(
                svc.create,
                "generate_postmortems",
                return_value={"status": "started"},
            ) as generate,
        ):
            svc.generate_postmortems("9500754", session_id="300753")

        self.assertEqual(
            generate.call_args.kwargs["session_entry"]["session_id"],
            "300753",
        )


class OpeningLeadTests(unittest.TestCase):
    def test_scores_cache_lead_joins_on_migration_ids(self):
        payload = [
            {
                "boardNumber": 7,
                "lead": "HK",
                "lineup": {
                    "northPlayer": {"id": 10, "migrationId": 100},
                    "eastPlayer": {"id": 2, "migrationId": 102},
                    "southPlayer": {"id": 1, "migrationId": 101},
                    "westPlayer": {"id": 3, "migrationId": 103},
                },
            }
        ]
        frame = pl.DataFrame(
            {
                "Board": [7],
                "Player_ID_N": ["100"],
                "Player_ID_E": ["102"],
                "Player_ID_S": ["101"],
                "Player_ID_W": ["103"],
                "Contract": ["3NT"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            scores = pathlib.Path(tmp) / "scores"
            scores.mkdir()
            (scores / "99_304747.json").write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(svc, "CACHE_DIR", pathlib.Path(tmp)):
                attached = svc.attach_opening_lead(frame, "304747")
        self.assertEqual(attached["Lead"][0], "HK")
        self.assertEqual(attached.height, 1)

    def test_missing_scores_cache_still_exposes_lead_column(self):
        frame = pl.DataFrame({"Board": [1], "Contract": ["PASS"]})
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(svc, "CACHE_DIR", pathlib.Path(tmp)):
                attached = svc.attach_opening_lead(frame, "1")
        self.assertIn("Lead", attached.columns)
        self.assertIsNone(attached["Lead"][0])


if __name__ == "__main__":
    unittest.main()
