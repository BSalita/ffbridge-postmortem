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


if __name__ == "__main__":
    unittest.main()
