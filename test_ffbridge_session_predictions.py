import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

import ffbridge_session_predictions as predictions


class RankingTests(unittest.TestCase):
    def test_club_rank_then_our_global_rank_then_ffbridge_and_diff(self):
        ranking = pl.DataFrame(
            {
                "sessionScore": [60.0, 40.0, 55.0, 45.0],
                "rank": [1, 4, 2, 3],
                "simultaneousId": ["club", "club", "other", "other"],
                "orientation": ["NS", "EW", "NS", "EW"],
                "section": ["A", "A", "A", "A"],
                "tableNumber": [1, 1, 2, 2],
                "team_player1_firstName": ["Ann", "Ed", "North", "East"],
                "team_player1_lastName": ["North", "East", "Two", "Two"],
                "team_player2_firstName": ["Sue", "Wes", "South", "West"],
                "team_player2_lastName": ["South", "West", "Two", "Two"],
            }
        )
        boards = pl.DataFrame(
            {
                "section_name": ["A"],
                "Pair_Number_NS": [1],
                "Pair_Number_EW": [1],
                "Player_Name_N": ["Ann North"],
                "Player_Name_S": ["Sue South"],
                "Player_Name_E": ["Ed East"],
                "Player_Name_W": ["Wes West"],
                "Pct_NS": [0.60],
                "Pct_NS_Pred": [0.50],
                "Board": [1],
                "Contract": ["3NN"],
                "Declarer": ["N"],
            }
        )
        ranked = predictions.ranking_rows(boards, ranking)
        north = ranked.filter(pl.col("Orientation") == "NS").row(0, named=True)
        east = ranked.filter(pl.col("FFBridge_Global_Rank") == 4).row(0, named=True)
        self.assertEqual(north["Club_Rank"], 1)
        self.assertEqual(north["FFBridge_Global_Rank"], 1)
        # Predicted 50 sits behind the other club's 55, so our global rank is 2.
        self.assertEqual(north["Our_Global_Rank"], 2)
        self.assertEqual(north["Rank_Diff"], 1)
        self.assertEqual(list(ranked.columns).index("Our_Global_Rank") + 1, list(ranked.columns).index("FFBridge_Global_Rank"))
        self.assertEqual(list(ranked.columns).index("FFBridge_Global_Rank") + 1, list(ranked.columns).index("Rank_Diff"))
        self.assertEqual(east["Club_Rank"], 2)
        self.assertAlmostEqual(east["Global_Pct_Pred"], 50.0)

    def test_board_rows_include_prediction_diff(self):
        boards = pl.DataFrame(
            {
                "Board": [1],
                "section_name": ["A"],
                "Pair_Number_NS": [1],
                "Pair_Number_EW": [1],
                "Player_Name_N": ["Ann"],
                "Player_Name_S": ["Sue"],
                "Player_Name_E": ["Ed"],
                "Player_Name_W": ["Wes"],
                "Contract": ["3NN"],
                "Contract_Pred": ["4SN"],
                "Declarer": ["N"],
                "Declarer_Direction_Pred": ["S"],
                "Pct_NS": [0.6],
                "Pct_NS_Pred": [0.5],
                "Pct_Diff": [-0.1],
            }
        )
        rows = predictions.board_prediction_rows(boards)
        self.assertEqual(rows["Contract_Pred"][0], "4SN")
        self.assertEqual(rows["Declarer_Direction"][0], "N")
        self.assertAlmostEqual(rows["Pct_Diff"][0], -0.1)

    def test_missing_models_still_return_published_ranks(self):
        boards = pl.DataFrame(
            {
                "Board": [1],
                "section_name": ["A"],
                "Pair_Number_NS": [1],
                "Pair_Number_EW": [1],
                "Pct_NS": [0.6],
                "Player_Name_N": ["Ann"],
                "Player_Name_S": ["Sue"],
                "Player_Name_E": ["Ed"],
                "Player_Name_W": ["Wes"],
            }
        )
        ranking = pl.DataFrame(
            {
                "sessionScore": [60.0, 40.0],
                "rank": [3, 8],
                "simultaneousId": ["club", "club"],
                "orientation": ["NS", "EW"],
                "section": ["A", "A"],
                "tableNumber": [1, 1],
                "team_player1_firstName": ["Ann", "Ed"],
                "team_player1_lastName": ["", ""],
                "team_player2_firstName": ["Sue", "Wes"],
                "team_player2_lastName": ["", ""],
            }
        )
        ranked = predictions.ranking_rows(boards, ranking)
        north = ranked.filter(pl.col("Orientation") == "NS").row(0, named=True)
        self.assertEqual(north["Club_Rank"], 1)
        self.assertEqual(north["FFBridge_Global_Rank"], 3)
        self.assertIsNone(north["Our_Global_Rank"])
        self.assertIsNone(north["Rank_Diff"])
        with patch.object(predictions, "models_dir", return_value=Path("missing-models")):
            payload = predictions.prediction_payload(
                boards,
                "1",
                Path("."),
                view="rankings",
                limit=20,
            )
        self.assertFalse(payload["predictions_available"])
        self.assertGreaterEqual(payload["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
