import os
import tempfile
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

    def test_same_table_number_at_two_clubs_keeps_this_clubs_rank(self):
        ranking = pl.DataFrame(
            {
                "sessionScore": [60.0, 50.0, 90.0, 40.0, 40.0, 10.0],
                "rank": [2, 3, 1, 4, 5, 6],
                "simultaneousId": ["A", "A", "B", "B", "A", "B"],
                "orientation": ["NS", "NS", "NS", "NS", "EW", "EW"],
                "section": ["A", "A", "A", "A", "A", "A"],
                "tableNumber": [1, 2, 1, 2, 1, 1],
                "team_player1_id": [1, 3, 5, 7, 11, 13],
                "team_player2_id": [2, 4, 6, 8, 12, 14],
                "team_player1_firstName": ["Ann", "Cara", "Other", "Other", "Ed", "Other"],
                "team_player1_lastName": ["N", "N", "N", "N", "E", "E"],
                "team_player2_firstName": ["Sue", "Dee", "Other", "Other", "Wes", "Other"],
                "team_player2_lastName": ["S", "S", "S", "S", "W", "W"],
            }
        )
        boards = pl.DataFrame(
            {
                "section_name": ["A", "A"],
                "Section_Name": ["A", "A"],
                "Pair_Number_NS": [1, 2],
                "Pair_Number_EW": [1, 1],
                "Player_ID_N": [1, 3],
                "Player_ID_S": [2, 4],
                "Player_ID_E": [11, 11],
                "Player_ID_W": [12, 12],
                "Player_Name_N": ["Ann N", "Cara N"],
                "Player_Name_S": ["Sue S", "Dee S"],
                "Player_Name_E": ["Ed E", "Ed E"],
                "Player_Name_W": ["Wes W", "Wes W"],
                "Pct_NS": [0.60, 0.50],
                "Pct_NS_Pred": [0.55, 0.45],
                "Board": [1, 2],
            }
        )
        attached = predictions._broadcast_ranks(boards, predictions.ranking_rows(boards, ranking))
        ranked = predictions.ranking_rows(boards, ranking)
        other_club = ranked.filter(pl.col("Club") == "B")
        self.assertEqual(other_club["Global_Pct_Pred"].null_count(), other_club.height)
        self.assertEqual(attached.height, boards.height)
        first = attached.filter(pl.col("Player_ID_N") == 1).row(0, named=True)
        second = attached.filter(pl.col("Player_ID_N") == 3).row(0, named=True)
        self.assertEqual(first["Club_Rank_NS"], 1)
        self.assertEqual(second["Club_Rank_NS"], 2)
        self.assertEqual(first["FFBridge_Global_Rank_NS"], 2)
        self.assertIsNotNone(first["Club_Rank_Pred_NS"])
        self.assertIsNotNone(first["Our_Global_Rank_NS"])
        self.assertIsNotNone(first["Rank_Diff_NS"])
        self.assertNotEqual(first["FFBridge_Global_Rank_NS"], 1)

    def test_same_prediction_at_every_table_shares_a_place(self):
        # The deal score is the same at every table. East-West receive the
        # complement, so they share one place and North-South share the next.
        # Neither direction is numbered by table, and the field is not all 1s.
        rows = []
        for board, pct in ((1, 0.42), (2, 0.40)):
            for table in (1, 2):
                rows.append({
                    "Board": board,
                    "section_name": "A",
                    "Pair_Number_NS": table,
                    "Pair_Number_EW": table,
                    "Player_ID_N": table,
                    "Player_ID_S": table + 10,
                    "Player_ID_E": table + 20,
                    "Player_ID_W": table + 30,
                    "Pct_NS_Pred": pct,
                })
        ranked = predictions.ranking_rows(pl.DataFrame(rows), None)
        east = ranked.filter(pl.col("Orientation") == "EW").sort("Table")
        north = ranked.filter(pl.col("Orientation") == "NS").sort("Table")
        self.assertEqual(east["Global_Pct_Pred"].to_list(), [59.0, 59.0])
        self.assertEqual(north["Global_Pct_Pred"].to_list(), [41.0, 41.0])
        self.assertEqual(east["Club_Rank_Pred"].to_list(), [1, 1])
        self.assertEqual(north["Club_Rank_Pred"].to_list(), [3, 3])

    def test_directional_bias_is_centered_before_club_ranking(self):
        rows = []
        for board in (1, 2):
            for table in range(1, 14):
                rows.append({
                    "Board": board,
                    "section_name": "A",
                    "Pair_Number_NS": table,
                    "Pair_Number_EW": table,
                    "Player_ID_N": table,
                    "Player_ID_S": table + 100,
                    "Player_ID_E": table + 200,
                    "Player_ID_W": table + 300,
                    "Player_Name_N": f"N{table}",
                    "Player_Name_S": f"S{table}",
                    "Player_Name_E": f"E{table}",
                    "Player_Name_W": f"W{table}",
                    # This raw model bias would rank all 13 EW pairs ahead of
                    # all 13 NS pairs without session-wide centering.
                    "Pct_NS_Pred": 0.36 + board / 50 + table / 1000,
                })
        biased = pl.DataFrame(rows)
        centered = predictions._recenter_session_predictions(biased)
        self.assertAlmostEqual(centered["Pct_NS_Pred"].mean(), 0.5, places=10)
        self.assertAlmostEqual(
            centered["Pct_NS_Pred"].std(),
            biased["Pct_NS_Pred"].std(),
            places=10,
        )

        ranked = predictions.ranking_rows(centered, None)
        ns_ranks = ranked.filter(pl.col("Orientation") == "NS")["Club_Rank_Pred"].to_list()
        ew_ranks = ranked.filter(pl.col("Orientation") == "EW")["Club_Rank_Pred"].to_list()
        self.assertLess(min(ns_ranks), max(ew_ranks))
        self.assertLess(min(ew_ranks), max(ns_ranks))
        self.assertLessEqual(abs(sum(ns_ranks) - sum(ew_ranks)), 1)

    def test_a_higher_table_prediction_ranks_ahead_of_the_other_tables(self):
        boards = pl.DataFrame({
            "Board": [1, 1],
            "section_name": ["A", "A"],
            "Pair_Number_NS": [1, 2],
            "Pair_Number_EW": [1, 2],
            "Player_ID_N": [1, 2],
            "Player_ID_S": [11, 12],
            "Player_ID_E": [21, 22],
            "Player_ID_W": [31, 32],
            "Pct_NS_Pred": [0.70, 0.30],
        })
        ranked = predictions.ranking_rows(boards, None)
        def place(orientation: str, table: str) -> int:
            return ranked.filter(
                (pl.col("Orientation") == orientation) & (pl.col("Table") == table)
            )["Club_Rank_Pred"][0]

        self.assertEqual(place("NS", "1"), 1)
        self.assertEqual(place("EW", "2"), 1)
        self.assertEqual(place("NS", "2"), 3)
        self.assertEqual(place("EW", "1"), 3)

    def test_contract_type_words_become_the_trained_codes(self):
        frame = pl.DataFrame(
            {
                "CT_N_S": ["Pass", "Partial", "Game", "SSlam", "GSlam"],
                "HCP_N": [10, 11, 12, 13, 14],
                "Dealer": ["N", "E", "S", "W", "N"],
            }
        )
        aligned = predictions._align_features(
            frame,
            {"CT_N_S": "Float32", "HCP_N": "Float32", "Dealer": "Utf8"},
            ["CT_N_S", "HCP_N", "Dealer"],
        )
        self.assertEqual(aligned["CT_N_S"].to_list(), [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(aligned["HCP_N"].to_list(), [10, 11, 12, 13, 14])
        self.assertEqual(aligned["Dealer"].to_list(), ["N", "E", "S", "W", "N"])

    def test_missing_elo_uses_the_models_neutral_training_value(self):
        aligned = predictions._align_features(
            pl.DataFrame({"Elo_NS": [1500.0, None]}),
            {"Elo_NS": "Float32"},
            ["Elo_NS"],
            {"Elo_NS": {"mean": 1425.0}},
        )
        self.assertEqual(aligned["Elo_NS"].to_list(), [1500.0, 1425.0])

    def test_models_dir_skips_a_configured_path_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty"
            found = root / "found"
            empty.mkdir()
            found.mkdir()
            (
                found / "ffbridge_club_predicted_contract_torch_model_schema.json"
            ).write_text("{}", encoding="utf-8")
            with (
                patch.dict(os.environ, {"FFBRIDGE_SAVED_MODELS": str(empty)}),
                patch.object(
                    predictions,
                    "_STAGED_MODELS_DIR",
                    found,
                ),
                patch.object(predictions, "_PUBLISHED_MODELS_DIR", root / "absent"),
                patch.object(predictions, "_HOST_MODELS_DIR", root / "also-absent"),
            ):
                self.assertEqual(predictions.models_dir(), found)

    def test_attach_elo_features_uses_pre_session_player_and_pair_ratings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "elo.results.parquet"
            pl.DataFrame(
                {
                    "tournament_id": ["123", "123"],
                    "player1_id": ["1", "3"],
                    "player2_id": ["2", "4"],
                    "player1_scratch_elo_before": [1500.0, 1300.0],
                    "player2_scratch_elo_before": [1400.0, 1200.0],
                    "scratch_pair_elo": [1475.0, 1275.0],
                }
            ).write_parquet(path)
            boards = pl.DataFrame(
                {
                    "Player_ID_N": ["1"],
                    "Player_ID_S": ["2"],
                    "Player_ID_E": ["3"],
                    "Player_ID_W": ["4"],
                }
            )
            with patch.object(predictions, "_elo_results_path", return_value=path):
                attached = predictions._attach_elo_features(boards, "123")

            self.assertEqual(attached["Elo_N"][0], 1500.0)
            self.assertEqual(attached["Elo_S"][0], 1400.0)
            self.assertEqual(attached["Elo_E"][0], 1300.0)
            self.assertEqual(attached["Elo_W"][0], 1200.0)
            self.assertEqual(attached["Elo_NS"][0], 1475.0)
            self.assertEqual(attached["Elo_EW"][0], 1275.0)


if __name__ == "__main__":
    unittest.main()
