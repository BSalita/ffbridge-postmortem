import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl

import ffbridge_session_predictions as predictions


class ProvenanceTests(unittest.TestCase):
    def test_reports_newest_checkpoint_and_history_build_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            stamps = {
                "ffbridge_club_predicted_declarer_direction_torch_model.pth": 1_760_900_000,
                "ffbridge_club_predicted_contract_torch_model.pth": 1_760_920_000,
                predictions.PLAYER_HISTORY_NAME: 1_760_890_000,
            }
            for name, stamp in stamps.items():
                (directory / name).write_bytes(b"x")
                os.utime(directory / name, (stamp, stamp))
            with patch.object(predictions, "_elo_results_path", return_value=None):
                provenance = predictions.prediction_provenance(directory)
        self.assertEqual(provenance["model_version"], "2025-10-20T00:26:40+00:00")
        self.assertEqual(
            provenance["models_built_at"],
            {
                "Declarer_Direction": "2025-10-19T18:53:20+00:00",
                "Contract": "2025-10-20T00:26:40+00:00",
                "Pct_NS": None,
            },
        )
        self.assertEqual(provenance["player_history_built_at"], "2025-10-19T16:06:40+00:00")
        self.assertIsNone(provenance["pair_history_built_at"])
        self.assertIsNone(provenance["elo_results_built_at"])


class SpreadAndConfidenceTests(unittest.TestCase):
    def test_spread_shrinks_with_boards_and_stays_inside_0_100(self):
        spread = predictions._prediction_spread(58.04, 24, 281)
        self.assertEqual(spread["Pct_Pred_SD"], 5.8)
        self.assertEqual(spread["Pct_Pred_Low"], 46.67)
        self.assertEqual(spread["Pct_Pred_High"], 69.41)
        self.assertFalse(spread["Low_Confidence"])
        edge = predictions._prediction_spread(99.0, 8, 19)
        self.assertEqual(edge["Pct_Pred_High"], 100.0)
        self.assertTrue(edge["Low_Confidence"])
        empty = predictions._prediction_spread(0.0, None, None)
        self.assertIsNone(empty["Pct_Pred_SD"])
        self.assertIsNone(empty["Low_Confidence"])

    def test_prior_sessions_count_earlier_elo_sessions_per_lancelot_id(self):
        results = pl.DataFrame(
            {
                "tournament_id": ["1", "2", "2", "3", "9"],
                "date": ["2026-10-01", "2026-10-05", "2026-10-05", "2026-10-16", "2026-10-20"],
                "player1_lancelot_id": ["246273", "246273", "77119", "246273", "246273"],
                "player2_lancelot_id": ["77119", "500", "246273", "77119", "77119"],
            }
        )
        boards = pl.DataFrame(
            {
                "Player_ID_N": ["246273"],
                "Player_ID_S": ["77119"],
                "Player_ID_E": ["500"],
                "Player_ID_W": ["999"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "elo.results.parquet"
            results.write_parquet(path)
            with patch.object(predictions, "_elo_results_path", return_value=path):
                out = predictions._attach_prior_sessions(boards, "3")
        self.assertEqual(
            out.select(f"Prior_Sessions_{seat}" for seat in "NESW").row(0),
            (2, 1, 2, 0),
        )

    def test_pair_takes_the_smaller_prior_count(self):
        boards = pl.DataFrame(
            {
                "Pair_Number_NS": [1, 1],
                "Player_ID_N": ["246273"] * 2,
                "Player_ID_S": ["77119"] * 2,
                "Player_Name_N": ["Robert SALITA"] * 2,
                "Player_Name_S": ["Ramendra DAS"] * 2,
                "Pct_NS_Pred": [0.6, 0.5],
                "Prior_Sessions_N": [281, 281],
                "Prior_Sessions_S": [12, 12],
            }
        )
        pair = predictions._predicted_pair_scores(boards).row(0, named=True)
        self.assertEqual(pair["Min_Prior_Sessions"], 12)
        self.assertEqual(pair["Boards"], 2)


class RankingTests(unittest.TestCase):
    def test_club_ranks_then_global_ranks_and_actual_minus_pred_diffs(self):
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
        east = ranked.filter(pl.col("Global_Scratch_Rank") == 4).row(0, named=True)
        self.assertEqual(north["Club_Rank"], 1)
        self.assertEqual(north["Global_Scratch_Rank"], 1)
        # Predicted 50 sits behind the other club's 55, so the predicted global rank is 2.
        self.assertEqual(north["Global_Scratch_Rank_Pred"], 2)
        self.assertEqual(north["Rank_Diff"], -1)
        self.assertAlmostEqual(north["Pct_Diff"], 10.0)
        columns = list(ranked.columns)
        self.assertEqual(columns.index("Global_Scratch_Rank") + 1, columns.index("Global_Scratch_Rank_Pred"))
        self.assertEqual(columns.index("Global_Scratch_Rank_Pred") + 1, columns.index("Rank_Diff"))
        self.assertEqual(east["Club_Rank"], 2)
        self.assertAlmostEqual(east["Global_Scratch_Pct_Pred"], 50.0)

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
                "Pct_Diff": [0.1],
            }
        )
        rows = predictions.board_prediction_rows(boards)
        self.assertEqual(rows["Contract_Pred"][0], "4SN")
        self.assertEqual(rows["Declarer_Direction"][0], "N")
        self.assertAlmostEqual(rows["Pct_Diff"][0], 0.1)

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
        self.assertEqual(north["Global_Scratch_Rank"], 3)
        self.assertIsNone(north["Global_Scratch_Rank_Pred"])
        self.assertIsNone(north["Rank_Diff"])
        with (
            patch.object(predictions, "models_dir", return_value=Path("missing-models")),
            patch.object(predictions, "_fetch_ranking", return_value=ranking.to_dicts()),
            tempfile.TemporaryDirectory() as cache,
        ):
            payload = predictions.prediction_payload(
                boards,
                "1",
                Path(cache),
                view="rankings",
                limit=20,
            )
        self.assertFalse(payload["predictions_available"])
        self.assertGreaterEqual(payload["row_count"], 1)
        self.assertEqual(payload["rows"][0]["Global_Scratch_Rank"], 3)

    def test_a_ranking_cached_before_publication_is_downloaded_again(self):
        placeholder = [
            {"sessionScore": 0.0, "rank": 1, "orientation": "NS", "tableNumber": 1},
            {"sessionScore": 0.0, "rank": 1, "orientation": "EW", "tableNumber": 1},
        ]
        published = [
            {"sessionScore": 53.82, "rank": 61, "orientation": "NS", "tableNumber": 1},
            {"sessionScore": 46.18, "rank": 150, "orientation": "EW", "tableNumber": 1},
        ]
        with tempfile.TemporaryDirectory() as cache:
            path = Path(cache) / "rankings" / "288737.json"
            path.parent.mkdir()
            path.write_text(json.dumps(placeholder), encoding="utf-8")
            with patch.object(predictions, "_fetch_ranking", return_value=published) as fetch:
                ranking, warning = predictions.load_ranking("288737", Path(cache))
                again, _ = predictions.load_ranking("288737", Path(cache))
            self.assertIsNone(warning)
            self.assertEqual(ranking["sessionScore"].to_list(), [53.82, 46.18])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), published)
            self.assertEqual(again["rank"].to_list(), [61, 150])
            self.assertEqual(fetch.call_count, 1)

    def test_unpublished_results_warn_instead_of_ranking_everyone_first(self):
        boards = pl.DataFrame(
            {
                "Board": [1, 1],
                "section_name": ["A", "A"],
                "Pair_Number_NS": [1, 2],
                "Pair_Number_EW": [1, 2],
                "Pct_NS": [0.6, 0.4],
            }
        )
        placeholder = [
            {"sessionScore": 0.0, "rank": 1, "orientation": orientation, "tableNumber": table}
            for orientation in ("NS", "EW")
            for table in (1, 2)
        ]
        with (
            patch.object(predictions, "models_dir", return_value=Path("missing-models")),
            patch.object(predictions, "_fetch_ranking", return_value=placeholder),
            tempfile.TemporaryDirectory() as cache,
        ):
            _, info = predictions.apply_session_predictions(boards, "288737", Path(cache))
            self.assertFalse((Path(cache) / "rankings" / "288737.json").exists())
        self.assertIn("has not published results for session 288737", info["warning"])
        self.assertFalse(info["results_published"])
        ranks = info["ranks"]
        self.assertEqual(ranks.height, 4)
        self.assertEqual(ranks["Global_Scratch_Pct"].null_count(), 4)
        self.assertEqual(ranks["Club_Rank"].null_count(), 4)
        self.assertEqual(ranks["Global_Scratch_Rank"].null_count(), 4)

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
        self.assertEqual(ranked["Club"].unique().to_list(), ["A"])
        self.assertEqual(ranked.height, 3)
        self.assertEqual(attached.height, boards.height)
        first = attached.filter(pl.col("Player_ID_N") == 1).row(0, named=True)
        second = attached.filter(pl.col("Player_ID_N") == 3).row(0, named=True)
        self.assertEqual(first["Club_Rank_NS"], 1)
        self.assertEqual(second["Club_Rank_NS"], 2)
        self.assertEqual(first["Global_Scratch_Rank_NS"], 2)
        self.assertIsNotNone(first["Club_Rank_Pred_NS"])
        self.assertIsNotNone(first["Global_Scratch_Rank_Pred_NS"])
        self.assertIsNotNone(first["Rank_Diff_NS"])
        self.assertNotEqual(first["Global_Scratch_Rank_NS"], 1)

    def test_another_clubs_row_without_ids_gets_no_prediction_by_table_number(self):
        ranking = pl.DataFrame(
            {
                "sessionScore": [56.4, 53.24, 50.0],
                "rank": [244, 430, 500],
                "simultaneousId": ["home", "away", "home"],
                "orientation": ["EW", "EW", "NS"],
                "section": ["A", "A", "A"],
                "tableNumber": [4, 4, 4],
                "team_player1_id": [21, None, 1],
                "team_player2_id": [31, None, 11],
                "team_player1_firstName": ["Cat", "Away", "Nor"],
                "team_player1_lastName": ["E", "E", "N"],
                "team_player2_firstName": ["Fran", "Away", "Sou"],
                "team_player2_lastName": ["W", "W", "S"],
            }
        )
        boards = pl.DataFrame(
            {
                "Board": [1],
                "section_name": ["A"],
                "Pair_Number_NS": [4],
                "Pair_Number_EW": [4],
                "Player_ID_N": [1],
                "Player_ID_S": [11],
                "Player_ID_E": [21],
                "Player_ID_W": [31],
                "Player_Name_N": ["Nor N"],
                "Player_Name_S": ["Sou S"],
                "Player_Name_E": ["Cat E"],
                "Player_Name_W": ["Fran W"],
                "Pct_NS_Pred": [0.52],
            }
        )
        ranked = predictions.ranking_rows(boards, ranking)
        self.assertEqual(ranked["Club"].to_list(), ["home", "home"])
        home = ranked.filter(pl.col("Orientation") == "EW").row(0, named=True)
        self.assertEqual(home["Pair"], "A4 - Cat E - Fran W")
        self.assertAlmostEqual(home["Global_Scratch_Pct_Pred"], 48.0)
        # 48 sits behind the away pair's published 53.24 in the national field.
        self.assertEqual(home["Global_Scratch_Rank_Pred"], 3)

    def test_handicapped_session_compares_scratch_and_shows_handicap_beside_it(self):
        # 288737: sessionScore and rank include the IV bonus (totalBonus).
        ranking = pl.DataFrame(
            {
                "sessionScore": [64.32, 60.0, 58.0, 45.0],
                "totalBonus": [10.5, 0.0, 2.0, 3.0],
                "rank": [1, 2, 3, 4],
                "simultaneousId": ["home", "away", "home", "home"],
                "orientation": ["NS", "NS", "NS", "EW"],
                "section": ["A", "A", "A", "A"],
                "tableNumber": [1, 1, 2, 1],
                "team_player1_id": [1, 5, 3, 11],
                "team_player2_id": [2, 6, 4, 12],
            }
        )
        boards = pl.DataFrame(
            {
                "Board": [1, 1],
                "section_name": ["A", "A"],
                "Pair_Number_NS": [1, 2],
                "Pair_Number_EW": [1, 1],
                "Player_ID_N": [1, 3],
                "Player_ID_S": [2, 4],
                "Player_ID_E": [11, 11],
                "Player_ID_W": [12, 12],
                "Pct_NS_Pred": [0.54, 0.50],
            }
        )
        ranked = predictions.ranking_rows(boards, ranking)
        bob = ranked.filter((pl.col("Orientation") == "NS") & (pl.col("Table") == "1")).row(0, named=True)
        self.assertAlmostEqual(bob["Global_Scratch_Pct"], 53.82)
        self.assertEqual(bob["Global_Scratch_Rank"], 3)
        self.assertEqual(bob["Club_Rank"], 2)
        self.assertAlmostEqual(bob["Pct_Diff"], -0.18)
        self.assertAlmostEqual(bob["Handicap_Pct"], 64.32)
        self.assertEqual(bob["Handicap_Global_Rank"], 1)
        self.assertEqual(bob["Handicap_Club_Rank"], 1)
        second = ranked.filter((pl.col("Orientation") == "NS") & (pl.col("Table") == "2")).row(0, named=True)
        self.assertAlmostEqual(second["Global_Scratch_Pct"], 56.0)
        self.assertEqual(second["Club_Rank"], 1)
        self.assertEqual(second["Handicap_Club_Rank"], 2)
        columns = list(ranked.columns)
        self.assertEqual(columns.index("Pct_Diff") + 1, columns.index("Handicap_Pct"))

    def test_scratch_session_has_no_handicap_columns_filled(self):
        ranking = pl.DataFrame(
            {
                "sessionScore": [58.78, 41.22],
                "totalBonus": [0.0, 0.0],
                "rank": [148, 900],
                "simultaneousId": ["home", "home"],
                "orientation": ["EW", "NS"],
                "section": ["A", "A"],
                "tableNumber": [12, 12],
                "team_player1_id": [21, 1],
                "team_player2_id": [31, 11],
            }
        )
        boards = pl.DataFrame(
            {
                "Board": [1],
                "section_name": ["A"],
                "Pair_Number_NS": [12],
                "Pair_Number_EW": [12],
                "Player_ID_N": [1],
                "Player_ID_S": [11],
                "Player_ID_E": [21],
                "Player_ID_W": [31],
                "Pct_NS_Pred": [0.45],
            }
        )
        ranked = predictions.ranking_rows(boards, ranking)
        east = ranked.filter(pl.col("Orientation") == "EW").row(0, named=True)
        self.assertAlmostEqual(east["Global_Scratch_Pct"], 58.78)
        self.assertEqual(east["Global_Scratch_Rank"], 148)
        self.assertIsNone(east["Handicap_Pct"])
        self.assertIsNone(east["Handicap_Global_Rank"])
        self.assertIsNone(east["Handicap_Club_Rank"])

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
        self.assertEqual(east["Global_Scratch_Pct_Pred"].to_list(), [59.0, 59.0])
        self.assertEqual(north["Global_Scratch_Pct_Pred"].to_list(), [41.0, 41.0])
        self.assertEqual(east["Club_Rank_Pred"].to_list(), [1, 1])
        self.assertEqual(north["Club_Rank_Pred"].to_list(), [3, 3])

    def test_a_stronger_ns_line_is_not_centered_away(self):
        # Every NS pair is predicted above 50%. Centering would force the NS
        # and EW lines to the same average; the raw model output is kept.
        frame = pl.DataFrame({
            "Board": [1, 1, 2, 2],
            "Pct_NS": [0.70, 0.40, 0.55, 0.65],
            "Pct_NS_Pred": [0.62, 0.58, 1.07, 0.56],
        })
        assigned = predictions._assign_ns_prediction(frame)
        self.assertEqual(assigned["Pct_NS_Pred"].to_list(), [0.62, 0.58, 1.0, 0.56])
        diffs = [round(value, 6) for value in assigned["Pct_Diff"].to_list()]
        self.assertEqual(diffs, [0.08, -0.18, -0.45, 0.09])

    def test_elo_gap_uses_the_filled_pair_ratings(self):
        frame = predictions._add_elo_missing_flags(
            pl.DataFrame({"Elo_NS": [1600.0, None], "Elo_EW": [1500.0, 1450.0]})
        )
        aligned = predictions._align_features(
            frame,
            {name: "Float32" for name in ("Elo_NS", "Elo_EW", "Elo_Gap", "Elo_NS_Missing")},
            ["Elo_NS", "Elo_EW", "Elo_Gap", "Elo_NS_Missing"],
            {
                "Elo_NS": {"mean": 1550.0},
                "Elo_EW": {"mean": 1520.0},
                "Elo_Gap": {"mean": 30.0},
                "Elo_NS_Missing": {"mean": 0.05},
            },
        )
        self.assertEqual(aligned["Elo_Gap"].to_list(), [100.0, 100.0])
        self.assertEqual(aligned["Elo_NS_Missing"].to_list(), [0.0, 1.0])

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
        self.assertEqual(aligned.schema["HCP_N"], pl.Float32)
        self.assertEqual(aligned["Dealer"].to_list(), ["N", "E", "S", "W", "N"])

    def test_a_missing_model_input_is_an_error_not_a_null_column(self):
        with self.assertRaisesRegex(ValueError, "SL_N_Longest"):
            predictions._align_features(
                pl.DataFrame({"HCP_N": [10]}),
                {"HCP_N": "Float32", "SL_N_Longest": "Float32"},
                ["HCP_N", "SL_N_Longest"],
            )

    def test_norths_longest_suit_comes_from_the_deal_not_the_shape_string(self):
        session = pl.DataFrame(
            {
                "PBN": ["N:J2.KQ7.K87.KT864 AKQ74.AJ52.94.92 98.T6.Q532.AQJ73 T653.9843.AJT6.5"],
                "SL_N_ML_SJ": ["5-3-3-2"],
            }
        )
        model_input = predictions._add_model_inputs(session, {"SL_N_Longest"}, Path("."))
        self.assertEqual(model_input["SL_N_Longest"].to_list(), [5.0])
        self.assertEqual(model_input.schema["SL_N_Longest"], pl.Float32)

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
            # The before ratings, not scratch_pair_elo (an after-session mean).
            self.assertEqual(attached["Elo_NS"][0], 1450.0)
            self.assertEqual(attached["Elo_EW"][0], 1250.0)

    def test_attach_elo_features_matches_lancelot_ids_not_the_other_person_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "elo.results.parquet"
            pl.DataFrame(
                {
                    "tournament_id": ["394099"],
                    "player1_id": ["999"],
                    "player2_id": ["998"],
                    "player1_lancelot_id": ["246273"],
                    "player2_lancelot_id": ["77119"],
                    "player1_scratch_elo_before": [1800.0],
                    "player2_scratch_elo_before": [1600.0],
                    "scratch_pair_elo": [1720.0],
                }
            ).write_parquet(path)
            boards = pl.DataFrame(
                {
                    "Player_ID_N": ["246273"],
                    "Player_ID_S": ["77119"],
                    "Player_ID_E": ["136662"],
                    "Player_ID_W": ["1692"],
                }
            )
            with patch.object(predictions, "_elo_results_path", return_value=path):
                attached = predictions._attach_elo_features(boards, "394099")

            self.assertEqual(attached["Elo_N"][0], 1800.0)
            self.assertEqual(attached["Elo_S"][0], 1600.0)
            self.assertIsNone(attached["Elo_E"][0])
            self.assertEqual(attached["Elo_NS"][0], 1700.0)

    def test_model_inputs_look_up_history_by_classic_id_before_the_session(self):
        from mlBridge.ffbridge_prediction_features import build_history_tables

        earlier = pl.DataFrame(
            {
                "Date": [date(2026, 1, 5)] * 2 + [date(2026, 2, 1)],
                "session_id": ["1", "1", "2"],
                "Declarer_Direction": ["N", "E", "N"],
                "Declarer_Pct": [1.0, 0.25, 0.0],
                "Player_ID_N": ["10"] * 3,
                "Player_ID_S": ["20"] * 3,
                "Player_ID_E": ["30"] * 3,
                "Player_ID_W": ["40"] * 3,
            }
        )
        players, pairs = build_history_tables(earlier.lazy())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            players.write_parquet(root / predictions.PLAYER_HISTORY_NAME)
            pairs.write_parquet(root / predictions.PAIR_HISTORY_NAME)
            elo = root / "elo.results.parquet"
            pl.DataFrame(
                {
                    "player1_id": ["10", "30"],
                    "player2_id": ["20", "40"],
                    "player1_lancelot_id": ["1010", "3030"],
                    "player2_lancelot_id": ["2020", "4040"],
                }
            ).write_parquet(elo)
            # The session on 2026-02-01 must not see that day's board.
            session = pl.DataFrame(
                {
                    "Date": ["2026-02-01"],
                    "PBN": ["N:AKQJ.T98.765.43 T98.765.43.AKQJ 765.43.AKQJ.T98 432.AKQJ.T98.765"],
                    "Player_ID_N": ["1010"],
                    "Player_ID_S": ["2020"],
                    "Player_ID_E": ["3030"],
                    "Player_ID_W": ["4040"],
                    "SL_N_S": [4],
                }
            )
            with patch.object(predictions, "_elo_results_path", return_value=elo):
                model_input = predictions._add_model_inputs(
                    session, {"Hist_Pct_N", "SL_E_C", "Fit_NS_S"}, root
                )
        row = model_input.row(0, named=True)
        # NS scored 1.0 and 0.75 on 2026-01-05: (1.75 + 30) / (2 + 60).
        self.assertAlmostEqual(row["Hist_Pct_N"], 31.75 / 62.0, places=5)
        self.assertAlmostEqual(row["Hist_Decl_Pct_N"], 8.5 / 16.0, places=5)
        self.assertGreater(row["Hist_Partner_Sessions_NS"], 0.0)
        self.assertEqual(row["SL_E_C"], 4.0)
        self.assertEqual(row["Fit_NS_S"], 7.0)
        self.assertEqual(session.schema["SL_N_S"], pl.Int64)


if __name__ == "__main__":
    unittest.main()
