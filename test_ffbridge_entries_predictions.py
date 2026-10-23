import unittest

import numpy as np
import polars as pl

import ffbridge_entries_predictions as ep


def _row(tid, day, p1, p2, e1, e2, *, lic1=None, lic2=None, lan1=None, lan2=None, n1=None, n2=None):
    return {
        "tournament_id": tid,
        "date": f"{day}T00:00:00+02:00",
        "player1_id": p1,
        "player2_id": p2,
        "player1_lancelot_id": lan1 or f"9{p1}",
        "player2_lancelot_id": lan2 or f"9{p2}",
        "player1_license_number": lic1 or f"0{p1}",
        "player2_license_number": lic2 or f"0{p2}",
        "player1_name": n1 or f"Name{p1} First",
        "player2_name": n2 or f"Name{p2} First",
        "player1_scratch_elo_after": e1,
        "player2_scratch_elo_after": e2,
    }


def _results():
    rows = [
        # Strong pair 11/12, together twice.
        _row("t1", "2026-09-01", "11", "12", 1900.0, 1880.0, n1="SALITA Robert", lic1="09500754"),
        _row("t2", "2026-10-01", "11", "12", 2000.0, 1980.0, n1="SALITA Robert", lic1="09500754"),
        # Average pair 21/22.
        _row("t1", "2026-09-01", "21", "22", 1500.0, 1500.0, n1="Jacoupy Christian", lic1="01353276"),
        # Weak pair 31/32.
        _row("t1", "2026-09-01", "31", "32", 1100.0, 1150.0),
        # Session on the prediction day must be ignored.
        _row("t9", "2026-10-23", "11", "12", 100.0, 100.0, n1="SALITA Robert", lic1="09500754"),
        # Two different players called Dupont Jean; 42 is more recent.
        _row("t1", "2026-09-01", "41", "43", 1400.0, 1400.0, n1="Dupont Jean"),
        _row("t2", "2026-10-01", "42", "43", 1600.0, 1400.0, n1="DUPONT Jean"),
    ]
    return pl.DataFrame(rows).lazy()


class NameTests(unittest.TestCase):
    def test_name_key_ignores_order_case_and_accents(self):
        self.assertEqual(ep.normalize_name("SALITA Robert"), ep.normalize_name("robert salita"))
        self.assertEqual(ep.normalize_name("Hélène Brâmi"), ep.normalize_name("BRAMI Helene"))


class PredictEntriesTests(unittest.TestCase):
    def predict(self, entries, **kwargs):
        return ep.predict_entries(entries, "2026-10-23", results=_results(), sims=4000, **kwargs)

    def test_stronger_pair_is_predicted_first_and_field_averages_fifty(self):
        out = self.predict([["0011", "0012"], ["0021", "0022"], ["0031", "0032"]])
        rows = out["rows"]
        self.assertEqual([row["Rank_Pred"] for row in rows], [1, 2, 3])
        self.assertTrue(rows[0]["Pair"].startswith("SALITA Robert"))
        self.assertAlmostEqual(sum(row["Pct_Pred"] for row in rows) / 3, 50.0, places=1)
        self.assertGreater(rows[0]["P_Win"], rows[1]["P_Win"])
        self.assertAlmostEqual(sum(row["P_Win"] for row in rows), 100.0, delta=0.5)
        for row in rows:
            self.assertLessEqual(row["Rank_Pred_Low"], row["Rank_Pred_High"])
            self.assertAlmostEqual(
                row["Pct_Pred_High"] - row["Pct_Pred"], ep.Z95 * ep.PCT_RESIDUAL_SD, places=1
            )

    def test_uses_latest_rating_before_the_day_not_the_same_day(self):
        out = self.predict([["0011", "0012"], ["0021", "0022"]])
        top = out["rows"][0]
        self.assertEqual(top["Player1_Elo"], 2000.0)
        self.assertEqual(top["Pair_Elo"], 1990.0)
        self.assertEqual(top["Pair_Prior_Sessions"], 2)
        self.assertEqual(top["Ratings_Age_Days"], 22)

    def test_slope_matches_calibration(self):
        out = self.predict([["0011", "0012"], ["0031", "0032"]])
        gap = out["rows"][0]["Pair_Elo"] - out["rows"][1]["Pair_Elo"]
        pct_gap = out["rows"][0]["Pct_Pred"] - out["rows"][1]["Pct_Pred"]
        self.assertAlmostEqual(pct_gap, ep.PCT_PER_ELO * gap, places=1)

    def test_players_by_licence_lancelot_id_person_id_and_name(self):
        out = self.predict([
            "Robert Salita / 912",
            ["01353276", "22"],
        ])
        names = {row["Player1"] for row in out["rows"]}
        self.assertEqual(names, {"SALITA Robert", "Jacoupy Christian"})
        self.assertEqual(out["warnings"], [])

    def test_unrated_player_gets_field_median_and_low_confidence(self):
        out = self.predict([["0011", "0012"], ["0021", "Nobody Known"]])
        weak = next(row for row in out["rows"] if row["Player1"] == "Jacoupy Christian")
        self.assertTrue(weak["Low_Confidence"])
        self.assertEqual(weak["Unrated_Players"], ["Nobody Known"])
        self.assertTrue(any("Nobody Known" in note for note in out["warnings"]))

    def test_ambiguous_name_takes_most_recent_and_warns(self):
        out = self.predict([["Jean Dupont", "0021"], ["0031", "0032"]])
        row = next(row for row in out["rows"] if row["Player2"] == "Jacoupy Christian" or row["Player1"] == "DUPONT Jean")
        self.assertEqual(row["Player1_Elo"], 1600.0)
        self.assertTrue(any("ambiguous" in note for note in out["warnings"]))

    def test_orientation_ranks_within_each_direction(self):
        out = self.predict([
            {"players": ["0011", "0012"], "orientation": "NS"},
            {"players": ["0031", "0032"], "orientation": "NS"},
            {"players": ["0021", "0022"], "orientation": "EW"},
        ])
        ew = [row for row in out["rows"] if row["Orientation"] == "EW"]
        self.assertEqual(ew[0]["Rank_Pred"], 1)
        self.assertEqual(ew[0]["P_Win"], 100.0)

    def test_mixed_orientation_is_an_error(self):
        with self.assertRaises(ValueError):
            self.predict([{"players": ["0011", "0012"], "orientation": "NS"}, ["0021", "0022"]])

    def test_same_player_in_two_pairs_is_an_error(self):
        with self.assertRaises(ValueError):
            self.predict([["0011", "0012"], ["0011", "0021"]])

    def test_entry_needs_two_players(self):
        with self.assertRaises(ValueError):
            self.predict([["0011"]])


class SimulationTests(unittest.TestCase):
    def test_equal_pairs_split_wins_evenly(self):
        sim = ep.simulate_ranks(np.array([50.0, 50.0, 50.0, 50.0]), 6.0, 20000, seed=1)
        for p in sim["p_win"]:
            self.assertAlmostEqual(p, 0.25, delta=0.02)
        for value in sim["expected"]:
            self.assertAlmostEqual(value, 2.5, delta=0.05)


class ApiRouteTests(unittest.TestCase):
    def test_post_entries_returns_ranked_rows(self):
        import os
        import tempfile

        try:
            from fastapi.testclient import TestClient
            import ffbridge_postmortem_api_server as server
        except ImportError as exc:  # the server needs mlBridge beside it
            self.skipTest(f"API server not importable here: {exc}")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x_Lancelot_API.results.parquet")
            _results().collect().write_parquet(path)
            old = os.environ.get("FFBRIDGE_ELO_RESULTS")
            os.environ["FFBRIDGE_ELO_RESULTS"] = path
            try:
                client = TestClient(server.app)
                ok = client.post(
                    "/predictions/entries",
                    json={"entries": [["0011", "0012"], "0021 / 0022"], "date": "2026-10-23", "sims": 1000},
                )
                bad = client.post("/predictions/entries", json={"entries": [["0011"]]})
            finally:
                if old is None:
                    os.environ.pop("FFBRIDGE_ELO_RESULTS", None)
                else:
                    os.environ["FFBRIDGE_ELO_RESULTS"] = old
        self.assertEqual(ok.status_code, 200, ok.text)
        body = ok.json()
        self.assertEqual(body["pairs"], 2)
        self.assertEqual(body["rows"][0]["Rank_Pred"], 1)
        self.assertEqual(body["provenance"]["sims"], 1000)
        self.assertEqual(bad.status_code, 400)


if __name__ == "__main__":
    unittest.main()
