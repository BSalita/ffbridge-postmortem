import unittest
from unittest.mock import patch

import ffbridge_entry_list as entry_list


def _person(pid, last, first, licence, iv=50):
    return {"id": pid, "ffbId": licence, "lastName": last, "firstName": first, "season": {"ranking": {"iv": iv}}}


SESSION = {
    "date": "2026-10-23T14:00:00+02:00",
    "expectedBoardCount": 24,
    "session": {"label": "Atout Simultane"},
    "group": {"phase": {"stade": {"organization": {"ffbCode": 5802079, "label": "Bridge Club Levallois Perret"}}}},
}


def _team(team_id, players, iv):
    return {"id": team_id, "iv": iv, "players": players, "currentTeamEntry": {"createdAt": "2026-10-21T17:54:30+02:00"}}


class _FakeAuth:
    token = "t"
    lancelot_id = "246273"


class EntryListTests(unittest.TestCase):
    def test_entrants_page_until_a_short_page(self):
        pages = {
            1: {"items": [_team(i, [_person(f"a{i}", "A", "x", f"L{i}"), _person(f"b{i}", "B", "y", f"M{i}")], 90) for i in range(entry_list.PAGE_SIZE)]},
            2: {"items": [_team(999, [_person("c", "C", "z", "N"), _person("d", "D", "w", "O")], 80)]},
        }
        calls = []

        def fake(path, params=None):
            calls.append(params["currentPage"])
            return pages[params["currentPage"]]

        with patch.object(entry_list, "_api_get", side_effect=fake):
            pairs = entry_list.session_entrants("422517")
        self.assertEqual(calls, [1, 2])
        self.assertEqual(len(pairs), entry_list.PAGE_SIZE + 1)
        self.assertEqual(pairs[-1]["players"][0]["lancelot_id"], "c")

    def test_session_meta_reads_the_organizing_club(self):
        with patch.object(entry_list, "_api_get", return_value=SESSION):
            meta = entry_list.session_meta("422517")
        self.assertEqual(meta["club_code"], "5802079")
        self.assertEqual(meta["label"], "Atout Simultane")
        self.assertEqual(meta["expected_boards"], 24)

    def test_forecast_group_session_passes_lancelot_ids_and_the_session_day(self):
        teams = {"items": [
            _team(1, [_person("77119", "DAS", "Ramendra", "2207977"), _person("246273", "SALITA", "Robert", "9500754")], 106),
            _team(2, [_person("14948", "BRUN", "Bernard", "663642"), _person("16915", "MAMOU MANI", "Claude", "757594")], 160),
        ]}

        def fake(path, params=None):
            return SESSION if path.startswith("/competitions/groupSessions/") else teams

        captured = {}

        def fake_predict(entries, on_date, sims):
            captured.update(entries=entries, on_date=on_date, sims=sims)
            return {"rows": [], "warnings": []}

        with patch.object(entry_list, "_api_get", side_effect=fake), \
                patch.object(entry_list.entries_forecast, "forecast_entries", side_effect=fake_predict):
            out = entry_list.forecast_group_session("422517", sims=500)
        self.assertEqual(captured["on_date"], "2026-10-23")
        self.assertEqual(captured["sims"], 500)
        self.assertEqual(captured["entries"][0], {"players": ["77119", "246273"], "label": "DAS Ramendra - SALITA Robert"})
        self.assertEqual(out["session"]["group_session_id"], "422517")
        self.assertEqual(out["team_iv"]["DAS Ramendra - SALITA Robert"], 106)

    def test_without_a_session_uses_the_next_entry(self):
        upcoming = {"items": [
            {"players": [_person("246273", "SALITA", "Robert", "9500754"), _person("33351", "JACOUPY", "Christian", "1")],
             "teamEntries": [{"groupSession": {"id": 422522}}]},
            {"players": [_person("246273", "SALITA", "Robert", "9500754"), _person("77119", "DAS", "Ramendra", "2")],
             "teamEntries": [{"groupSession": {"id": 422517}}]},
        ]}
        later = dict(SESSION, date="2026-10-27T14:00:00+02:00")

        def fake(path, params=None):
            if path.startswith("/persons/"):
                return upcoming
            if path == "/competitions/groupSessions/422522":
                return later
            return SESSION

        with patch.object(entry_list, "_api_get", side_effect=fake), \
                patch.object(entry_list.create, "ensure_lancelot_auth", return_value=_FakeAuth()):
            rows = entry_list.upcoming_entries("2026-10-23")
        self.assertEqual([row["group_session_id"] for row in rows], ["422517", "422522"])
        self.assertEqual(rows[0]["partner"], "DAS Ramendra")

    def test_403_means_private_entries(self):
        import requests

        response = requests.Response()
        response.status_code = 403
        error = requests.HTTPError(response=response)
        with patch.object(entry_list, "_api_get", side_effect=error), \
                patch.object(entry_list.create, "ensure_lancelot_auth", return_value=_FakeAuth()):
            with self.assertRaises(entry_list.EntriesPrivate):
                entry_list.upcoming_entries("2026-10-23", person_id="77119")

    def test_another_players_entries_name_their_partner(self):
        upcoming = {"items": [
            {"players": [_person("77119", "DAS", "Ramendra", "2"), _person("246273", "SALITA", "Robert", "9500754")],
             "teamEntries": [{"groupSession": {"id": 422517}}]},
        ]}

        def fake(path, params=None):
            self.assertTrue(path.startswith("/persons/77119/") or path.startswith("/competitions/"))
            return upcoming if path.startswith("/persons/") else SESSION

        with patch.object(entry_list, "_api_get", side_effect=fake), \
                patch.object(entry_list.create, "ensure_lancelot_auth", return_value=_FakeAuth()):
            rows = entry_list.upcoming_entries("2026-10-23", person_id="77119")
        self.assertEqual(rows[0]["partner"], "SALITA Robert")

    def test_entries_with_forecast_pick_the_players_own_row(self):
        rows = [{"group_session_id": "422517", "date": "2026-10-23T14:00:00+02:00"},
                {"group_session_id": "422999", "date": "2026-10-24T14:00:00+02:00"}]
        forecast = {
            "pairs": 2,
            "rows": [{"Pair": "BRUN Bernard - MAMOU MANI Claude", "Rank_Forecast": 1, "Pct_Forecast": 61.0, "Orientation": None},
                     {"Pair": "DAS Ramendra - SALITA Robert", "Rank_Forecast": 2, "Pct_Forecast": 55.5, "Orientation": None}],
            "pair_of_player": {"246273": "DAS Ramendra - SALITA Robert"},
            "entry_list_read_at": "2026-10-23T10:00:00+00:00",
        }

        def fake_predict(gsid, sims):
            if gsid == "422999":
                raise ValueError("Group session 422999 has 1 registered pairs")
            return forecast

        with patch.object(entry_list, "upcoming_entries", return_value=rows), \
                patch.object(entry_list, "forecast_group_session", side_effect=fake_predict):
            out = entry_list.entries_with_forecast("246273", "2026-10-23", sims=500)
        self.assertEqual((out[0]["forecast_rank"], out[0]["forecast_pct"], out[0]["pairs"]), (2, 55.5, 2))
        self.assertEqual(out[0]["pair"], "DAS Ramendra - SALITA Robert")
        self.assertIn("1 registered pairs", out[1]["forecast_error"])

    def test_fewer_than_two_pairs_fails(self):
        def fake(path, params=None):
            return SESSION if path.startswith("/competitions/groupSessions/") else {"items": []}

        with patch.object(entry_list, "_api_get", side_effect=fake):
            with self.assertRaises(ValueError):
                entry_list.forecast_group_session("422517")


if __name__ == "__main__":
    unittest.main()
