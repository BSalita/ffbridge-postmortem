import unittest
from unittest.mock import patch

import ffbridge_player_game_service as games
import ffbridge_postmortem_create as create


def _ranking_row(team_id: int, score: float) -> dict:
    return {
        "sessionScore": score,
        "simultaneousId": 5802079,
        "team": {"id": team_id},
    }


class PlayerGameSummaryTests(unittest.TestCase):
    def test_formats_requested_rich_summary(self):
        candidate = {
            "session_id": "300900",
            "date": "2026-08-29",
            "raw_date": "2026-08-29T14:00:00+02:00",
            "session_label": "Atout Simultané",
            "moment": "A",
            "group_id": "21333",
            "club_code": "5802079",
            "club_name": "Bridge Club Levallois Perret",
            "scope": "specified_club",
        }
        ranking = {
            "rank": 16,
            "theoreticalRank": 90,
            "sessionScore": 57.8,
            "simultaneousId": 5802079,
            "orientation": "EW",
            "section": "A",
            "tableNumber": 8,
            "team": {
                "id": 99,
                "player1": {
                    "id": 101,
                    "migrationId": 201,
                    "ffbId": 301,
                    "firstName": "Christian",
                    "lastName": "JACOUPY",
                },
                "player2": {
                    "id": 246273,
                    "migrationId": 597539,
                    "ffbId": 9500754,
                    "firstName": "Robert",
                    "lastName": "SALITA",
                },
            },
        }
        full_ranking = [
            _ranking_row(1, 65),
            _ranking_row(2, 62),
            _ranking_row(3, 60),
            _ranking_row(99, 57.8),
            *[_ranking_row(team_id, 57 - team_id / 100) for team_id in range(4, 20)],
        ]
        resolved = create.ResolvedPlayer(
            "246273",
            "9500754",
            "Robert Salita",
            "597539",
        )

        with patch.object(
            games.create.mlBridgeFFLib,
            "get_session_ranking",
            return_value=full_ranking,
        ), patch.object(
            games,
            "_session_club_context",
            return_value={},
        ):
            result = games._game(candidate, ranking, resolved)

        self.assertEqual(result["local_rank"], 4)
        self.assertEqual(result["team_count"], 20)
        self.assertEqual(result["player_seat"], "West")
        self.assertEqual(result["partner_seat"], "East")
        self.assertEqual(
            result["summary"],
            "Atout Simultané — 29/08/2026 — Après-midi · 20 équipes · "
            "SALITA Robert finished 4th of 20 (G. 16, TH. 90) · "
            "Série A, table 8 · SALITA Robert West, JACOUPY Christian East · "
            "57,80 %",
        )

    def test_player_is_mandatory(self):
        with self.assertRaisesRegex(ValueError, "player is required"):
            games.last_game("")

    def test_played_today_returns_boolean_and_rich_game(self):
        auth = create.LancelotAuth("token", "246273", "9500754", "597539")
        resolved = create.ResolvedPlayer(
            "246273",
            "9500754",
            "Robert Salita",
            "597539",
        )
        candidate = {"session_id": "300900", "date": "2026-08-29"}
        game = {"summary": "summary", "player_name": "SALITA Robert"}
        with (
            patch.object(games.create, "ensure_lancelot_auth", return_value=auth),
            patch.object(
                games.create,
                "resolve_player_query",
                return_value=(resolved, "Robert Salita"),
            ),
            patch.object(games, "_candidate_sessions", return_value=[candidate]),
            patch.object(games, "_personal_ranking", return_value={"rank": 1}),
            patch.object(games, "_game", return_value=game),
        ):
            result = games.played_today("Robert Salita")

        self.assertTrue(result["played"])
        self.assertEqual(result["summary"], "summary")
        self.assertEqual(result["games"], [game])


if __name__ == "__main__":
    unittest.main()
