import unittest
from unittest.mock import patch

import polars as pl

import ffbridge_streamlit as app


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class StreamlitLancelotRoutingTests(unittest.TestCase):
    def test_default_source_is_lancelot_without_health_based_fallback(self):
        with patch.object(
            app,
            "probe_api_sources",
            side_effect=AssertionError("default selection must not probe"),
        ):
            self.assertEqual(app.auto_detect_api_source(), app.API_SOURCE_LANCELOT)

    def test_startup_loads_tokens_without_authenticating(self):
        state = SessionState()
        with (
            patch.object(app.st, "session_state", state),
            patch.object(app, "load_dotenv"),
            patch.dict(
                app.os.environ,
                {
                    "FFBRIDGE_BEARER_TOKEN_LANCELOT": "configured-token",
                    "FFBRIDGE_EASI_TOKEN": "configured-easi",
                },
            ),
            patch.object(
                app.pm_create,
                "ensure_lancelot_auth",
                side_effect=AssertionError("startup must not authenticate"),
            ),
        ):
            app.initialize_ffbridge_bearer_token()

        self.assertEqual(state["ffbridge_bearer_token"], "configured-token")
        self.assertEqual(state["ffbridge_easi_token"], "configured-easi")
        self.assertFalse(state["lancelot_token_valid"])

    def test_numeric_license_uses_index_before_live_person_search(self):
        state = SessionState(
            api_source=app.API_SOURCE_LANCELOT,
            logged_in_lancelot_id=None,
            logged_in_license_number=None,
            logged_in_player_id=None,
        )
        indexed = pl.DataFrame(
            [
                {
                    "person_id": "246273",
                    "person_firstname": "",
                    "person_lastname": "",
                    "person_license_number": "9500754",
                    "person_migration_id": 597539,
                }
            ],
            schema=app._LANCELOT_SEARCH_SCHEMA,
        )
        with (
            patch.object(app.st, "session_state", state),
            patch.object(
                app,
                "_license_lookup_from_index_or_api",
                return_value=indexed,
            ),
            patch.object(
                app,
                "_search_persons_lancelot",
                side_effect=AssertionError("known license must not call Lancelot"),
            ),
        ):
            result = app.search_members("9500754")

        self.assertEqual(result["person_id"].to_list(), ["246273"])

    def test_indexed_game_list_does_not_require_login(self):
        state = SessionState(
            lancelot_token_valid=False,
            game_urls_d={},
        )
        listed = {
            "player_id": "246273",
            "player_license_number": "9500754",
            "classic_person_id": "597539",
            "sessions": [
                {
                    "session_id": "300749",
                    "description": "2026-08-17 Octopus",
                    "date": "2026-08-17",
                    "listing_source": "shared Lancelot player-session index",
                }
            ],
        }
        with (
            patch.object(app.st, "session_state", state),
            patch.object(app.pm_api, "list_source_sessions", return_value=listed),
        ):
            result = app._populate_game_urls_for_player_lancelot("9500754")

        self.assertTrue(result)
        self.assertIn(300749, state["game_urls_d"]["9500754"])
        self.assertNotIn("player_search_error", state)


if __name__ == "__main__":
    unittest.main()
