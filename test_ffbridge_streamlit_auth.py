import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

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

    def test_generation_error_is_saved_for_the_next_rerun(self):
        state = SessionState(
            game_urls_d={"9500754": {300749: {}}},
            debug_mode=False,
        )
        error = app.pm_api.FfbridgeApiClientError("writer generation failed")
        with (
            patch.object(app.st, "session_state", state),
            patch.object(app.st, "spinner", side_effect=lambda *_a, **_k: nullcontext()),
            patch.object(app.st, "error") as show_error,
            patch.object(app, "populate_game_urls_for_player", return_value=True),
            patch.object(app.pm_api, "generate_and_wait", side_effect=error),
        ):
            failed = app._change_game_state_lancelot("9500754", 300749)

        self.assertTrue(failed)
        self.assertEqual(state["report_error"], "writer generation failed")
        show_error.assert_called_once_with("writer generation failed")

    def test_deferred_report_failure_does_not_immediately_rerun(self):
        state = SessionState(
            player_id="9500754",
            deferred_start_report=True,
            game_urls_d={"9500754": {300749: {}}},
            report_error=None,
        )
        view = object.__new__(app.FFBridgeApp)
        view.create_sidebar = Mock()
        with (
            patch.object(app.st, "session_state", state),
            patch.object(app.st, "error"),
            patch.object(app.st, "rerun") as rerun,
            patch.object(app, "populate_game_urls_for_player", return_value=True),
            patch.object(app, "change_game_state", return_value=True),
        ):
            view.create_ui()

        rerun.assert_not_called()
        self.assertFalse(state["deferred_start_report"])

    def test_results_url_is_rebuilt_from_session_group_when_meta_lacks_it(self):
        state = SessionState(
            session_id=300749,
            org_id="5802079",
            team_id=15106224,
            game_url=None,
            cache_dir="cache",
            game_urls_d={},
        )
        with (
            patch.object(app.st, "session_state", state),
            patch.object(
                app.pm_create,
                "resolve_session_group_id",
                return_value="21333",
            ),
        ):
            url = app._ensure_game_results_url()
        self.assertEqual(
            url,
            "https://www.ffbridge.fr/competitions/results/groups/"
            "21333/sessions/300749/pairs/15106224",
        )
        self.assertEqual(state["game_url"], url)

    def test_broken_none_group_url_is_rejected(self):
        self.assertIsNone(
            app._usable_results_url(
                "https://www.ffbridge.fr/competitions/results/groups/"
                "None/sessions/300749/pairs/1"
            )
        )

    def test_latest_session_is_newest_date_not_first_list_item(self):
        june_first = {
            282792: {"date": "2026-06-01", "description": "2026-06-01 Ronde"},
            282839: {"date": "2026-08-25", "description": "2026-08-25 Rondes de France"},
            300751: {"date": "2026-08-24", "description": "2026-08-24 Octopus"},
        }
        ordered = app._games_newest_first(june_first)
        self.assertEqual(list(ordered), [282839, 300751, 282792])
        self.assertEqual(app._latest_session_id(june_first), 282839)

    def test_omitted_session_loads_august_not_june(self):
        june_first = {
            282792: {"date": "2026-06-01", "description": "2026-06-01 Ronde"},
            282839: {"date": "2026-08-25", "description": "2026-08-25 Rondes de France"},
        }
        state = SessionState(
            game_urls_d={"9500754": june_first},
            debug_mode=False,
        )
        with (
            patch.object(app.st, "session_state", state),
            patch.object(app.st, "spinner", side_effect=lambda *_a, **_k: nullcontext()),
            patch.object(app, "populate_game_urls_for_player", return_value=True),
            patch.object(
                app.pm_api,
                "generate_and_wait",
                return_value={
                    "status": "ok",
                    "session_id": "282839",
                    "results": [{
                        "session_id": "282839",
                        "meta": {
                            "session_id": 282839,
                            "tournament_date": "2026-06-01",
                        },
                    }],
                },
            ) as generate,
            patch.object(app.pm_api, "postmortem_dataframe", return_value=pl.DataFrame()),
            patch.object(app, "filter_dataframe", return_value=pl.DataFrame()),
            patch.object(app, "_ensure_game_results_url"),
            patch.object(app, "get_session_duckdb_connection", return_value=Mock()),
        ):
            failed = app._change_game_state_lancelot("9500754", None)

        self.assertFalse(failed)
        generate.assert_called_once()
        self.assertEqual(generate.call_args.args[0], "9500754")
        self.assertEqual(generate.call_args.kwargs["session_id"], "282839")
        self.assertEqual(state["session_id"], 282839)
        self.assertEqual(state["tournament_date"], "2026-08-25")

    def test_go_clears_a_bookmarked_june_session(self):
        state = SessionState(
            session_id=282792,
            club_session_ids_selectbox="282792, 2026-06-01 Ronde",
            _url_loaded_session_key=("9500754", 282792),
        )
        with patch.object(app.st, "session_state", state):
            app._clear_selected_session()
        self.assertIsNone(state["session_id"])
        self.assertNotIn("club_session_ids_selectbox", state)
        self.assertNotIn("_url_loaded_session_key", state)


if __name__ == "__main__":
    unittest.main()
