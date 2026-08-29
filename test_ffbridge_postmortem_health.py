import unittest
from unittest.mock import Mock, patch

import requests

import ffbridge_postmortem_api_client as api
import ffbridge_postmortem_api_server as api_server


class WriterHealthTests(unittest.TestCase):
    @patch.object(api.requests, "request")
    def test_ready_writer_returns_structured_health(self, request):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {
            "detail": "ready",
            "jobs_running": 1,
            "last_job_id": "job-123",
            "last_parquet_write_at": "2026-08-23T18:15:00+00:00",
        }
        request.return_value = response

        result = api.writer_health()

        self.assertTrue(result["ok"])
        self.assertTrue(result["sidecar_up"])
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["detail"], "ready")
        self.assertEqual(result["jobs_running"], 1)
        self.assertEqual(result["last_job_id"], "job-123")
        self.assertEqual(
            result["last_parquet_write_at"],
            "2026-08-23T18:15:00+00:00",
        )
        self.assertLess(result["latency_ms"], 2000)
        request.assert_called_once_with(
            "GET",
            f"{api.FFBRIDGE_POSTMORTEM_API_BASE_URL}/health",
            params={},
            timeout=1.5,
        )

    @patch.object(api.requests, "request")
    def test_connection_refused_is_sidecar_down(self, request):
        request.side_effect = requests.ConnectionError("connection refused")

        result = api.writer_health()

        self.assertFalse(result["ok"])
        self.assertFalse(result["sidecar_up"])
        self.assertIsNone(result["http_status"])
        self.assertEqual(result["detail"], "sidecar_down")

    @patch.object(api.requests, "request")
    def test_writer_5xx_is_sidecar_error(self, request):
        response = Mock(ok=False, status_code=503)
        response.json.return_value = {"detail": "not ready", "hint": "restart writer"}
        request.return_value = response

        result = api.writer_health()

        self.assertFalse(result["ok"])
        self.assertFalse(result["sidecar_up"])
        self.assertEqual(result["http_status"], 503)
        self.assertEqual(result["detail"], "sidecar_error")
        self.assertEqual(result["error"], "not ready")

    @patch.object(
        api_server.create,
        "generate_health",
        return_value={
            "jobs_running": 1,
            "last_job_id": "job-123",
            "last_error": None,
            "last_parquet_write_at": "2026-08-23T18:15:00+00:00",
        },
    )
    @patch.object(api_server.svc, "dataset_info")
    def test_api_health_has_job_diagnostics_without_lancelot_work(
        self,
        dataset_info,
        generate_health,
    ):
        result = api_server.health()

        self.assertTrue(result["ok"])
        self.assertTrue(result["sidecar_up"])
        self.assertEqual(result["detail"], "ready")
        self.assertEqual(result["jobs_running"], 1)
        self.assertEqual(result["last_job_id"], "job-123")
        self.assertIsNone(result["last_error"])
        generate_health.assert_called_once_with(api_server.svc.CACHE_DIR)
        dataset_info.assert_not_called()


class PlayedOnDateBoundaryTests(unittest.TestCase):
    @patch.object(api, "_get_json")
    def test_api_client_passes_required_player_and_clubs(self, get_json):
        get_json.return_value = {"found": True}

        result = api.last_game("Alice Smith", ["21333"])

        self.assertEqual(result, {"found": True})
        get_json.assert_called_once_with(
            "/player-games/last",
            {"player": "Alice Smith", "clubs": ["21333"]},
        )

    def test_api_registers_player_game_routes(self):
        paths = {route.path for route in api_server.app.routes}
        self.assertIn("/player-games/last", paths)
        self.assertIn("/player-games/played-today", paths)


if __name__ == "__main__":
    unittest.main()
