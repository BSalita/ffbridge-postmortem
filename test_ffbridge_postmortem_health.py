import unittest
from unittest.mock import Mock, patch

import requests

import ffbridge_postmortem_api_client as api
import ffbridge_postmortem_api_server as api_server
import ffbridge_postmortem_mcp_server as mcp_server


class WriterHealthTests(unittest.TestCase):
    @patch.object(api.requests, "request")
    def test_ready_writer_returns_structured_health(self, request):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"detail": "ready"}
        request.return_value = response

        result = api.writer_health()

        self.assertTrue(result["ok"])
        self.assertTrue(result["sidecar_up"])
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["detail"], "ready")
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

    def test_writer_tool_maps_sidecar_failure(self):
        failure = api.FfbridgeApiClientError(
            "connection refused",
            hint="restart writer",
            reason="sidecar_down",
        )

        result = mcp_server._writer_tool(Mock(side_effect=failure))

        self.assertEqual(result["error"], "writer_unavailable")
        self.assertEqual(result["reason"], "sidecar_down")
        self.assertEqual(result["detail"], "connection refused")

    @patch.object(api_server.svc, "dataset_info")
    def test_api_health_has_no_cache_or_lancelot_work(self, dataset_info):
        result = api_server.health()

        self.assertTrue(result["ok"])
        self.assertTrue(result["sidecar_up"])
        self.assertEqual(result["detail"], "ready")
        dataset_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
