"""HTTP client for the FFBridge postmortem API.

MCP and Streamlit call this instead of importing the generate/read library.
Configure with FFBRIDGE_POSTMORTEM_API_BASE_URL (default http://127.0.0.1:8517).
"""

from __future__ import annotations

import io
import os
import time
from typing import Any, Dict, Optional

import polars as pl
import requests

FFBRIDGE_POSTMORTEM_API_BASE_URL = os.environ.get(
    "FFBRIDGE_POSTMORTEM_API_BASE_URL", "http://127.0.0.1:8517"
).rstrip("/")
_TIMEOUT_S = 300
_HEALTH_TIMEOUT_S = 1.5
_JOB_POLL_S = 2.0


class FfbridgeApiClientError(RuntimeError):
    def __init__(
        self,
        detail: str,
        hint: Optional[str] = None,
        status_code: Optional[int] = None,
        reason: str = "api_error",
    ):
        message = detail if not hint else f"{detail} ({hint})"
        super().__init__(message)
        self.detail = detail
        self.hint = hint
        self.status_code = status_code
        self.reason = reason


def _request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_s: float = _TIMEOUT_S,
) -> requests.Response:
    url = f"{FFBRIDGE_POSTMORTEM_API_BASE_URL}{path}"
    try:
        resp = requests.request(
            method,
            url,
            params={k: v for k, v in (params or {}).items() if v is not None},
            timeout=timeout_s,
        )
    except requests.Timeout as exc:
        raise FfbridgeApiClientError(
            f"FFBridge writer timed out at {FFBRIDGE_POSTMORTEM_API_BASE_URL}",
            hint="Check the writer process on port 8517.",
            reason="timeout",
        ) from exc
    except requests.RequestException as exc:
        raise FfbridgeApiClientError(
            f"FFBridge writer unreachable at {FFBRIDGE_POSTMORTEM_API_BASE_URL}: {exc}",
            hint="Start or restart the supervised writer process on port 8517.",
            reason="sidecar_down",
        ) from exc
    if not resp.ok:
        try:
            body = resp.json()
        except ValueError:
            body = {}
        reason = "sidecar_error" if resp.status_code >= 500 else "api_error"
        raise FfbridgeApiClientError(
            body.get("detail") or f"{resp.status_code} from {url}",
            hint=body.get("hint"),
            status_code=resp.status_code,
            reason=reason,
        )
    return resp


def _get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    return _request("GET", path, params=params).json()


def _post_json(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    return _request("POST", path, params=params).json()


def dataset_info() -> Dict[str, Any]:
    return _get_json("/info")


def writer_health(timeout_s: float = _HEALTH_TIMEOUT_S) -> Dict[str, Any]:
    """Probe the writer without authentication, Lancelot calls, or generation."""
    started = time.perf_counter()
    try:
        response = _request("GET", "/health", timeout_s=timeout_s)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        return {
            "ok": True,
            "sidecar_up": True,
            "http_status": response.status_code,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "detail": payload.get("detail", "ready"),
        }
    except FfbridgeApiClientError as exc:
        return {
            "ok": False,
            "sidecar_up": False,
            "http_status": exc.status_code,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "detail": exc.reason,
            "error": exc.detail,
            "hint": exc.hint,
        }


def resolve_player(player_id: str) -> Dict[str, Any]:
    return _get_json("/players/resolve", {"player_id": player_id})


def cached_sessions(player_id: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    return _get_json("/sessions", {"player_id": player_id, "limit": limit})


def list_source_sessions(
    player_id: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    return _get_json(
        "/source-sessions",
        {"player_id": player_id, "date_from": date_from, "date_to": date_to},
    )


def generate(
    player_id: str,
    session_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    force: bool = False,
    continue_on_error: Optional[bool] = None,
) -> Dict[str, Any]:
    return _post_json(
        "/generate",
        {
            "player_id": player_id,
            "session_id": session_id,
            "date_from": date_from,
            "date_to": date_to,
            "force": force,
            "continue_on_error": continue_on_error,
        },
    )


def generate_status(job_id: str) -> Dict[str, Any]:
    return _get_json(f"/generate/{job_id}")


def wait_for_generate(job_id: str, poll_s: float = _JOB_POLL_S) -> Dict[str, Any]:
    """Block until a generate job is ok or error."""
    while True:
        payload = generate_status(job_id)
        status = payload.get("status")
        if status in ("ok", "error", "cached", "completed"):
            return payload
        time.sleep(poll_s)


def generate_and_wait(
    player_id: str,
    session_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    force: bool = False,
    continue_on_error: Optional[bool] = None,
) -> Dict[str, Any]:
    payload = generate(
        player_id,
        session_id=session_id,
        date_from=date_from,
        date_to=date_to,
        force=force,
        continue_on_error=continue_on_error,
    )
    job_id = payload.get("job_id")
    if payload.get("status") == "started" and job_id:
        return wait_for_generate(job_id)
    return payload


def postmortem_meta(player_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    return _get_json(f"/postmortems/{player_id}/meta", {"session_id": session_id})


def postmortem_boards(
    player_id: str,
    session_id: Optional[str] = None,
    only_my_boards: bool = True,
    columns: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    return _get_json(
        f"/postmortems/{player_id}/boards",
        {
            "session_id": session_id,
            "only_my_boards": only_my_boards,
            "columns": columns,
            "limit": limit,
        },
    )


def postmortem_sql(
    player_id: str,
    sql: str,
    session_id: Optional[str] = None,
    limit: int = 500,
) -> Dict[str, Any]:
    return _get_json(
        f"/postmortems/{player_id}/sql",
        {"sql": sql, "session_id": session_id, "limit": limit},
    )


def postmortem_schema(
    player_id: str,
    session_id: Optional[str] = None,
    pattern: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    return _get_json(
        f"/postmortems/{player_id}/schema",
        {"session_id": session_id, "pattern": pattern, "limit": limit},
    )


def postmortem_dataframe(player_id: str, session_id: Optional[str] = None) -> pl.DataFrame:
    resp = _request("GET", f"/postmortems/{player_id}/parquet", {"session_id": session_id})
    return pl.read_parquet(io.BytesIO(resp.content))
