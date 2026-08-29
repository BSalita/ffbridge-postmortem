"""MCP server that proxies the FFBridge postmortem FastAPI.

Transport: streamable HTTP (endpoint /mcp) on FFBRIDGE_POSTMORTEM_MCP_PORT
(default 8512), stateless with JSON responses. This process does not call
Lancelot or run augmentation; it GET/POST-s FFBRIDGE_POSTMORTEM_API_BASE_URL
(default http://127.0.0.1:8517). Same pattern as acbl_club_mcp_server.py.

  python ffbridge_postmortem_api_server.py
  python ffbridge_postmortem_mcp_server.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

import ffbridge_postmortem_api_client as api

FFBRIDGE_POSTMORTEM_MCP_PORT = int(os.environ.get("FFBRIDGE_POSTMORTEM_MCP_PORT", "8512"))

mcp = MCPServer("ffbridge-postmortem")


def _client_error_payload(exc: api.FfbridgeApiClientError) -> Dict[str, Any]:
    return {
        "error": exc.detail,
        "hint": exc.hint,
        "status_code": exc.status_code,
        "reason": exc.reason,
    }


def _tool(fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except api.FfbridgeApiClientError as exc:
        return _client_error_payload(exc)


def _writer_tool(fn, *args, **kwargs) -> Dict[str, Any]:
    """Return a distinct, actionable error when the writer cannot serve."""
    try:
        return fn(*args, **kwargs)
    except api.FfbridgeApiClientError as exc:
        if exc.reason in ("sidecar_down", "sidecar_error", "timeout"):
            return {
                "ok": False,
                "error": "writer_unavailable",
                "reason": exc.reason,
                "detail": exc.detail,
                "hint": exc.hint,
                "http_status": exc.status_code,
            }
        return _client_error_payload(exc)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse(
        {"service": "ffbridge-postmortem-mcp", **api.writer_health()}
    )


@mcp.tool()
def ffbridge_postmortem_dataset_info() -> Dict[str, Any]:
    """Summary of the FFBridge postmortem cache and how to generate new
    postmortems with ffbridge_postmortem_generate (same path as Streamlit)."""
    return _tool(api.dataset_info)


@mcp.tool()
def ffbridge_postmortem_writer_health() -> Dict[str, Any]:
    """Cheap read-only writer readiness probe. Requires no player_id and does
    not authenticate, call Lancelot, or start generation. Safe to poll; also
    reports running jobs, the last job/error, and last parquet write time."""
    return api.writer_health()


@mcp.tool()
def ffbridge_postmortem_sessions(player_id: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """List cached FFBridge postmortem sessions (newest first), optionally for
    one player. player_id may be a Lancelot person id or FFBridge license
    number (246273 and 9500754 are the same player)."""
    return _tool(api.cached_sessions, player_id, limit)


@mcp.tool()
def ffbridge_postmortem_boards(
    player_id: str,
    session_id: Optional[str] = None,
    only_my_boards: bool = True,
    columns: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Per-board results for one cached FFBridge postmortem: contract,
    declarer, result, tricks, scores, matchpoint percentages, and the deal
    (PBN). session_id: omit for the player's most recently cached game.
    Generate missing sessions with ffbridge_postmortem_generate first."""
    return _tool(
        api.postmortem_boards,
        player_id,
        session_id,
        only_my_boards,
        columns,
        limit,
    )


@mcp.tool()
def ffbridge_postmortem_sql(
    player_id: str,
    sql: str,
    session_id: Optional[str] = None,
    limit: int = 500,
) -> Dict[str, Any]:
    """Run a DuckDB SQL query against one cached FFBridge postmortem,
    registered as table 'self'. Personalization macros: {Player_Direction},
    {Partner_Direction}, {Pair_Direction}, {Opponent_Pair_Direction}.
    Boolean helpers include Boards_I_Played, Boards_I_Declared,
    Boards_We_Declared, Boards_Opponent_Declared."""
    return _tool(api.postmortem_sql, player_id, sql, session_id, limit)


@mcp.tool()
def ffbridge_postmortem_schema(
    player_id: str,
    session_id: Optional[str] = None,
    pattern: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """Column names and dtypes of one cached FFBridge postmortem dataframe.
    Pass a case-insensitive regex pattern (e.g. 'Pct|Score', '^DD_', 'HCP')."""
    return _tool(api.postmortem_schema, player_id, session_id, pattern, limit)


@mcp.tool()
def ffbridge_postmortem_list_source_sessions(
    player_id: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """List playable Lancelot sessions for a player without generating.

    player_id: Lancelot person id, Classic/migration id, or FFBridge license
    number. Prefix with lancelot:, classic:, or license: to disambiguate.
    date_from / date_to are optional YYYY-MM-DD filters. Each row has
    session_id, date, club, and already_cached.
    """
    return _writer_tool(api.list_source_sessions, player_id, date_from, date_to)


@mcp.tool()
def ffbridge_postmortem_last_game(
    player: str,
    clubs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a player's latest published FFBridge game and a concise summary.

    player is required and may be a name, FFBridge license number, Lancelot
    person id, or Classic/migration id. Without clubs, searches configured
    simultaneous series. clubs optionally adds current-season ordinary games
    at named clubs, club codes, group IDs, or FFBridge group URLs.
    """
    return _writer_tool(api.last_game, player, clubs)


@mcp.tool()
def ffbridge_postmortem_played_today(
    player: str,
    clubs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Whether a required player has a published FFBridge result today.

    Input and club coverage are the same as ffbridge_postmortem_last_game.
    Returns structured game fields plus a human-readable summary for each
    matching game.
    """
    return _writer_tool(api.played_today, player, clubs)


@mcp.tool()
def ffbridge_postmortem_generate(
    player_id: str,
    session_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    force: bool = False,
    continue_on_error: Optional[bool] = None,
) -> Dict[str, Any]:
    """Generate postmortem parquet(s) via the FFBridge postmortem API
    (same Lancelot + augment path as Streamlit).

    player_id: Lancelot person id, Classic/migration id, or FFBridge license
    number; optional prefixes are lancelot:, classic:, and license:.
    session_id: omit (or 'latest') for the most recent game; set with
    date_from / date_to (YYYY-MM-DD) to generate a range.
    force rebuilds even if a cache file exists.
    continue_on_error defaults to true for date-range jobs: a bad session
    is recorded as status=error and the job continues.
    Slow work returns status=started or queued and a durable job_id; poll
    ffbridge_postmortem_generate_status (includes failed_session_ids).
    """
    return _writer_tool(
        api.generate,
        player_id,
        session_id,
        date_from,
        date_to,
        force,
        continue_on_error,
    )


@mcp.tool()
def ffbridge_postmortem_generate_status(job_id: str) -> Dict[str, Any]:
    """Poll a ffbridge_postmortem_generate job: status, progress,
    failed_session_ids, and per-session
    {session_id, player_id, status, cache_file, error} results."""
    return _writer_tool(api.generate_status, job_id)


if __name__ == "__main__":
    print(
        f"[ffbridge-postmortem-mcp] starting on :{FFBRIDGE_POSTMORTEM_MCP_PORT} "
        f"(endpoint /mcp, health /health); api -> {api.FFBRIDGE_POSTMORTEM_API_BASE_URL}",
        flush=True,
    )
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=FFBRIDGE_POSTMORTEM_MCP_PORT,
        stateless_http=True,
        json_response=True,
    )
