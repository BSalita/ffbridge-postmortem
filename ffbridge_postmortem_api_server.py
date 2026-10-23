"""FastAPI service for FFBridge postmortem cache + on-demand generate.

Library code lives in ffbridge_postmortem_create / ffbridge_postmortem_service.
The MCP server and Streamlit app are HTTP clients of this API.

  python ffbridge_postmortem_api_server.py
  GET http://127.0.0.1:8517/docs

Session predictions, with no player and nothing saved under a player:
  GET /predictions/boards?session_id= or ?club=&date=
  GET /predictions/rankings?session_id= or ?club=&date=
  POST /predictions/entries  (ranking predicted from the entered pairs, before play)
  GET  /predictions/entries  (same, from the entry list read live on FFBridge)
  GET  /entries/upcoming  (the service account's upcoming entries)
  date is YYYY-MM-DD. Omit it for the club's latest session.
  The player routes still read one cached postmortem:
  GET /postmortems/{player_id}/predictions/boards
  GET /postmortems/{player_id}/predictions/rankings

Favorite-group jobs (what the website runs as one postmortem):
  POST /postmortems/{player_id}/run-group
  GET  /run-group/{job_id}
  GET  /run-group/{job_id}/result
MCP clients call ffbridge_postmortem_report_start, then
ffbridge_postmortem_report_status, then
ffbridge_postmortem_report_result. They do not call /sql once per favorite.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import io
import os
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import ffbridge_entries_predictions as entries_predictions
import ffbridge_entry_list as entry_list
import ffbridge_postmortem_create as create
import ffbridge_postmortem_group_job as group_job
import ffbridge_postmortem_service as svc

FFBRIDGE_POSTMORTEM_API_PORT = int(os.environ.get("FFBRIDGE_POSTMORTEM_API_PORT", "8517"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    create.initialize_generate_jobs(svc.CACHE_DIR)
    yield
    create.shutdown_generate_jobs()


app = FastAPI(
    title="FFBridge Postmortem API",
    description=(
        "Cached postmortem reads plus on-demand Lancelot generate "
        "(same path Streamlit used to run in-process). "
        "Predictions for a session or a club and date, with no player: "
        "/predictions/boards and /predictions/rankings. "
        "A cached player's postmortem is "
        "/postmortems/{player_id}/predictions/boards and "
        "/postmortems/{player_id}/predictions/rankings. "
        "A brief, full, or predictions report is a job: "
        "POST /postmortems/{player_id}/run-group, then "
        "GET /run-group/{job_id}, then GET /run-group/{job_id}/result. "
        "MCP clients use ffbridge_postmortem_report_start, "
        "ffbridge_postmortem_report_status, and "
        "ffbridge_postmortem_report_result."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(FileNotFoundError)
async def not_found_handler(request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def key_error_handler(request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "hint": type(exc).__name__},
    )


@app.get("/health")
def health() -> dict:
    """Liveness. No archive scan and no cache directory listing.

    A full postmortem runs on a background thread. This handler stays on the
    event loop and reads in-memory job counters, so the 2s MCP health probe
    still returns while favorites are executing.
    """
    payload = {
        "ok": True,
        "sidecar_up": True,
        "status": "ok",
        "service": "ffbridge-postmortem-api",
        "detail": "ready",
        "group_runs": group_job.live_counts(),
    }
    payload.update(create.generate_health(svc.CACHE_DIR, include_files=False))
    return payload


@app.get("/info")
def dataset_info() -> dict:
    return svc.dataset_info()


@app.get("/players/resolve")
def resolve_player(
    player_id: str = Query(
        ...,
        description=(
            "Lancelot, Classic/migration, or FFBridge license id; optional "
            "lancelot:, classic:, or license: prefix"
        ),
    ),
) -> dict:
    resolved = create.resolve_player(player_id)
    return {
        "requested_id": resolved.requested_id,
        "player_id": resolved.lancelot_id,
        "player_license_number": resolved.license_number,
        "classic_person_id": resolved.classic_person_id,
        "aliases": resolved.aliases(),
    }


@app.get("/sessions")
def cached_sessions(
    player_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    sessions = svc.list_cached_postmortems(player_id)[:limit]
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/source-sessions")
def source_sessions(
    player_id: str,
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
) -> dict:
    return svc.list_source_sessions(player_id, date_from=date_from, date_to=date_to)


@app.get("/player-games/last")
def last_game(
    player: str = Query(
        ...,
        description="Required player name or FFBridge/Lancelot/Classic number",
    ),
    clubs: Optional[list[str]] = Query(
        None,
        description=(
            "Optional club names, codes, group IDs, or FFBridge group URLs. "
            "Without clubs, only simultaneous games are searched."
        ),
    ),
) -> dict:
    return svc.last_game(player, clubs)


@app.get("/player-games/played-today")
def played_today(
    player: str = Query(
        ...,
        description="Required player name or FFBridge/Lancelot/Classic number",
    ),
    clubs: Optional[list[str]] = Query(
        None,
        description=(
            "Optional club names, codes, group IDs, or FFBridge group URLs. "
            "Without clubs, only simultaneous games are searched."
        ),
    ),
) -> dict:
    return svc.played_today(player, clubs)


@app.get("/archive/rows")
def archive_rows(
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
    series_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    player_id: Optional[str] = Query(None),
    columns: Optional[str] = Query(None, description="Comma-separated projection"),
    limit: int = Query(svc.DEFAULT_SQL_ROW_LIMIT, ge=1, le=svc.MAX_SQL_ROW_LIMIT),
) -> dict:
    projected = [value.strip() for value in columns.split(",")] if columns else None
    return svc.archive_rows(
        date_from=date_from,
        date_to=date_to,
        series_id=series_id,
        session_id=session_id,
        player_id=player_id,
        columns=projected,
        limit=limit,
    )


@app.post("/generate")
def generate(
    player_id: str,
    session_id: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
    force: bool = Query(False),
    continue_on_error: Optional[bool] = Query(
        None,
        description="Default true for date-range jobs; one bad session does not abort the job.",
    ),
) -> dict:
    return svc.generate_postmortems(
        player_id,
        session_id=session_id,
        date_from=date_from,
        date_to=date_to,
        force=force,
        continue_on_error=continue_on_error,
    )


@app.get("/generate/{job_id}")
def generate_status(job_id: str) -> dict:
    return svc.generate_status(job_id)


@app.post("/postmortems/{player_id}/run-group")
def start_group_run(player_id: str, body: dict = Body(...)) -> dict:
    """Start every favorite in a group. Returns a job id before the SQL runs.

    JSON body: session_id, group (Summarize_Briefly, Summarize, or
    AI_Predictions), statements (list of {id, title, sql}), optional
    group_alias, force, and limit (max 2000). The session frame is loaded
    once. Poll GET /run-group/{job_id}. Read GET /run-group/{job_id}/result
    for favorites as they finish. The same player, session, group, and SQL
    returns the finished job with cached true. force true starts a new job.
    MCP clients call ffbridge_postmortem_report_start instead of this route.
    """
    return group_job.start_run_group(
        player_id,
        str(body.get("session_id") or ""),
        str(body.get("group") or ""),
        body.get("statements") or [],
        group_alias=body.get("group_alias"),
        force=bool(body.get("force") or False),
        limit=body.get("limit"),
    )


@app.get("/run-group/{job_id}")
def group_run_status(job_id: str) -> dict:
    """Status of a group run: queued, running, done, or failed.

    done and total count favorites. elapsed_s is seconds since start.
    completed lists each finished favorite with title, elapsed_s, row_count,
    and error. Poll until done or failed. MCP clients call
    ffbridge_postmortem_report_status.
    """
    return group_job.run_group_status(job_id)


@app.get("/run-group/{job_id}/result")
def group_run_result(
    job_id: str,
    offset: int = Query(0, ge=0, description="Favorite index to start at. Not a row index."),
    limit: Optional[int] = Query(
        None, ge=1, le=80, description="How many favorites to return. Omit for all finished favorites."
    ),
    title: Optional[str] = Query(
        None, description="One favorite title or id. Case-insensitive."
    ),
) -> dict:
    """Favorites finished so far, in group order, including while the job is running.

    offset and limit page by favorite, not by row. title returns one favorite.
    complete is true when status is done or failed. next_offset is the next
    page, or null when nothing further is waiting. Each section has columns,
    rows, row_count, and elapsed_s. results_file on a finished job is the
    server cache name, not a path the client can open. MCP clients call
    ffbridge_postmortem_report_result.
    """
    return group_job.run_group_result(
        job_id, offset=offset, limit=limit, title=title
    )


@app.get("/postmortems/{player_id}/meta")
def postmortem_meta(player_id: str, session_id: Optional[str] = None) -> dict:
    _df, meta = svc.load_postmortem(player_id, session_id)
    return meta


@app.get("/postmortems/{player_id}/boards")
def postmortem_boards(
    player_id: str,
    session_id: Optional[str] = None,
    only_my_boards: bool = Query(True),
    columns: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=svc.MAX_SQL_ROW_LIMIT),
) -> dict:
    cols = [c.strip() for c in columns.split(",")] if columns else None
    if session_id is not None:
        try:
            return svc.hierarchical_board_results(
                player_id,
                session_id,
                only_my_boards=only_my_boards,
                columns=cols,
                limit=limit,
            )
        except FileNotFoundError:
            pass
    df, meta = svc.load_postmortem(player_id, session_id)
    return svc.board_results(df, meta, only_my_boards=only_my_boards, columns=cols, limit=limit)


def _session_prediction(
    view: str,
    session_id: Optional[str],
    club: Optional[str],
    on_date: Optional[str],
    limit: int,
) -> dict:
    return svc.prediction_by_session(
        session_id=session_id,
        club=club,
        on_date=on_date,
        view=view,
        limit=limit,
    )


@app.get("/predictions/boards")
def session_prediction_boards(
    session_id: Optional[str] = Query(None, description="FFBridge session id. Enough on its own."),
    club: Optional[str] = Query(None, description="Club code or club name. Used when session_id is omitted."),
    date: Optional[str] = Query(None, description="YYYY-MM-DD. Omit for the club's latest session."),
    limit: int = Query(svc.DEFAULT_SQL_ROW_LIMIT, ge=1, le=svc.MAX_SQL_ROW_LIMIT),
) -> dict:
    """Board predictions for a session, or for a club on a date. No player is saved."""
    return _session_prediction("boards", session_id, club, date, limit)


@app.get("/predictions/rankings")
def session_prediction_rankings(
    session_id: Optional[str] = Query(None, description="FFBridge session id. Enough on its own."),
    club: Optional[str] = Query(None, description="Club code or club name. Used when session_id is omitted."),
    date: Optional[str] = Query(None, description="YYYY-MM-DD. Omit for the club's latest session."),
    limit: int = Query(svc.DEFAULT_SQL_ROW_LIMIT, ge=1, le=svc.MAX_SQL_ROW_LIMIT),
) -> dict:
    """Club prediction leaderboard for a session, or for a club on a date."""
    return _session_prediction("rankings", session_id, club, date, limit)


@app.post("/predictions/entries")
def session_prediction_entries(payload: dict = Body(...)) -> dict:
    """Predicted ranking from a session's entries, before it is played.

    Body: {"entries": [["licence or name", "licence or name"], ...],
    "date": "YYYY-MM-DD", "sims": 20000}. An entry may also be
    "A / B" or {"players": [a, b], "orientation": "NS"|"EW", "label": "..."}.
    """
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("entries must be a list of pairs")
    return entries_predictions.predict_entries(
        entries,
        payload.get("date"),
        sims=int(payload.get("sims") or entries_predictions.DEFAULT_SIMS),
    )


@app.get("/predictions/entries")
def session_prediction_entry_list(
    group_session_id: Optional[str] = Query(
        None,
        description="Lancelot group session id. Omit for the service account's next entered session.",
    ),
    sims: int = Query(entries_predictions.DEFAULT_SIMS, ge=100, le=entries_predictions.MAX_SIMS),
) -> dict:
    """The same forecast, from the session's entry list read live on FFBridge."""
    return entry_list.predict_group_session(group_session_id, sims=sims)


@app.get("/entries/upcoming")
def upcoming_entries(
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD. Omit for today."),
    player_id: Optional[str] = Query(None, description="Lancelot person id. Omit for the service account."),
    with_forecast: bool = Query(False, description="Add the player's forecast rank and pct to each entry."),
    sims: int = Query(5000, ge=500, le=100000),
) -> dict:
    """A player's upcoming entries (group sessions), soonest first.

    403 when FFBridge keeps that player's entries private.
    """
    try:
        if with_forecast:
            rows = entry_list.entries_with_forecast(player_id, date_from, sims=sims)
        else:
            rows = entry_list.upcoming_entries(date_from, person_id=player_id)
    except entry_list.EntriesPrivate as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"sessions": rows, "count": len(rows)}


@app.get("/postmortems/{player_id}/predictions/boards")
def prediction_boards(
    player_id: str,
    session_id: Optional[str] = None,
    limit: int = Query(svc.DEFAULT_SQL_ROW_LIMIT, ge=1, le=svc.MAX_SQL_ROW_LIMIT),
) -> dict:
    """Board-by-board contract, declarer, and percentage predictions."""
    return svc.prediction_report(player_id, session_id, "boards", limit)


@app.get("/postmortems/{player_id}/predictions/rankings")
def prediction_rankings(
    player_id: str,
    session_id: Optional[str] = None,
    limit: int = Query(svc.DEFAULT_SQL_ROW_LIMIT, ge=1, le=svc.MAX_SQL_ROW_LIMIT),
) -> dict:
    """Club rank, our global rank, FFBridge's global rank, and the difference."""
    return svc.prediction_report(player_id, session_id, "rankings", limit)


@app.get("/postmortems/{player_id}/sql")
def postmortem_sql(
    player_id: str,
    sql: str,
    session_id: Optional[str] = None,
    limit: int = Query(svc.DEFAULT_SQL_ROW_LIMIT, ge=1, le=svc.MAX_SQL_ROW_LIMIT),
) -> dict:
    df, meta = svc.load_postmortem(player_id, session_id)
    result = svc.run_sql(df, sql, meta, limit=limit)
    result["meta"] = meta
    return result


@app.get("/postmortems/{player_id}/schema")
def postmortem_schema(
    player_id: str,
    session_id: Optional[str] = None,
    pattern: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=svc.MAX_SCHEMA_COLUMNS),
) -> dict:
    df, _ = svc.load_postmortem(player_id, session_id)
    return svc.schema_columns(df, pattern=pattern, limit=limit)


@app.get("/postmortems/{player_id}/parquet")
def postmortem_parquet(player_id: str, session_id: Optional[str] = None) -> Response:
    frame, meta = svc.load_postmortem(str(player_id), session_id)
    buffer = io.BytesIO()
    frame.write_parquet(buffer, compression="zstd")
    filename = f"df-{meta['session_id']}-{meta.get('matched_player_id') or player_id}.parquet"
    return Response(
        content=buffer.getvalue(),
        media_type="application/vnd.apache.parquet",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "X-FFBridge-Cache-File": filename,
            "X-FFBridge-Data-Source": meta["data_source"],
        },
    )


if __name__ == "__main__":
    import uvicorn

    print(
        f"[ffbridge-postmortem-api] starting on :{FFBRIDGE_POSTMORTEM_API_PORT}; "
        f"cache -> {svc.CACHE_DIR}",
        flush=True,
    )
    uvicorn.run(
        "ffbridge_postmortem_api_server:app",
        host="0.0.0.0",
        port=FFBRIDGE_POSTMORTEM_API_PORT,
        reload=False,
    )
