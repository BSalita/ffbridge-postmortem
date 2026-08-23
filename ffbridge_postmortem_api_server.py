"""FastAPI service for FFBridge postmortem cache + on-demand generate.

Library code lives in ffbridge_postmortem_create / ffbridge_postmortem_service.
The MCP server and Streamlit app are HTTP clients of this API.

  python ffbridge_postmortem_api_server.py
  GET http://127.0.0.1:8517/docs
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import ffbridge_postmortem_create as create
import ffbridge_postmortem_service as svc

FFBRIDGE_POSTMORTEM_API_PORT = int(os.environ.get("FFBRIDGE_POSTMORTEM_API_PORT", "8517"))

app = FastAPI(
    title="FFBridge Postmortem API",
    description=(
        "Cached postmortem reads plus on-demand Lancelot generate "
        "(same path Streamlit used to run in-process)."
    ),
    version="1.0.0",
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


@app.get("/health")
def health() -> dict:
    """Cheap readiness probe: no cache scan, authentication, or Lancelot call."""
    return {
        "ok": True,
        "sidecar_up": True,
        "status": "ok",
        "service": "ffbridge-postmortem-api",
        "detail": "ready",
    }


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
    df, meta = svc.load_postmortem(player_id, session_id)
    cols = [c.strip() for c in columns.split(",")] if columns else None
    return svc.board_results(df, meta, only_my_boards=only_my_boards, columns=cols, limit=limit)


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
    path = svc.resolve_cache_file(str(player_id), session_id)
    return Response(
        content=path.read_bytes(),
        media_type="application/vnd.apache.parquet",
        headers={
            "Content-Disposition": f'inline; filename="{path.name}"',
            "X-FFBridge-Cache-File": path.name,
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
