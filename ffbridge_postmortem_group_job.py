"""Run one postmortem favorite group without holding an HTTP request open.

The website loads a session once and runs every favorite against that frame.
This job does the same work in a background thread. Callers get a job id
immediately, then poll status and read finished favorites while the rest
are still running. A finished group is cached by player, session, group,
and the SQL that was run, so the same postmortem returns on the next start.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import duckdb

import ffbridge_postmortem_service as svc

_MAX_STATEMENTS = 80
_MAX_SQL_CHARS = 20000
_LOCK = threading.RLock()
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOB_DIRS: Dict[str, pathlib.Path] = {}
_THREADS: Dict[str, threading.Thread] = {}
_CONNECTIONS: Dict[str, Any] = {}
_CANCELLED: set[str] = set()
_DUCKDB_THREADS = 2


def _valid_job_id(job_id: str) -> bool:
    return bool(job_id) and all(character.isalnum() or character in "-_" for character in job_id)


def _store_dir(cache_dir: pathlib.Path) -> pathlib.Path:
    directory = pathlib.Path(cache_dir) / "group_runs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _job_path(cache_dir: pathlib.Path, job_id: str) -> pathlib.Path:
    if not _valid_job_id(job_id):
        raise KeyError(f"Unknown group job_id {job_id}")
    return _store_dir(cache_dir) / f"{job_id}.json"


def _cache_pointer(cache_dir: pathlib.Path, cache_key: str) -> pathlib.Path:
    if not _valid_job_id(cache_key):
        raise KeyError(f"Unknown group cache key {cache_key}")
    return _store_dir(cache_dir) / f"cache-{cache_key}.json"


def _cache_key(player_id: str, session_id: str, group: str, statements: List[Dict[str, str]]) -> str:
    digest = hashlib.sha256()
    digest.update(player_id.encode())
    digest.update(b"\0")
    digest.update(session_id.encode())
    digest.update(b"\0")
    digest.update(group.encode())
    for statement in statements:
        digest.update(b"\0")
        digest.update(statement["id"].encode())
        digest.update(b"\0")
        digest.update(statement["title"].encode())
        digest.update(b"\0")
        digest.update(statement["sql"].encode())
    return digest.hexdigest()


def _normalize_statements(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("statements are required")
    if len(raw) > _MAX_STATEMENTS:
        raise ValueError(f"a group can run at most {_MAX_STATEMENTS} favorites")
    statements: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each statement must be an object")
        sql = str(item.get("sql") or "").strip()
        title = str(item.get("title") or "").strip()
        favorite_id = str(item.get("id") or "").strip()
        if not sql or not title:
            raise ValueError("each statement needs a title and sql")
        if len(sql) > _MAX_SQL_CHARS:
            raise ValueError(f"sql for {title} is too long")
        statements.append({"id": favorite_id or title, "title": title, "sql": sql})
    return statements


def _clamp_limit(limit: Optional[int]) -> int:
    if limit is None:
        return svc.MAX_SQL_ROW_LIMIT
    return max(1, min(int(limit), svc.MAX_SQL_ROW_LIMIT))


def _freeze(payload: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload, default=str))


def _persist(job: Dict[str, Any]) -> None:
    cache_dir = _JOB_DIRS[job["job_id"]]
    path = _job_path(cache_dir, job["job_id"])
    temporary = path.with_name(
        f".{job['job_id']}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)
    if job.get("status") == "done" and job.get("cache_key"):
        pointer = _cache_pointer(cache_dir, job["cache_key"])
        pointer_tmp = pointer.with_name(f".{pointer.name}.{os.getpid()}.tmp")
        pointer_tmp.write_text(
            json.dumps({"job_id": job["job_id"]}),
            encoding="utf-8",
        )
        os.replace(pointer_tmp, pointer)


def _remember(job: Dict[str, Any], cache_dir: pathlib.Path) -> None:
    _JOBS[job["job_id"]] = job
    _JOB_DIRS[job["job_id"]] = pathlib.Path(cache_dir)


def _load_job(job_id: str, cache_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    if not _valid_job_id(job_id):
        raise KeyError(f"Unknown group job_id {job_id}")
    with _LOCK:
        cached = _JOBS.get(job_id)
        if cached is not None:
            return cached
        directory = pathlib.Path(cache_dir) if cache_dir is not None else _JOB_DIRS.get(
            job_id, svc.CACHE_DIR
        )
    path = _job_path(directory, job_id)
    if not path.is_file():
        raise KeyError(f"Unknown group job_id {job_id}")
    job = json.loads(path.read_text(encoding="utf-8"))
    with _LOCK:
        _remember(job, path.parent.parent)
    return job


def _elapsed(job: Dict[str, Any]) -> float:
    started = job.get("started_epoch")
    if started is None:
        return 0.0
    finished = job.get("finished_epoch")
    end = finished if finished is not None else time.time()
    return round(float(end) - float(started), 3)


def _results_file(job: Dict[str, Any]) -> Optional[str]:
    if job.get("status") != "done":
        return None
    return f"group_runs/{job['job_id']}.json"


def _public_start(job: Dict[str, Any], *, cached: bool) -> Dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "cached": cached,
        "player_id": job["player_id"],
        "session_id": job["session_id"],
        "group": job["group"],
        "group_alias": job.get("group_alias"),
        "done": job["done"],
        "total": job["total"],
        "elapsed_s": _elapsed(job),
        "results_file": _results_file(job),
    }


def _snapshot(job_id: str) -> Dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            job = _load_job(job_id)
        return {
            **{key: value for key, value in job.items() if key != "sections"},
            "sections": list(job.get("sections") or []),
        }


def _mark(job_id: str, **changes: Any) -> None:
    with _LOCK:
        job = _JOBS[job_id]
        job.update(changes)
        job["heartbeat"] = time.time()
        if changes.get("status") in ("done", "failed"):
            job["finished_epoch"] = time.time()
        _persist(job)


def _cached_done_job(
    cache_dir: pathlib.Path, cache_key: str
) -> Optional[Dict[str, Any]]:
    pointer = _cache_pointer(cache_dir, cache_key)
    if not pointer.is_file():
        return None
    try:
        job_id = str(json.loads(pointer.read_text(encoding="utf-8")).get("job_id") or "")
        job = _load_job(job_id, cache_dir)
    except (OSError, ValueError, KeyError):
        return None
    if job.get("status") != "done":
        return None
    return job


def _inflight_locked(cache_key: str) -> Optional[Dict[str, Any]]:
    for job in _JOBS.values():
        if job.get("cache_key") != cache_key:
            continue
        if job.get("status") not in ("queued", "running"):
            continue
        thread = _THREADS.get(str(job.get("job_id")))
        if thread is not None and thread.is_alive():
            return job
    return None


def _cancelled(job_id: str) -> bool:
    with _LOCK:
        return job_id in _CANCELLED


def _cancel_locked(job_id: str) -> None:
    """Stop a running group so a replacement does not stack a second full scan."""
    _CANCELLED.add(job_id)
    connection = _CONNECTIONS.get(job_id)
    if connection is not None:
        try:
            connection.interrupt()
        except Exception:
            pass
    if job_id in _JOBS and _JOBS[job_id].get("status") in ("queued", "running"):
        _mark(
            job_id,
            status="failed",
            error="group run was replaced before it finished",
        )


def live_counts() -> Dict[str, int]:
    with _LOCK:
        statuses = [str(job.get("status") or "") for job in _JOBS.values()]
    return {
        "running": statuses.count("running"),
        "queued": statuses.count("queued"),
    }


def start_run_group(
    player_id: str,
    session_id: str,
    group: str,
    statements: List[Dict[str, Any]],
    *,
    group_alias: Optional[str] = None,
    force: bool = False,
    limit: Optional[int] = None,
    cache_dir: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Queue every statement. Returns before the first favorite finishes."""
    player = str(player_id or "").strip()
    session = str(session_id or "").strip()
    group_id = str(group or "").strip()
    if not player or not session or not group_id:
        raise ValueError("player_id, session_id, and group are required")
    normalized = _normalize_statements(statements)
    directory = pathlib.Path(cache_dir) if cache_dir is not None else svc.CACHE_DIR
    key = _cache_key(player, session, group_id, normalized)
    with _LOCK:
        running = _inflight_locked(key)
        if running is not None and not force:
            return _public_start(running, cached=False)
        if running is not None and force:
            _cancel_locked(running["job_id"])
        if not force:
            existing = _cached_done_job(directory, key)
            if existing is not None:
                return _public_start(existing, cached=True)
        now = time.time()
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "status": "queued",
            "player_id": player,
            "session_id": session,
            "group": group_id,
            "group_alias": group_alias or group_id,
            "cache_key": key,
            "statements": normalized,
            "limit": _clamp_limit(limit),
            "done": 0,
            "total": len(normalized),
            "sections": [],
            "error": None,
            "warning": None,
            "started_epoch": now,
            "finished_epoch": None,
            "heartbeat": now,
        }
        _remember(job, directory)
        _persist(job)
        thread = threading.Thread(
            target=_run_job,
            args=(job_id,),
            name=f"ffbridge-group-{job_id[:8]}",
            daemon=True,
        )
        _THREADS[job_id] = thread
        thread.start()
    return _public_start(job, cached=False)


def _execute_statement(con: Any, sql: str, meta: Dict[str, Any], limit: int) -> Dict[str, Any]:
    prepared = svc.prepare_postmortem_sql(sql, meta)
    try:
        result = svc.polars_result(con.execute(prepared))
    except Exception as exc:
        return {
            "error": str(exc),
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
        }
    payload = svc.sql_result_payload(result, prepared, limit)
    payload.pop("sql", None)
    return payload


def _prepare_frame(df: Any, meta: Dict[str, Any], statements: List[Dict[str, str]]) -> Any:
    import ffbridge_session_predictions as session_predictions

    if not any(session_predictions.sql_needs_predictions(item["sql"]) for item in statements):
        return df, None
    try:
        predicted, info = session_predictions.apply_session_predictions(
            df, meta.get("session_id"), svc.CACHE_DIR
        )
    except Exception as exc:
        return df, str(exc)
    warning = info.get("warning") if isinstance(info, dict) else None
    return predicted, warning


def _run_job(job_id: str) -> None:
    job = _load_job(job_id)
    try:
        _mark(job_id, status="running")
        df, meta = svc.load_postmortem(job["player_id"], job["session_id"])
        frame, warning = _prepare_frame(df, meta, job["statements"])
        if warning:
            _mark(job_id, warning=warning)
        con = duckdb.connect(config={"enable_external_access": "false"})
        try:
            con.execute(f"SET threads={_DUCKDB_THREADS}")
            with _LOCK:
                _CONNECTIONS[job_id] = con
            con.register(svc.CON_REGISTER_NAME, frame)
            for statement in job["statements"]:
                if _cancelled(job_id):
                    _mark(
                        job_id,
                        status="failed",
                        error="group run was replaced before it finished",
                    )
                    return
                started = time.perf_counter()
                payload = _execute_statement(con, statement["sql"], meta, job["limit"])
                payload.update(
                    {
                        "id": statement["id"],
                        "title": statement["title"],
                        "elapsed_s": round(time.perf_counter() - started, 3),
                    }
                )
                section = _freeze(payload)
                with _LOCK:
                    current = _JOBS[job_id]
                    current["sections"].append(section)
                    current["done"] = len(current["sections"])
                    current["status"] = "running"
                    current["heartbeat"] = time.time()
                    _persist(current)
        finally:
            with _LOCK:
                _CONNECTIONS.pop(job_id, None)
            con.close()
        if not _cancelled(job_id):
            _mark(job_id, status="done")
    except Exception as exc:
        if not _cancelled(job_id):
            _mark(job_id, status="failed", error=str(exc))
    finally:
        with _LOCK:
            _CANCELLED.discard(job_id)
            _THREADS.pop(job_id, None)


def _interrupt_if_abandoned(job: Dict[str, Any]) -> Dict[str, Any]:
    if job.get("status") not in ("queued", "running"):
        return job
    thread = _THREADS.get(job["job_id"])
    if thread is not None and thread.is_alive():
        return job
    heartbeat = float(job.get("heartbeat") or 0)
    if time.time() - heartbeat < 30:
        return job
    _mark(
        job["job_id"],
        status="failed",
        error="group run stopped before it finished",
    )
    return _load_job(job["job_id"])


def run_group_status(job_id: str) -> Dict[str, Any]:
    job = _interrupt_if_abandoned(_load_job(job_id))
    snapshot = _snapshot(job["job_id"])
    completed = [
        {
            "id": section.get("id"),
            "title": section.get("title"),
            "elapsed_s": section.get("elapsed_s"),
            "row_count": section.get("row_count", 0),
            "error": section.get("error"),
        }
        for section in snapshot["sections"]
    ]
    return {
        "job_id": snapshot["job_id"],
        "status": snapshot["status"],
        "player_id": snapshot["player_id"],
        "session_id": snapshot["session_id"],
        "group": snapshot["group"],
        "group_alias": snapshot.get("group_alias"),
        "done": snapshot["done"],
        "total": snapshot["total"],
        "elapsed_s": _elapsed(snapshot),
        "completed": completed,
        "error": snapshot.get("error"),
        "warning": snapshot.get("warning"),
        "results_file": _results_file(snapshot),
    }


def _section_matches(section: Dict[str, Any], title: Optional[str]) -> bool:
    if not title:
        return True
    wanted = title.strip().casefold()
    return wanted in {
        str(section.get("title") or "").casefold(),
        str(section.get("id") or "").casefold(),
    }


def run_group_result(
    job_id: str,
    *,
    offset: int = 0,
    limit: Optional[int] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Finished favorites so far, in group order. Pages by favorite, not by row."""
    job = _interrupt_if_abandoned(_load_job(job_id))
    snapshot = _snapshot(job["job_id"])
    matched = [
        section
        for section in snapshot["sections"]
        if _section_matches(section, title)
    ]
    start = max(0, int(offset))
    if limit is None:
        page = matched[start:]
    else:
        page = matched[start : start + max(1, int(limit))]
    next_offset = start + len(page)
    finished = snapshot["status"] in ("done", "failed")
    exhausted = next_offset >= len(matched)
    if exhausted and finished:
        next_offset_value: Optional[int] = None
    elif exhausted:
        next_offset_value = len(matched)
    else:
        next_offset_value = next_offset
    return {
        "job_id": snapshot["job_id"],
        "status": snapshot["status"],
        "complete": finished,
        "player_id": snapshot["player_id"],
        "session_id": snapshot["session_id"],
        "group": snapshot["group"],
        "group_alias": snapshot.get("group_alias"),
        "done": snapshot["done"],
        "total": snapshot["total"],
        "elapsed_s": _elapsed(snapshot),
        "offset": start,
        "limit": limit,
        "next_offset": next_offset_value,
        "error": snapshot.get("error"),
        "warning": snapshot.get("warning"),
        "results_file": _results_file(snapshot),
        "sections": page,
    }
