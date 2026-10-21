"""Async favorite-group jobs: one frame, progress, cache, and paging."""

from __future__ import annotations

import pathlib
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import duckdb
import polars as pl

import ffbridge_postmortem_group_job as group_job


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Board": [1, 2],
            "Contract": ["3NT", "4H"],
            "Result": [1, -1],
            "Score_NS": [430, -100],
            "Pct_NS": [0.8, 0.2],
        }
    )


def _meta() -> dict:
    return {
        "session_id": "394099",
        "player_direction": "N",
        "partner_direction": "S",
        "pair_direction": "NS",
        "opponent_pair_direction": "EW",
    }


def _statements() -> list[dict]:
    return [
        {
            "id": "Boards_We_Played",
            "title": "Boards Your Partnership Played",
            "sql": "SELECT Board, Result, Contract, Score_{Pair_Direction} FROM self ORDER BY Board",
        },
        {
            "id": "Bad",
            "title": "Bad favorite",
            "sql": "SELECT Not_A_Column FROM self",
        },
        {
            "id": "Best",
            "title": "Best boards",
            "sql": "SELECT Board, Contract FROM self WHERE Board = 1",
        },
    ]


def _wait(job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.time() + timeout_s
    status = {}
    while time.time() < deadline:
        status = group_job.run_group_status(job_id)
        if status["status"] in ("done", "failed"):
            return status
        time.sleep(0.02)
    raise AssertionError(status)


class GroupJobTests(unittest.TestCase):
    def test_one_frame_runs_every_favorite_and_caches_the_group(self) -> None:
        loads = {"n": 0}
        connects = {"n": 0}
        real_connect = duckdb.connect

        def counting_connect(*args, **kwargs):
            connects["n"] += 1
            return real_connect(*args, **kwargs)

        def fake_load(player_id, session_id=None):
            loads["n"] += 1
            self.assertEqual(player_id, "246273")
            self.assertEqual(session_id, "394099")
            return _frame(), _meta()

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = pathlib.Path(tmp)
            with patch.object(group_job.svc, "load_postmortem", fake_load), patch.object(
                group_job.duckdb, "connect", counting_connect
            ):
                started = group_job.start_run_group(
                    "246273",
                    "394099",
                    "Summarize",
                    _statements(),
                    group_alias="Full postmortem",
                    cache_dir=cache_dir,
                )
                self.assertIn(started["status"], ("queued", "running", "done"))
                self.assertFalse(started["cached"])
                status = _wait(started["job_id"])
            self.assertEqual(status["status"], "done")
            self.assertEqual(status["done"], 3)
            self.assertEqual(status["total"], 3)
            self.assertGreaterEqual(status["elapsed_s"], 0)
            self.assertEqual(loads["n"], 1)
            self.assertEqual(connects["n"], 1)
            titles = [item["title"] for item in status["completed"]]
            self.assertEqual(
                titles,
                ["Boards Your Partnership Played", "Bad favorite", "Best boards"],
            )
            self.assertTrue(all(item["elapsed_s"] is not None for item in status["completed"]))
            self.assertIsNotNone(status["completed"][1]["error"])
            self.assertEqual(status["results_file"], f"group_runs/{started['job_id']}.json")
            self.assertTrue((cache_dir / status["results_file"]).is_file())

            result = group_job.run_group_result(started["job_id"], offset=0, limit=1)
            self.assertEqual(len(result["sections"]), 1)
            self.assertEqual(result["next_offset"], 1)
            played = result["sections"][0]
            self.assertIn("Result", played["columns"])
            self.assertEqual(played["rows"][0]["Result"], 1)
            self.assertEqual(played["rows"][0]["Score_NS"], 430)

            page = group_job.run_group_result(started["job_id"], title="Best boards")
            self.assertEqual(page["sections"][0]["row_count"], 1)
            self.assertIsNone(page["sections"][0].get("error"))

            with patch.object(group_job.svc, "load_postmortem", fake_load):
                again = group_job.start_run_group(
                    "246273",
                    "394099",
                    "Summarize",
                    _statements(),
                    cache_dir=cache_dir,
                )
            self.assertTrue(again["cached"])
            self.assertEqual(again["job_id"], started["job_id"])
            self.assertEqual(again["status"], "done")
            self.assertEqual(loads["n"], 1)

            with patch.object(group_job.svc, "load_postmortem", fake_load):
                forced = group_job.start_run_group(
                    "246273",
                    "394099",
                    "Summarize",
                    _statements(),
                    force=True,
                    cache_dir=cache_dir,
                )
                self.assertFalse(forced["cached"])
                self.assertNotEqual(forced["job_id"], started["job_id"])
                self.assertEqual(_wait(forced["job_id"])["status"], "done")
            self.assertEqual(loads["n"], 2)

    def test_partial_results_are_readable_while_the_group_is_running(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        calls = {"n": 0}
        real = group_job._execute_statement

        def slow(con, sql, meta, limit):
            calls["n"] += 1
            if calls["n"] == 2:
                entered.set()
                self.assertTrue(release.wait(5))
            return real(con, sql, meta, limit)

        def fake_load(player_id, session_id=None):
            return _frame(), _meta()

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(group_job.svc, "load_postmortem", fake_load), patch.object(
                group_job, "_execute_statement", slow
            ):
                started = group_job.start_run_group(
                    "246273",
                    "394099",
                    "Summarize_Briefly",
                    _statements(),
                    cache_dir=pathlib.Path(tmp),
                )
                self.assertTrue(entered.wait(5))
                status = group_job.run_group_status(started["job_id"])
                self.assertEqual(status["status"], "running")
                self.assertEqual(status["done"], 1)
                self.assertEqual(status["total"], 3)
                partial = group_job.run_group_result(started["job_id"])
                self.assertFalse(partial["complete"])
                self.assertEqual(len(partial["sections"]), 1)
                self.assertEqual(partial["sections"][0]["title"], "Boards Your Partnership Played")
                again = group_job.start_run_group(
                    "246273",
                    "394099",
                    "Summarize_Briefly",
                    _statements(),
                    cache_dir=pathlib.Path(tmp),
                )
                self.assertEqual(again["job_id"], started["job_id"])
                self.assertEqual(again["status"], "running")
                self.assertFalse(again["cached"])
                release.set()
                finished = _wait(started["job_id"])
            self.assertEqual(finished["status"], "done")
            self.assertEqual(finished["done"], 3)
            self.assertEqual(calls["n"], 3)

    def test_a_missing_session_fails_the_job(self) -> None:
        def fake_load(player_id, session_id=None):
            raise FileNotFoundError(f"no postmortem for {session_id}")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(group_job.svc, "load_postmortem", fake_load):
                started = group_job.start_run_group(
                    "246273",
                    "394099",
                    "Summarize",
                    _statements()[:1],
                    cache_dir=pathlib.Path(tmp),
                )
                status = _wait(started["job_id"])
        self.assertEqual(status["status"], "failed")
        self.assertIn("394099", status["error"])
        self.assertEqual(status["done"], 0)

    def test_route_starts_the_job(self) -> None:
        import ffbridge_postmortem_api_server as api

        with patch.object(
            api.group_job,
            "start_run_group",
            return_value={"job_id": "abc", "status": "queued"},
        ) as start:
            payload = api.start_group_run(
                "246273",
                {
                    "session_id": "394099",
                    "group": "Summarize",
                    "statements": _statements()[:1],
                    "force": False,
                },
            )
        self.assertEqual(payload["job_id"], "abc")
        start.assert_called_once()
        args, kwargs = start.call_args
        self.assertEqual(args[0], "246273")
        self.assertEqual(args[1], "394099")
        self.assertEqual(args[2], "Summarize")


class DuckDbStatementTests(unittest.TestCase):
    def test_statement_reads_arrow_when_connection_has_no_pl(self) -> None:
        class _Result:
            def fetch_arrow_table(self):
                return pl.DataFrame({"Board": [1], "Contract": ["3NT"]}).to_arrow()

        class _Connection:
            def execute(self, sql):
                return _Result()

        payload = group_job._execute_statement(
            _Connection(),
            "SELECT Board, Contract FROM self",
            _meta(),
            10,
        )
        self.assertNotIn("error", payload)
        self.assertEqual(payload["rows"], [{"Board": 1, "Contract": "3NT"}])


if __name__ == "__main__":
    unittest.main()
