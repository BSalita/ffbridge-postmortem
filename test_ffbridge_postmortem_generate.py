import pathlib
import tempfile
import unittest
from unittest.mock import Mock, patch

import polars as pl

import ffbridge_postmortem_create as create


class GenerateJobTests(unittest.TestCase):
    def tearDown(self):
        with create._jobs_lock:
            for job_id in ("test-job", "other-job"):
                create._jobs.pop(job_id, None)
                create._job_threads.pop(job_id, None)
                create._job_store_dirs.pop(job_id, None)

    def _job(self, *, continue_on_error: bool = True) -> create.GenerateJob:
        return create.GenerateJob(
            job_id="test-job",
            status="started",
            player_id="246273",
            player_license_number="9500754",
            requested_id="9500754",
            session_ids=["bad", "good"],
            force=False,
            started_at="2026-08-23T08:13:02",
            continue_on_error=continue_on_error,
        )

    @staticmethod
    def _create_result(_player_id, session_id, *_args, **_kwargs):
        if session_id == "bad":
            raise ValueError("bad score")
        return {
            "session_id": session_id,
            "player_id": "246273",
            "status": "ok",
            "cache_file": f"df-{session_id}-246273.parquet",
            "meta": None,
        }

    def test_range_job_records_error_and_continues(self):
        job = self._job()
        sessions = {"sessions": [{"session_id": "bad"}, {"session_id": "good"}]}

        with (
            patch.object(create, "list_source_sessions", return_value=sessions),
            patch.object(create, "create_lancelot_postmortem", side_effect=self._create_result),
            patch.object(create, "tqdm", side_effect=lambda values, **_kwargs: values),
            patch.object(create, "_persist_job"),
        ):
            create._run_generate_job(
                job,
                create.LancelotAuth("token", "246273", "9500754"),
                pathlib.Path("cache"),
                False,
            )

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.failed_session_ids, ["bad"])
        self.assertEqual([result["status"] for result in job.results], ["error", "ok"])
        self.assertEqual(job.results[0]["error"], "bad score")
        self.assertEqual(job.progress["done"], 2)
        self.assertIsNone(job.progress["current_session_id"])

    def test_stop_on_error_counts_failed_session(self):
        job = self._job(continue_on_error=False)
        sessions = {"sessions": [{"session_id": "bad"}, {"session_id": "good"}]}

        with (
            patch.object(create, "list_source_sessions", return_value=sessions),
            patch.object(create, "create_lancelot_postmortem", side_effect=self._create_result),
            patch.object(create, "tqdm", side_effect=lambda values, **_kwargs: values),
            patch.object(create, "_persist_job"),
        ):
            create._run_generate_job(
                job,
                create.LancelotAuth("token", "246273", "9500754"),
                pathlib.Path("cache"),
                False,
            )

        self.assertEqual(job.status, "error")
        self.assertEqual(job.error, "bad score")
        self.assertEqual(job.progress["done"], 1)
        self.assertEqual(len(job.results), 1)

    def test_session_list_failure_finishes_job_as_error(self):
        job = self._job()

        with (
            patch.object(create, "list_source_sessions", side_effect=RuntimeError("API unavailable")),
            patch.object(create, "_persist_job"),
        ):
            create._run_generate_job(
                job,
                create.LancelotAuth("token", "246273", "9500754"),
                pathlib.Path("cache"),
                False,
            )

        self.assertEqual(job.status, "error")
        self.assertEqual(job.error, "API unavailable")
        self.assertIsNotNone(job.finished_at)
        self.assertIsNone(job.progress["current_session_id"])

    def test_generate_status_loads_job_after_memory_store_is_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = pathlib.Path(tmp)
            job = self._job()
            job.status = "running"
            job.owner_id = "previous-process"
            create._persist_job(job, cache_dir)
            with create._jobs_lock:
                create._jobs.pop(job.job_id, None)
                create._job_store_dirs.pop(job.job_id, None)

            with patch.object(create, "DEFAULT_CACHE_DIR", cache_dir):
                result = create.generate_status(job.job_id)

            self.assertEqual(result["job_id"], job.job_id)
            self.assertEqual(result["status"], "running")
            self.assertEqual(result["owner_id"], "previous-process")

    def test_stale_job_is_claimed_for_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = pathlib.Path(tmp)
            job = self._job()
            job.status = "running"
            job.owner_id = "dead-process"
            job.heartbeat_at = "2020-01-01T00:00:00+00:00"
            create._persist_job(job, cache_dir)
            with create._jobs_lock:
                create._jobs.pop(job.job_id, None)
                create._job_threads.pop(job.job_id, None)

            auth = create.LancelotAuth("token", "246273", "9500754")
            with (
                patch.object(create, "ensure_lancelot_auth", return_value=auth),
                patch.object(create, "_start_job_thread") as start_job,
            ):
                create._recover_jobs_once(cache_dir)

            recovered = create._load_job_file(
                cache_dir / create._JOB_STORE_DIRNAME / f"{job.job_id}.json"
            )
            self.assertEqual(recovered.status, "recovering")
            self.assertEqual(recovered.owner_id, create._PROCESS_INSTANCE_ID)
            self.assertEqual(recovered.recovery_count, 1)
            start_job.assert_called_once()

    def test_repeatedly_interrupted_session_is_skipped_and_range_continues(self):
        job = self._job()
        job.session_attempts["bad"] = create._MAX_SESSION_PROCESS_ATTEMPTS
        sessions = {"sessions": [{"session_id": "bad"}, {"session_id": "good"}]}

        with (
            patch.object(create, "list_source_sessions", return_value=sessions),
            patch.object(
                create,
                "create_lancelot_postmortem",
                side_effect=self._create_result,
            ) as build,
            patch.object(create, "tqdm", side_effect=lambda values, **_kwargs: values),
            patch.object(create, "_persist_job"),
        ):
            create._run_generate_job(
                job,
                create.LancelotAuth("token", "246273", "9500754"),
                pathlib.Path("cache"),
                False,
            )

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.failed_session_ids, ["bad"])
        self.assertEqual([result["status"] for result in job.results], ["error", "ok"])
        build.assert_called_once()
        self.assertEqual(build.call_args.args[1], "good")

    def test_second_range_job_is_durably_queued(self):
        job = self._job()
        job.job_id = "other-job"
        running_thread = Mock()
        running_thread.is_alive.return_value = True
        with create._jobs_lock:
            create._job_threads["test-job"] = running_thread

        with patch.object(create, "_persist_job") as persist:
            thread = create._start_job_thread(
                job,
                create.LancelotAuth("token", "246273", "9500754"),
                pathlib.Path("cache"),
            )

        self.assertIsNone(thread)
        self.assertEqual(job.status, "queued")
        self.assertIsNone(job.owner_id)
        persist.assert_called_once_with(job, pathlib.Path("cache"))

    def test_generate_aliases_reuse_same_canonical_player_job(self):
        job = self._job()
        job.player_id = "136662"
        job.player_license_number = "4958370"
        job.session_ids = ["300001"]
        job.progress = {"done": 0, "total": 1}
        resolved = create.ResolvedPlayer("136662", "4958370", "", "322582")
        identifiers = [
            "lancelot:136662",
            "classic:322582",
            "license:4958370",
        ]

        with (
            patch.object(create, "initialize_generate_jobs"),
            patch.object(
                create,
                "ensure_lancelot_auth",
                return_value=create.LancelotAuth("token", "246273", "9500754"),
            ),
            patch.object(create, "resolve_player", return_value=resolved),
            patch.object(create, "_active_job_for_player", return_value=job),
        ):
            results = [
                create.generate_postmortems(
                    identifier,
                    date_from="2025-01-01",
                    date_to="2026-12-31",
                )
                for identifier in identifiers
            ]

        self.assertEqual({result["player_id"] for result in results}, {"136662"})
        self.assertEqual({result["job_id"] for result in results}, {"test-job"})
        self.assertTrue(all(result["reused_job"] for result in results))


class OtherPlayerSessionTests(unittest.TestCase):
    def test_shared_index_lists_only_target_player_in_date_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = pathlib.Path(tmp) / "player_session_index"
            results = pl.DataFrame(
                {
                    "tournament_id": ["300001", "300002", "300003"],
                    "tournament_name": ["One", "Two", "Three"],
                    "date": ["2025-02-01", "2024-12-31", "2025-03-01"],
                    "series_id": [1, 2, 3],
                    "team_id": ["11", "22", "33"],
                    "club_id": ["A", "B", "C"],
                    "club_name": ["Club A", "Club B", "Club C"],
                    "player1_name": ["Guy", "Guy", "Other"],
                    "player2_name": ["One", "Two", "Guy"],
                    "player1_lancelot_id": ["136662", "136662", "999"],
                    "player2_lancelot_id": ["111", "222", "136662"],
                    "player1_classic_person_id": ["322582", "322582", "888"],
                    "player2_classic_person_id": ["101", "202", "322582"],
                    "player1_license_number": ["4958370", "4958370", "999999"],
                    "player2_license_number": ["111111", "222222", "4958370"],
                }
            )
            create.mlBridgeFFIndexLib.build_and_write_index(
                results,
                index_dir=index_dir,
            )

            with patch.dict(
                create.os.environ,
                {"FFBRIDGE_PLAYER_SESSION_INDEX_DIR": str(index_dir)},
            ):
                sessions = create.fetch_other_player_source_sessions(
                    "136662",
                    date_from="2025-01-01",
                    date_to="2025-12-31",
                )

        self.assertEqual([row["session_id"] for row in sessions], ["300003", "300001"])
        self.assertEqual(
            sessions[0]["listing_source"],
            "shared Lancelot player-session index",
        )
        self.assertEqual(sessions[1]["club"], "Club A")

    def test_list_source_uses_shared_index_for_non_logged_in_player(self):
        auth = create.LancelotAuth("token", "246273", "9500754")
        resolved = create.ResolvedPlayer("136662", "4958370", "4958370", "322582")
        indexed = [
            {
                "session_id": "300001",
                "date": "2025-02-01",
                "club": "Club A",
            }
        ]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(create, "ensure_lancelot_auth", return_value=auth),
            patch.object(create, "resolve_player", return_value=resolved),
            patch.object(
                create,
                "fetch_other_player_source_sessions",
                return_value=indexed,
            ) as indexed_source,
            patch.object(create, "fetch_logged_in_source_sessions") as logged_in,
        ):
            result = create.list_source_sessions(
                "4958370",
                date_from="2025-01-01",
                date_to="2025-12-31",
                cache_dir=pathlib.Path(tmp),
            )

        self.assertEqual(result["player_id"], "136662")
        self.assertEqual(result["count"], 1)
        self.assertFalse(result["sessions"][0]["already_cached"])
        indexed_source.assert_called_once_with(
            "136662",
            date_from="2025-01-01",
            date_to="2025-12-31",
        )
        logged_in.assert_not_called()

    def test_list_source_accepts_all_indexed_identifier_namespaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = pathlib.Path(tmp) / "player_session_index"
            create.mlBridgeFFIndexLib.build_and_write_index(
                pl.DataFrame(
                    {
                        "tournament_id": ["300001"],
                        "tournament_name": ["One"],
                        "date": ["2025-02-01"],
                        "series_id": [1],
                        "team_id": ["11"],
                        "club_id": ["A"],
                        "club_name": ["Club A"],
                        "player1_name": ["Guy"],
                        "player2_name": ["Partner"],
                        "player1_lancelot_id": ["136662"],
                        "player2_lancelot_id": ["111"],
                        "player1_classic_person_id": ["322582"],
                        "player2_classic_person_id": ["101"],
                        "player1_license_number": ["4958370"],
                        "player2_license_number": ["111111"],
                    }
                ),
                index_dir=index_dir,
            )
            auth = create.LancelotAuth("token", "246273", "9500754")
            identifiers = [
                "lancelot:136662",
                "classic:322582",
                "license:4958370",
            ]
            with (
                patch.dict(
                    create.os.environ,
                    {"FFBRIDGE_PLAYER_SESSION_INDEX_DIR": str(index_dir)},
                ),
                patch.object(create, "ensure_lancelot_auth", return_value=auth),
            ):
                results = [
                    create.list_source_sessions(
                        identifier,
                        cache_dir=pathlib.Path(tmp) / "cache",
                    )
                    for identifier in identifiers
                ]

        self.assertEqual({result["player_id"] for result in results}, {"136662"})
        self.assertEqual({result["count"] for result in results}, {1})


if __name__ == "__main__":
    unittest.main()
