import pathlib
import unittest
from unittest.mock import patch

import ffbridge_postmortem_create as create


class GenerateJobTests(unittest.TestCase):
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

        with patch.object(create, "list_source_sessions", side_effect=RuntimeError("API unavailable")):
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


if __name__ == "__main__":
    unittest.main()
