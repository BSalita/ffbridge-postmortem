import os
import pathlib
import tempfile
import unittest
from unittest import mock

import ffbridge_postmortem_archive as archive


class ReplaceRetryTest(unittest.TestCase):
    def test_retries_through_a_brief_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = pathlib.Path(tmp) / "a.tmp"
            destination = pathlib.Path(tmp) / "a"
            source.write_text("new")
            destination.write_text("old")
            real_replace = os.replace
            calls = []

            def flaky(src, dst):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError(5, "Access is denied")
                real_replace(src, dst)

            with mock.patch.object(archive.os, "replace", flaky), \
                    mock.patch.object(archive.time, "sleep"):
                archive._replace(source, destination)
            self.assertEqual(destination.read_text(), "new")
            self.assertEqual(len(calls), 3)

    def test_raises_when_lock_persists(self):
        with mock.patch.object(archive.os, "replace",
                               side_effect=PermissionError(5, "denied")), \
                mock.patch.object(archive.time, "sleep"), \
                mock.patch.object(archive, "REPLACE_RETRY_SECONDS", 0.0):
            with self.assertRaises(PermissionError):
                archive._replace(pathlib.Path("x"), pathlib.Path("y"))


if __name__ == "__main__":
    unittest.main()
