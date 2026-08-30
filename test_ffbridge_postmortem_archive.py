import os
import pathlib
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

import polars as pl

import ffbridge_postmortem_archive as archive
import ffbridge_postmortem_create as create
import ffbridge_postmortem_normalized as normalized
import ffbridge_postmortem_service as service


def _session_frame(pair_direction: str = "NS", contract: str = "3NT") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [date(2026, 8, 28), date(2026, 8, 28)],
            "Pair_Direction": [pair_direction, pair_direction],
            "Board": [1, 1],
            "Contract": [contract, "4S"],
            "Declarer_Direction": ["N", "E"],
            "Pct_NS": [0.6, 0.4],
            "Player_ID_N": ["10", "20"],
            "Player_ID_S": ["11", "21"],
            "Player_ID_E": ["20", "10"],
            "Player_ID_W": ["21", "11"],
            "Player_Name_N": ["North", "East"],
            "Player_Name_S": ["South", "West"],
            "Player_Name_E": ["East", "North"],
            "Player_Name_W": ["West", "South"],
        }
    )


class ArchiveWriteTests(unittest.TestCase):
    def test_player_perspective_does_not_create_duplicate_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            first = archive.archive_session(
                _session_frame("NS"),
                "100",
                archive_dir=root,
                context={"series_id": "series-a"},
            )
            second = archive.archive_session(
                _session_frame("EW"),
                "100",
                archive_dir=root,
                context={"series_id": "series-a"},
            )
            self.assertTrue(first["created"])
            self.assertFalse(second["created"])
            self.assertEqual(first["revision"], second["revision"])
            self.assertEqual(archive.read_manifest(root).height, 1)
            stored = pl.read_parquet(first["archive_file"])
            self.assertTrue(stored["Pair_Direction"].is_null().all())

    def test_changed_content_creates_explicit_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            first = archive.archive_session(_session_frame(), "100", archive_dir=root)
            second = archive.archive_session(
                _session_frame(contract="2H"), "100", archive_dir=root
            )
            self.assertNotEqual(first["revision"], second["revision"])
            self.assertEqual(archive.read_manifest(root).height, 2)
            self.assertEqual(archive.latest_manifest(root).height, 1)

    def test_incompatible_schema_fails_before_manifest_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            archive.archive_session(_session_frame(), "100", archive_dir=root)
            incompatible = _session_frame().drop("Pct_NS")
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                archive.archive_session(incompatible, "101", archive_dir=root)
            self.assertEqual(archive.read_manifest(root).height, 1)

    def test_indexes_and_compaction_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            archive.archive_session(
                _session_frame(),
                "100",
                archive_dir=root,
                context={"series_id": "series-a", "organization_name": "Club"},
            )
            counts = archive.rebuild_indexes(root)
            self.assertEqual(counts["sessions"], 1)
            self.assertGreaterEqual(counts["player_sessions"], 4)
            first = archive.compact_archive(root)
            second = archive.compact_archive(root)
            self.assertEqual(first["partitions"], 1)
            self.assertEqual(second["partitions"], 0)
            files = archive.dataset_files(root)
            self.assertEqual(len(files), 1)
            compacted = pl.read_parquet(files[0])
            self.assertEqual(compacted.height, 2)
            self.assertIn("session_id", compacted.columns)


class ArchiveReadTests(unittest.TestCase):
    def test_archive_is_personalized_and_preferred_to_legacy_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "archive"
            cache = pathlib.Path(tmp) / "cache"
            cache.mkdir()
            archive.archive_session(_session_frame(), "100", archive_dir=root)
            archive.rebuild_indexes(root)
            legacy = _session_frame(contract="1C")
            legacy.write_parquet(cache / "df-100-10.parquet")
            resolved = create.ResolvedPlayer(
                lancelot_id="10",
                license_number="10",
                requested_id="10",
            )
            with (
                patch.object(archive, "DEFAULT_ARCHIVE_DIR", root),
                patch.object(service, "CACHE_DIR", cache),
                patch.object(create, "resolve_player", return_value=resolved),
            ):
                frame, meta = service.load_postmortem("10", "100")
            self.assertEqual(meta["data_source"], "archive")
            self.assertEqual(frame["Pair_Direction"].unique().to_list(), ["NS"])
            self.assertEqual(frame["Contract"][0], "3NT")

    def test_filtered_archive_rows_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            archive.archive_session(
                _session_frame(),
                "100",
                archive_dir=root,
                context={"series_id": "series-a"},
            )
            archive.rebuild_indexes(root)
            archive.compact_archive(root)
            with patch.object(archive, "DEFAULT_ARCHIVE_DIR", root):
                result = service.archive_rows(
                    session_id="100",
                    columns=["Board", "Contract"],
                    limit=1,
                )
            self.assertEqual(result["row_count"], 1)
            self.assertTrue(result["truncated"])
            self.assertEqual(result["columns"], ["Board", "Contract"])


class NormalizedArchiveTests(unittest.TestCase):
    def test_normalized_report_preserves_flat_report_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "archive"
            output = pathlib.Path(tmp) / "normalized"
            source = _session_frame()
            archive.archive_session(source, "100", archive_dir=root)
            metadata = normalized.build_normalized_subset(root, output)
            actual = normalized.normalized_player_report(
                output,
                session_id="100",
                player_ids=["10"],
                columns=["Board", "Contract", "Pct_NS"],
            )
            expected = source.select("Board", "Contract", "Pct_NS")
            self.assertTrue(expected.equals(actual, null_equal=True))
            self.assertLess(metadata["board_rows"], metadata["result_rows"])
            self.assertLess(
                metadata["results_storage_columns"],
                metadata["result_logical_columns"],
            )

    def test_production_hierarchy_is_incremental_and_report_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "archive"
            seed = pathlib.Path(tmp) / "seed"
            production = pathlib.Path(tmp) / "hierarchical"
            source = _session_frame()
            archived = archive.archive_session(source, "100", archive_dir=root)
            normalized.build_normalized_subset(root, seed)
            normalized.initialize_hierarchical_layout(
                production, seed / "metadata.json"
            )
            written = normalized.write_hierarchical_session(
                source,
                session_id="100",
                revision=archived["revision"],
                output_dir=production,
                series_id="series-a",
            )
            second = normalized.write_hierarchical_session(
                source,
                session_id="100",
                revision=archived["revision"],
                output_dir=production,
                series_id="series-a",
            )
            self.assertTrue(written["created"])
            self.assertFalse(second["created"])
            compacted = normalized.compact_hierarchical_archive(production)
            self.assertEqual(compacted["partitions"], 1)

            resolved = create.ResolvedPlayer(
                lancelot_id="10",
                license_number="10",
                requested_id="10",
            )
            with (
                patch.object(service, "HIERARCHICAL_DIR", production),
                patch.object(create, "resolve_player", return_value=resolved),
            ):
                report = service.hierarchical_board_results(
                    "10",
                    "100",
                    columns=["Board", "Contract", "Pct_NS"],
                )
                expected_frame, expected_meta = service.personalize(source, "10")
                loaded, meta = service.load_postmortem("10", "100")
            expected = pl.DataFrame(
                service.board_results(
                    expected_frame,
                    expected_meta,
                    columns=["Board", "Contract", "Pct_NS"],
                )["rows"]
            )
            self.assertTrue(
                expected.equals(pl.DataFrame(report["rows"]), null_equal=True)
            )
            self.assertEqual(
                report["meta"]["data_source"], "hierarchical_archive"
            )
            self.assertEqual(meta["data_source"], "hierarchical_archive")
            self.assertGreater(loaded.height, 0)


class HierarchicalResolveTests(unittest.TestCase):
    def test_env_wins_over_published_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = pathlib.Path(tmp) / "explicit"
            published = pathlib.Path(tmp) / "published"
            published.mkdir()
            (published / "metadata.json").write_text("{}", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "FFBRIDGE_POSTMORTEM_HIERARCHICAL_DIR": str(explicit),
                    "FFBRIDGE_CACHE_DIR": str(published.parent),
                },
                clear=False,
            ):
                self.assertEqual(
                    normalized.resolve_hierarchical_dir(pathlib.Path(tmp)),
                    explicit,
                )

    def test_published_cache_dir_is_used_when_env_is_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = pathlib.Path(tmp) / "ffbridge"
            published = cache_root / "postmortem_archive_hierarchical"
            published.mkdir(parents=True)
            (published / "metadata.json").write_text("{}", encoding="utf-8")
            env = {
                key: value
                for key, value in os.environ.items()
                if key
                not in {
                    "FFBRIDGE_POSTMORTEM_HIERARCHICAL_DIR",
                    "FFBRIDGE_CACHE_DIR",
                }
            }
            env["FFBRIDGE_CACHE_DIR"] = str(cache_root)
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(
                    normalized.resolve_hierarchical_dir(pathlib.Path(tmp)),
                    published,
                )


if __name__ == "__main__":
    unittest.main()
