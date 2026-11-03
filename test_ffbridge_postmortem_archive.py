import json
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
    def test_cast_to_reference_casts_only_columns_whose_type_differs(self):
        frame = pl.DataFrame(
            {
                "a": pl.Series([1, 2], dtype=pl.Int64),
                "b": [None, None],  # Null dtype, as an unused Bridge+ seat infers
                "c": ["x", "y"],
                "extra": [1.5, 2.5],
            }
        )
        reference = {"a": pl.Int16, "b": pl.String, "c": pl.String, "missing": pl.Int8}
        out = archive._cast_to_reference(frame, reference)
        self.assertEqual(out.columns, frame.columns)
        self.assertEqual(out.schema["a"], pl.Int16)
        self.assertEqual(out.schema["b"], pl.String)
        self.assertEqual(out.schema["c"], pl.String)
        self.assertEqual(out.schema["extra"], pl.Float64)
        self.assertEqual(out["a"].to_list(), [1, 2])
        # Nothing to cast returns the frame unchanged.
        self.assertIs(archive._cast_to_reference(out, reference), out)

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

    def test_missing_lead_is_stored_as_null_to_match_archive_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            with_lead = _session_frame().with_columns(
                pl.lit("HK").alias("Lead")
            )
            archive.archive_session(with_lead, "100", archive_dir=root)
            archived = archive.archive_session(
                _session_frame(), "101", archive_dir=root
            )
            stored = pl.read_parquet(archived["archive_file"])
            self.assertIn("Lead", stored.columns)
            self.assertEqual(stored["Lead"].null_count(), stored.height)
            self.assertEqual(archive.read_manifest(root).height, 2)

    def test_null_and_wider_dtypes_are_stored_as_the_reference_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            reference = _session_frame().with_columns(
                pl.lit("HK").alias("Lead"),
                pl.lit(579).cast(pl.Int64).alias("MP_Top"),
                pl.Series("Score", [140, -140], dtype=pl.Int16),
                pl.Series("Score_NS", [-140, 140], dtype=pl.Int16),
                pl.Series("Score_EW", [140, -140], dtype=pl.Int16),
                pl.concat_list(["Player_ID_N", "Player_ID_S"]).alias("Pair_IDs_NS"),
                pl.concat_list(["Player_Name_N", "Player_Name_S"]).alias("Pair_Names_NS"),
            )
            archive.archive_session(reference, "100", archive_dir=root)
            shifted = reference.with_columns(
                pl.lit(None).alias("Lead"),
                pl.lit(579.0).alias("MP_Top"),
                pl.col("Score").cast(pl.Int32),
                pl.col("Score_NS").cast(pl.Int32),
                pl.col("Score_EW").cast(pl.Int32),
                pl.lit(None).alias("Player_ID_N"),
                pl.lit(None).alias("Player_ID_S"),
                pl.lit(None).alias("Player_Name_N"),
                pl.lit(None).alias("Player_Name_S"),
            ).with_columns(
                pl.concat_list(["Player_ID_N", "Player_ID_S"]).alias("Pair_IDs_NS"),
                pl.concat_list(["Player_Name_N", "Player_Name_S"]).alias("Pair_Names_NS"),
            )
            archived = archive.archive_session(shifted, "101", archive_dir=root)
            stored = pl.read_parquet(archived["archive_file"])
            self.assertEqual(stored.schema["Lead"], pl.String)
            self.assertEqual(stored.schema["MP_Top"], pl.Int64)
            self.assertEqual(stored.schema["Score"], pl.Int16)
            self.assertEqual(stored.schema["Score_NS"], pl.Int16)
            self.assertEqual(stored.schema["Score_EW"], pl.Int16)
            self.assertEqual(stored.schema["Player_ID_N"], pl.String)
            self.assertEqual(stored.schema["Player_Name_S"], pl.String)
            self.assertEqual(stored.schema["Pair_IDs_NS"], pl.List(pl.String))
            self.assertEqual(stored.schema["Pair_Names_NS"], pl.List(pl.String))
            self.assertEqual(stored["MP_Top"][0], 579)
            self.assertEqual(archive.read_manifest(root).height, 2)

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
            self.assertIn("Board", loaded.columns)
            self.assertEqual(loaded["Board"].to_list(), [1, 1])

    def test_hierarchical_write_permits_additive_quality_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "archive"
            seed = pathlib.Path(tmp) / "seed"
            production = pathlib.Path(tmp) / "hierarchical"
            source = _session_frame()
            archive.archive_session(source, "100", archive_dir=root)
            normalized.build_normalized_subset(root, seed)
            normalized.initialize_hierarchical_layout(
                production, seed / "metadata.json"
            )
            augmented = source.with_columns(
                pl.lit(1).cast(pl.Int8).alias("Par_Contract_NS"),
                pl.lit(-1).cast(pl.Int8).alias("Par_Contract_EW"),
                pl.lit(False).alias("Is_Sacrifice_Opportunity"),
            )
            written = normalized.write_hierarchical_session(
                augmented,
                session_id="100",
                revision="additive",
                output_dir=production,
            )
            self.assertTrue(written["created"])
            report = normalized.normalized_player_report(
                production,
                session_id="100",
                player_ids=["10"],
                columns=["Board", "Contract", "Pct_NS"],
            )
            self.assertEqual(report.height, 2)

    def test_hierarchical_write_rejects_missing_validated_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "archive"
            seed = pathlib.Path(tmp) / "seed"
            production = pathlib.Path(tmp) / "hierarchical"
            source = _session_frame()
            archive.archive_session(source, "100", archive_dir=root)
            normalized.build_normalized_subset(root, seed)
            normalized.initialize_hierarchical_layout(
                production, seed / "metadata.json"
            )
            with self.assertRaisesRegex(ValueError, "Missing="):
                normalized.write_hierarchical_session(
                    source.drop("Pct_NS"),
                    session_id="101",
                    revision="missing",
                    output_dir=production,
                )

    def test_mp_dd_pct_declarer_is_forced_to_results(self):
        mapping = normalized._correct_column_mapping(
            {
                "MP_DD_Pct_Declarer": {
                    "table": "boards",
                    "storage_column": "MP_DD_Pct_Declarer",
                    "field": None,
                }
            }
        )
        self.assertEqual(mapping["MP_DD_Pct_Declarer"]["table"], "results")

    def test_varying_mp_dd_pct_declarer_does_not_fail_hierarchical_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            production = pathlib.Path(tmp)
            source = _session_frame().with_columns(
                pl.Series("MP_DD_Pct_Declarer", [0.6, 0.4])
            )
            constant = source.with_columns(pl.lit(0.5).alias("MP_DD_Pct_Declarer"))
            root = pathlib.Path(tmp) / "archive"
            seed = pathlib.Path(tmp) / "seed"
            archive.archive_session(constant, "100", archive_dir=root)
            normalized.build_normalized_subset(root, seed)
            metadata = json.loads((seed / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["column_mapping"]["MP_DD_Pct_Declarer"]["table"],
                "results",
            )
            normalized.initialize_hierarchical_layout(
                production, seed / "metadata.json"
            )
            written = normalized.write_hierarchical_session(
                source,
                session_id="101",
                revision="mixed-contracts",
                output_dir=production,
            )
            self.assertTrue(written["created"])
            report = normalized.normalized_player_report(
                production,
                session_id="101",
                player_ids=["10"],
                columns=["Board", "MP_DD_Pct_Declarer"],
            )
            self.assertEqual(sorted(report["MP_DD_Pct_Declarer"].to_list()), [0.4, 0.6])


def _lead_frame() -> pl.DataFrame:
    return _session_frame().with_columns(
        pl.Series("BidSuit", ["N", "S"]),
        pl.Series("PBN", ["N:AKQ.. E:... S:... W:..."] * 2),
        pl.Series("Lead", ["HK", None]),
        pl.Series("DD_Lead_Cost", [1, None], dtype=pl.Int64),
        pl.Series("Lead_Is_DD_Best", [False, None]),
    )


def _lead_source() -> pl.DataFrame:
    return _session_frame().with_columns(
        pl.Series("BidSuit", ["N", "S"]),
        pl.Series("PBN", ["N:AKQ.. E:... S:... W:..."] * 2),
    )


def _fixed_lead_scorer(row, inputs: pl.DataFrame) -> pl.DataFrame:
    """Test scorer: a lead for North player 10 only, cost = row id."""
    return inputs.select(
        "_result_row_id",
        pl.when(pl.col("Player_ID_N") == "10")
        .then(pl.lit("SA"))
        .otherwise(None)
        .alias("Lead"),
        pl.col("_result_row_id").cast(pl.UInt8).alias("DD_Lead_Cost"),
        (pl.col("_result_row_id") == 0).alias("Lead_Is_DD_Best"),
    )


def _short_lead_scorer(row, inputs: pl.DataFrame) -> pl.DataFrame:
    return _fixed_lead_scorer(row, inputs).head(1)


def _previous_layout_archive(tmp: str) -> pathlib.Path:
    """One session archived on LAYOUT_VERSION - 1: no lead columns."""
    root = pathlib.Path(tmp) / "archive"
    seed = pathlib.Path(tmp) / "seed"
    production = pathlib.Path(tmp) / "hierarchical"
    archive.archive_session(_lead_source(), "100", archive_dir=root)
    metadata = normalized.build_normalized_subset(root, seed)
    # The flat archive adds a null Lead; the previous layout never mapped it.
    for column in normalized.OPENING_LEAD_COLUMNS:
        metadata["column_mapping"].pop(column, None)
    (seed / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    normalized.initialize_hierarchical_layout(production, seed / "metadata.json")
    written = normalized.write_hierarchical_session(
        _lead_source(), session_id="100", revision="r1", output_dir=production
    )
    old_tag = f"layout_version={normalized.LAYOUT_VERSION - 1}"
    new_tag = f"layout_version={normalized.LAYOUT_VERSION}"
    for key in ("boards_path", "results_path"):
        path = pathlib.Path(written[key])
        old_path = pathlib.Path(str(path).replace(new_tag, old_tag))
        old_path.parent.mkdir(parents=True, exist_ok=True)
        path.replace(old_path)
    manifest = normalized.read_hierarchical_manifest(production).with_columns(
        pl.col("boards_path", "results_path").str.replace(new_tag, old_tag, literal=True)
    )
    normalized._atomic_write_parquet(manifest, production / "manifest.parquet")
    metadata_path = production / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["layout_version"] = normalized.LAYOUT_VERSION - 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return production


class HierarchicalOpeningLeadTests(unittest.TestCase):
    def _production(self, tmp: str, source: pl.DataFrame) -> pathlib.Path:
        root = pathlib.Path(tmp) / "archive"
        seed = pathlib.Path(tmp) / "seed"
        production = pathlib.Path(tmp) / "hierarchical"
        archive.archive_session(source, "100", archive_dir=root)
        metadata = normalized.build_normalized_subset(root, seed)
        for column in normalized.OPENING_LEAD_COLUMNS:
            if column in source.columns:
                self.assertEqual(
                    metadata["column_mapping"][column],
                    {"table": "results", "storage_column": column, "field": None},
                )
        normalized.initialize_hierarchical_layout(production, seed / "metadata.json")
        return production

    def test_lead_columns_are_stored_typed_and_sd_set_null_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            sd = {
                column: pl.lit(None, dtype=dtype)
                for optional in normalized.OPTIONAL_RESULT_SETS
                for column, dtype in optional.items()
            }
            production = self._production(tmp, _lead_frame().with_columns(**sd))
            written = normalized.write_hierarchical_session(
                _lead_frame(),
                session_id="101",
                revision="no-sd",
                output_dir=production,
            )
            self.assertTrue(written["created"])
            stored = pl.read_parquet(written["results_path"])
            for column, dtype in normalized.OPENING_LEAD_COLUMNS.items():
                self.assertEqual(stored.schema[column], dtype, column)
            report = normalized.normalized_player_report(
                production,
                session_id="101",
                player_ids=["10"],
                columns=["Board", *normalized.OPENING_LEAD_COLUMNS],
            )
            self.assertEqual(report["Lead"].to_list(), ["HK", None])
            self.assertEqual(report["DD_Lead_Cost"].to_list(), [1, None])
            self.assertEqual(report["Lead_Is_DD_Best"].to_list(), [False, None])
            self.assertEqual(report["SD_Lead_Cost"].null_count(), 2)

            with self.assertRaisesRegex(ValueError, "part of the single-dummy"):
                normalized.write_hierarchical_session(
                    _lead_frame().with_columns(pl.lit(0.5).alias("SD_Lead_Cost")),
                    session_id="102",
                    revision="partial-sd",
                    output_dir=production,
                )
            with self.assertRaisesRegex(ValueError, "Missing=.*DD_Lead_Cost"):
                normalized.write_hierarchical_session(
                    _lead_frame().drop("DD_Lead_Cost"),
                    session_id="103",
                    revision="no-dd",
                    output_dir=production,
                )

    def test_postmortem_sql_selects_stored_dd_lead_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            production = self._production(tmp, _lead_frame())
            normalized.write_hierarchical_session(
                _lead_frame(),
                session_id="100",
                revision="r1",
                output_dir=production,
            )
            resolved = create.ResolvedPlayer(
                lancelot_id="10", license_number="10", requested_id="10"
            )
            with (
                patch.object(service, "HIERARCHICAL_DIR", production),
                patch.object(create, "resolve_player", return_value=resolved),
            ):
                frame, meta = service.load_postmortem("10", "100")
                result = service.run_sql(
                    frame,
                    "SELECT Board, Lead, DD_Lead_Cost FROM self ORDER BY DD_Lead_Cost",
                    meta,
                )
            self.assertEqual(meta["data_source"], "hierarchical_archive")
            self.assertEqual(
                sorted(
                    ((r["Lead"], r["DD_Lead_Cost"]) for r in result["rows"]),
                    key=str,
                ),
                sorted([("HK", 1), (None, None)], key=str),
            )

    def test_latest_postmortem_reads_newest_archived_game_unless_cache_is_newer(self):
        with tempfile.TemporaryDirectory() as tmp:
            production = self._production(tmp, _lead_frame())
            for session_id, played in (("100", date(2026, 8, 28)), ("101", date(2026, 9, 5))):
                normalized.write_hierarchical_session(
                    _lead_frame().with_columns(pl.lit(played).alias("Date")),
                    session_id=session_id,
                    revision="r1",
                    output_dir=production,
                )
            cache_dir = pathlib.Path(tmp) / "cache"
            cache_dir.mkdir()
            # A pre-upgrade cache file for an archived game must not win.
            _session_frame().write_parquet(cache_dir / "df-100-10.parquet")
            resolved = create.ResolvedPlayer(
                lancelot_id="10", license_number="10", requested_id="10"
            )
            with (
                patch.object(service, "HIERARCHICAL_DIR", production),
                patch.object(service, "CACHE_DIR", cache_dir),
                patch.object(archive, "DEFAULT_ARCHIVE_DIR", pathlib.Path(tmp) / "no-archive"),
                patch.object(create, "resolve_player", return_value=resolved),
            ):
                frame, meta = service.load_postmortem("10")
                self.assertEqual(meta["data_source"], "hierarchical_archive")
                self.assertEqual(meta["session_id"], "101")
                self.assertIn("DD_Lead_Cost", frame.columns)

                # A game played after the last archive run is only in the cache.
                _session_frame().with_columns(
                    pl.lit(date(2026, 9, 10)).alias("Date")
                ).write_parquet(cache_dir / "df-102-10.parquet")
                _frame, meta = service.load_postmortem("10")
                self.assertEqual(meta["data_source"], "cache")
                self.assertEqual(meta["session_id"], "102")


class HierarchicalMigrationTests(unittest.TestCase):
    def test_migration_adds_lead_columns_and_rejects_old_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            production = _previous_layout_archive(tmp)
            metadata_path = production / "metadata.json"
            old_tag = f"layout_version={normalized.LAYOUT_VERSION - 1}"
            with self.assertRaisesRegex(ValueError, "migrate"):
                normalized.normalized_player_report(
                    production, session_id="100", player_ids=["10"], columns=["Board"]
                )

            summary = normalized.migrate_hierarchical_layout(
                production, _fixed_lead_scorer, delete_old_fragments=True
            )

            self.assertEqual(summary["fragments"], 1)
            self.assertEqual(summary["leads"], 1)
            self.assertEqual(summary["dd_lead_scored"], 2)
            migrated = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(migrated["layout_version"], normalized.LAYOUT_VERSION)
            for column in normalized.OPENING_LEAD_COLUMNS:
                self.assertEqual(
                    migrated["column_mapping"][column],
                    {"table": "results", "storage_column": column, "field": None},
                )
            self.assertEqual(list(production.rglob(f"{old_tag}/**/*.parquet")), [])
            report = normalized.normalized_player_report(
                production,
                session_id="100",
                player_ids=["10"],
                columns=["Board", "Contract", *normalized.OPENING_LEAD_COLUMNS],
            )
            self.assertEqual(report["Contract"].to_list(), ["3NT", "4S"])
            self.assertEqual(report["Lead"].to_list(), ["SA", None])
            self.assertEqual(report["DD_Lead_Cost"].to_list(), [0, 1])
            self.assertEqual(report["Lead_Is_DD_Best"].to_list(), [True, False])
            self.assertEqual(report["SD_Lead_Samples"].null_count(), 2)
            info = normalized.hierarchical_info(production)
            self.assertEqual(info["layout_version"], normalized.LAYOUT_VERSION)
            self.assertEqual(info["missing_fragment_files"], 0)

            # The migrated metadata is the stage 4 seed: new sessions use it.
            normalized.initialize_hierarchical_layout(production, metadata_path)
            later = normalized.write_hierarchical_session(
                _lead_frame(), session_id="101", revision="r1", output_dir=production
            )
            self.assertTrue(later["created"])
            with self.assertRaisesRegex(ValueError, "layout version"):
                normalized.migrate_hierarchical_layout(production, _fixed_lead_scorer)

    def test_migration_with_workers_matches_sequential(self):
        with tempfile.TemporaryDirectory() as tmp:
            production = _previous_layout_archive(tmp)
            summary = normalized.migrate_hierarchical_layout(
                production, _fixed_lead_scorer, workers=2
            )
            self.assertEqual(summary["dd_lead_scored"], 2)
            report = normalized.normalized_player_report(
                production,
                session_id="100",
                player_ids=["10"],
                columns=["Board", "DD_Lead_Cost"],
            )
            self.assertEqual(report["DD_Lead_Cost"].to_list(), [0, 1])

    def test_migration_rejects_a_scorer_that_drops_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            production = _previous_layout_archive(tmp)
            with self.assertRaisesRegex(ValueError, "1 rows for 2 results"):
                normalized.migrate_hierarchical_layout(production, _short_lead_scorer)
            metadata = json.loads(
                (production / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["layout_version"], normalized.LAYOUT_VERSION - 1)


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
