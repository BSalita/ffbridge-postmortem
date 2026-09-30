"""Migrate the FFBridge hierarchical archive from layout version 2 to 3.

Moves result-only columns (e.g. MP_DD_Pct_Declarer) from boards to results.
Stop the postmortem container first; the compacted dataset is rebuilt.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from datetime import datetime

import ffbridge_postmortem_normalized as normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hierarchical-dir",
        type=pathlib.Path,
        default=None,
        help="Archive directory (default: resolved like the service).",
    )
    parser.add_argument(
        "--delete-old-fragments",
        action="store_true",
        help="Delete the layout_version=2 fragment files after success.",
    )
    args = parser.parse_args()
    directory = args.hierarchical_dir or normalized.resolve_hierarchical_dir()
    if directory is None:
        raise FileNotFoundError("No hierarchical archive found; pass --hierarchical-dir")
    started = datetime.now()
    clock = time.perf_counter()
    print(f"Started {started:%Y-%m-%d %H:%M:%S}: migrating {directory}")
    summary = normalized.migrate_hierarchical_layout(
        directory, delete_old_fragments=args.delete_old_fragments
    )
    print(json.dumps(summary, indent=2, default=str))
    print(
        f"Ended {datetime.now():%Y-%m-%d %H:%M:%S} "
        f"(elapsed {time.perf_counter() - clock:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
