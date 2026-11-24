#!/usr/bin/env python3
"""Validate the seven nuPlan DB files and all base tokens used by interplan10."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "official_interplan_database_manifest.json"
INTERPLAN10 = (
    HERE
    / "upstream/interPlan/interplan/planning/script/config/common/scenario_filter/interplan10.yaml"
)
REQUIRED_TABLES = {
    "log",
    "lidar_pc",
    "ego_pose",
    "lidar_box",
    "track",
    "scenario_tag",
    "traffic_light_status",
}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    destination = Path(manifest["destination"])
    expected = [destination / name for name in manifest["database_files"]]
    missing_files = [str(path) for path in expected if not path.is_file()]

    config = yaml.safe_load(INTERPLAN10.read_text())
    modified_tokens = config["scenario_tokens"]
    base_tokens = sorted({token.rsplit("-", 1)[0] for token in modified_tokens})
    found_tokens = set()
    database_checks = {}

    for path in expected:
        if not path.is_file():
            continue
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        available = {
            row[0]
            for row in connection.execute(
                "SELECT lower(hex(token)) FROM lidar_pc "
                f"WHERE lower(hex(token)) IN ({','.join('?' for _ in base_tokens)})",
                base_tokens,
            )
        }
        found_tokens.update(available)
        database_checks[path.name] = {
            "size_bytes": path.stat().st_size,
            "sqlite_quick_check": quick_check,
            "required_tables_present": REQUIRED_TABLES.issubset(tables),
            "interplan10_base_tokens": sorted(available),
        }
        connection.close()

    missing_tokens = sorted(set(base_tokens) - found_tokens)
    passed = (
        not missing_files
        and not missing_tokens
        and all(
            check["sqlite_quick_check"] == "ok" and check["required_tables_present"]
            for check in database_checks.values()
        )
    )
    result = {
        "passed": passed,
        "destination": str(destination),
        "expected_database_count": len(expected),
        "present_database_count": len(expected) - len(missing_files),
        "interplan10_modified_scenario_count": len(modified_tokens),
        "required_base_tokens": base_tokens,
        "missing_files": missing_files,
        "missing_base_tokens": missing_tokens,
        "database_checks": database_checks,
    }
    output = HERE / "official_interplan_data_check.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
