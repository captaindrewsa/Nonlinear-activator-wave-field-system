#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_record(run_dir: Path, config_path: Path, result_path: Path) -> Dict[str, Any]:
    config = load_json(config_path)
    result = load_json(result_path)

    record: Dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "run_name": run_dir.name,
        "config_path": str(config_path.resolve()),
        "result_path": str(result_path.resolve()),
        "run_id": result.get("run_id") or config.get("run_id") or run_dir.name,
        "config": config,
        "result": result,
    }

    return record


def find_run_dirs(root_dir: Path) -> List[Path]:
    run_dirs: List[Path] = []

    for path in sorted(root_dir.rglob("config.json")):
        run_dir = path.parent
        result_path = run_dir / "result.json"
        if result_path.exists():
            run_dirs.append(run_dir)

    return run_dirs


def build_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_runs = len(records)
    completed_runs = sum(
        1 for record in records
        if record.get("result", {}).get("status") == "completed"
    )
    unstable_runs = sum(
        1 for record in records
        if record.get("result", {}).get("status") == "numerically_unstable"
    )

    failure_reasons: Dict[str, int] = {}
    for record in records:
        reason = record.get("result", {}).get("failure_reason")
        if reason:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return {
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "unstable_runs": unstable_runs,
        "failure_reasons": failure_reasons,
    }


def combine_project_runs(root_dir: Path) -> Dict[str, Any]:
    run_dirs = find_run_dirs(root_dir)
    records = []

    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        result_path = run_dir / "result.json"
        try:
            records.append(make_record(run_dir, config_path, result_path))
        except Exception as exc:
            records.append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "run_name": run_dir.name,
                    "error": repr(exc),
                }
            )

    payload = {
        "project_root": str(root_dir.resolve()),
        "n_records": len(records),
        "summary": build_summary([r for r in records if "result" in r]),
        "records": records,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine all config.json and result.json files from run subfolders into one JSON."
    )
    parser.add_argument(
        "root_dir",
        help="Root directory of the project/series, e.g. ./models/article2/series_a",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path. Default: <root_dir>/combined_results.json",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    output_path = (
        Path(args.output)
        if args.output is not None
        else root_dir / "combined_results.json"
    )

    combined = combine_project_runs(root_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, ensure_ascii=False)

    print(f"Saved combined JSON to: {output_path}")
    print(f"Found records: {combined['n_records']}")


if __name__ == "__main__":
    main()