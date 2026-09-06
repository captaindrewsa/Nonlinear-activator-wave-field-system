#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    text = str(value).strip()
    if text == "":
        return None
    try:
        x = float(text)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def summarize_track(track_path: Path) -> Dict[str, Any]:
    with track_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    summary: Dict[str, Any] = {
        "path": str(track_path.resolve()),
        "n_rows": len(rows),
        "columns": fieldnames,
        "available": True,
    }
    if not rows:
        summary["empty"] = True
        return summary

    t_vals = [to_float(r.get("t")) for r in rows]
    v_maxabs = [to_float(r.get("v_maxabs")) for r in rows]
    v_rms = [to_float(r.get("v_rms")) for r in rows]
    psi_maxabs = [to_float(r.get("psi_maxabs")) for r in rows]
    phi_max = [to_float(r.get("phi_max")) for r in rows]
    mass = [to_float(r.get("mass")) for r in rows]
    total_energy = [to_float(r.get("total_energy")) for r in rows]
    spectral_peak_k = [to_float(r.get("spectral_peak_k")) for r in rows]
    spectral_energy_high = [to_float(r.get("spectral_energy_high")) for r in rows]

    def valid_pairs(ts: List[Optional[float]], xs: List[Optional[float]]):
        return [(t, x) for t, x in zip(ts, xs) if t is not None and x is not None]

    vpairs = valid_pairs(t_vals, v_maxabs)
    epairs = valid_pairs(t_vals, total_energy)
    ppairs = valid_pairs(t_vals, spectral_peak_k)
    hpairs = valid_pairs(t_vals, spectral_energy_high)

    vmax_peak_t = None
    vmax_peak = None
    if vpairs:
        vmax_peak_t, vmax_peak = max(vpairs, key=lambda z: z[1])

    energy_peak_t = None
    energy_peak = None
    if epairs:
        energy_peak_t, energy_peak = max(epairs, key=lambda z: z[1])

    first_t = next((x for x in t_vals if x is not None), None)
    last_t = next((x for x in reversed(t_vals) if x is not None), None)
    first_v = next((x for x in v_maxabs if x is not None), None)
    last_v = next((x for x in reversed(v_maxabs) if x is not None), None)
    first_e = next((x for x in total_energy if x is not None), None)
    last_e = next((x for x in reversed(total_energy) if x is not None), None)

    def max_growth_ratio(series: List[Optional[float]]) -> Optional[float]:
        prev = None
        best = None
        for cur in series:
            if cur is None:
                continue
            if prev is not None and prev != 0:
                ratio = cur / prev
                if math.isfinite(ratio):
                    best = ratio if best is None else max(best, ratio)
            prev = cur
        return best

    def onset_time(series: List[Optional[float]], times: List[Optional[float]], factor: float) -> Optional[float]:
        baseline = next((x for x in series if x is not None), None)
        if baseline is None or baseline <= 0:
            return None
        threshold = baseline * factor
        for t, x in zip(times, series):
            if t is not None and x is not None and x >= threshold:
                return t
        return None

    summary["t_window"] = {"t_first": first_t, "t_last": last_t}
    summary["v_diagnostics"] = {
        "initial_v_maxabs": first_v,
        "final_v_maxabs": last_v,
        "peak_v_maxabs": vmax_peak,
        "time_of_peak_v_maxabs": vmax_peak_t,
        "v_maxabs_growth_factor": safe_ratio(last_v, first_v),
        "max_stepwise_growth_ratio_v_maxabs": max_growth_ratio(v_maxabs),
        "t_v_x10": onset_time(v_maxabs, t_vals, 10.0),
        "t_v_x100": onset_time(v_maxabs, t_vals, 100.0),
        "final_v_rms": next((x for x in reversed(v_rms) if x is not None), None),
        "peak_v_rms": max((x for x in v_rms if x is not None), default=None),
        "final_psi_maxabs": next((x for x in reversed(psi_maxabs) if x is not None), None),
        "peak_psi_maxabs": max((x for x in psi_maxabs if x is not None), default=None),
        "final_phi_max": next((x for x in reversed(phi_max) if x is not None), None),
        "peak_phi_max": max((x for x in phi_max if x is not None), default=None),
    }
    summary["energy_diagnostics"] = {
        "initial_total_energy": first_e,
        "final_total_energy": last_e,
        "peak_total_energy": energy_peak,
        "time_of_peak_total_energy": energy_peak_t,
        "energy_growth_factor": safe_ratio(last_e, first_e),
        "max_stepwise_growth_ratio_total_energy": max_growth_ratio(total_energy),
        "t_energy_x10": onset_time(total_energy, t_vals, 10.0),
        "t_energy_x100": onset_time(total_energy, t_vals, 100.0),
    }
    summary["spectral_diagnostics"] = {
        "initial_spectral_peak_k": next((x for x in spectral_peak_k if x is not None), None),
        "final_spectral_peak_k": next((x for x in reversed(spectral_peak_k) if x is not None), None),
        "peak_spectral_peak_k": max((x for x in spectral_peak_k if x is not None), default=None),
        "initial_spectral_energy_high": next((x for x in spectral_energy_high if x is not None), None),
        "final_spectral_energy_high": next((x for x in reversed(spectral_energy_high) if x is not None), None),
        "peak_spectral_energy_high": max((x for x in spectral_energy_high if x is not None), default=None),
    }
    summary["mass_diagnostics"] = {
        "initial_mass": next((x for x in mass if x is not None), None),
        "final_mass": next((x for x in reversed(mass) if x is not None), None),
        "peak_mass": max((x for x in mass if x is not None), default=None),
    }

    return summary


def make_record(
    run_dir: Path,
    include_config: bool,
    include_result: bool,
    include_track: bool,
    track_required: bool,
) -> Dict[str, Any]:
    config_path = run_dir / "config.json"
    result_path = run_dir / "result.json"
    track_path = run_dir / "track.csv"

    config = load_json(config_path) if include_config else None
    result = load_json(result_path) if include_result else None
    run_id = None
    if result:
        run_id = result.get("run_id")
    if not run_id and config:
        run_id = config.get("run_id")
    if not run_id:
        run_id = run_dir.name

    record: Dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "run_name": run_dir.name,
        "config_path": str(config_path.resolve()) if config_path.exists() else None,
        "result_path": str(result_path.resolve()) if result_path.exists() else None,
        "track_path": str(track_path.resolve()) if track_path.exists() else None,
        "run_id": run_id,
    }

    if include_config and config is not None:
        record["config"] = config
    if include_result and result is not None:
        record["result"] = result

    if include_track:
        if track_path.exists():
            record["track_summary"] = summarize_track(track_path)
        elif track_required:
            raise FileNotFoundError(f"track.csv not found in {run_dir}")
        else:
            record["track_summary"] = {"available": False, "path": None}

    return record


def find_run_dirs(root_dir: Path, require_track: bool) -> List[Path]:
    run_dirs: List[Path] = []
    for path in sorted(root_dir.rglob("config.json")):
        run_dir = path.parent
        result_path = run_dir / "result.json"
        track_path = run_dir / "track.csv"
        if result_path.exists() and (not require_track or track_path.exists()):
            run_dirs.append(run_dir)
    return run_dirs


def build_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_runs = len(records)
    completed_runs = sum(1 for r in records if r.get("result", {}).get("status") == "completed")
    unstable_runs = sum(1 for r in records if r.get("result", {}).get("status") == "numerically_unstable")

    failure_reasons: Dict[str, int] = {}
    for record in records:
        reason = record.get("result", {}).get("failure_reason")
        if reason:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    track_present = sum(1 for r in records if r.get("track_summary", {}).get("available") is True)
    return {
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "unstable_runs": unstable_runs,
        "failure_reasons": failure_reasons,
        "track_summaries_present": track_present,
    }


def combine_project_runs(
    root_dir: Path,
    include_config: bool,
    include_result: bool,
    include_track: bool,
    track_required: bool,
) -> Dict[str, Any]:
    run_dirs = find_run_dirs(root_dir, require_track=False)
    records: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        try:
            records.append(make_record(run_dir, include_config, include_result, include_track, track_required))
        except Exception as exc:
            records.append({
                "run_dir": str(run_dir.resolve()),
                "run_name": run_dir.name,
                "error": repr(exc),
            })

    valid_records = [r for r in records if include_result and "result" in r]
    payload = {
        "project_root": str(root_dir.resolve()),
        "n_records": len(records),
        "options": {
            "include_config": include_config,
            "include_result": include_result,
            "include_track": include_track,
            "track_required": track_required,
        },
        "summary": build_summary(valid_records if valid_records else []),
        "records": records,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine config.json, result.json and summarized track.csv files from run subfolders into one JSON."
    )
    parser.add_argument("root_dir", help="Root directory of the project/series.")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path. Default: <root_dir>/combined_results.json")

    parser.add_argument("--config", dest="include_config", action="store_true", default=True,
                        help="Include config.json contents (default: on).")
    parser.add_argument("--no-config", dest="include_config", action="store_false",
                        help="Do not include config.json contents.")
    parser.add_argument("--result", dest="include_result", action="store_true", default=True,
                        help="Include result.json contents (default: on).")
    parser.add_argument("--no-result", dest="include_result", action="store_false",
                        help="Do not include result.json contents.")
    parser.add_argument("--track", dest="include_track", action="store_true", default=True,
                        help="Include summarized track.csv diagnostics (default: on).")
    parser.add_argument("--no-track", dest="include_track", action="store_false",
                        help="Do not include track.csv summary.")
    parser.add_argument("--require-track", action="store_true",
                        help="Treat missing track.csv as an error for a run.")

    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    output_path = Path(args.output) if args.output is not None else root_dir / "combined_results.json"
    combined = combine_project_runs(
        root_dir=root_dir,
        include_config=args.include_config,
        include_result=args.include_result,
        include_track=args.include_track,
        track_required=args.require_track,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, ensure_ascii=False)

    print(f"Saved combined JSON to: {output_path}")
    print(f"Found records: {combined['n_records']}")
    print(f"Options: {combined['options']}")


if __name__ == "__main__":
    main()
