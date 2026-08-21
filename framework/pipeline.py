#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
import time
from pathlib import Path


def find_configs(cfg_dir: Path):
    return sorted([p for p in cfg_dir.glob("*.json") if p.is_file()])


def run_one(sim_path: Path, cfg_path: Path, device: str):
    cmd = [
        sys.executable,
        str(sim_path),
        "--config",
        str(cfg_path),
        "--device",
        device,
    ]

    print("=" * 90)
    print(f"RUN: {cfg_path.name}")
    print("CMD:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    print("=" * 90)

    t0 = time.time()
    try:
        result = subprocess.run(cmd)
        rc = result.returncode
    except KeyboardInterrupt:
        raise
    except Exception as e:
        dt = time.time() - t0
        return {
            "config": cfg_path.name,
            "ok": False,
            "returncode": -999,
            "elapsed_sec": dt,
            "error": str(e),
        }

    dt = time.time() - t0
    return {
        "config": cfg_path.name,
        "ok": (rc == 0),
        "returncode": rc,
        "elapsed_sec": dt,
        "error": None,
    }


def print_summary(results):
    print("\n" + "#" * 90)
    print("BATCH SUMMARY")
    print("#" * 90)

    ok_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - ok_count

    for i, r in enumerate(results, start=1):
        status = "OK" if r["ok"] else "FAIL"
        print(
            f"{i:02d}. {status:4s} | "
            f"{r['config']:<40s} | "
            f"rc={r['returncode']:<4d} | "
            f"time={r['elapsed_sec']:.1f}s"
        )
        if r["error"]:
            print(f"    error: {r['error']}")

    print("-" * 90)
    print(f"TOTAL : {len(results)}")
    print(f"OK    : {ok_count}")
    print(f"FAIL  : {fail_count}")
    print("#" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Sequential batch runner for sim_framework.py configs"
    )
    parser.add_argument(
        "sim_path",
        type=str,
        help="Path to sim_framework.py (or sim_framework_v1-3.py)",
    )
    parser.add_argument(
        "config_dir",
        type=str,
        help="Path to directory containing *.json configs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "directml"],
        help="Device passed through to sim_framework.py",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining configs even if one fails",
    )

    args = parser.parse_args()

    sim_path = Path(args.sim_path).resolve()
    config_dir = Path(args.config_dir).resolve()

    if not sim_path.exists():
        print(f"ERROR: simulator file not found: {sim_path}")
        sys.exit(1)

    if not config_dir.exists() or not config_dir.is_dir():
        print(f"ERROR: config dir not found or not a directory: {config_dir}")
        sys.exit(1)

    configs = find_configs(config_dir)
    if not configs:
        print(f"ERROR: no *.json configs found in {config_dir}")
        sys.exit(1)

    print(f"Simulator : {sim_path}")
    print(f"Config dir : {config_dir}")
    print(f"Device     : {args.device}")
    print(f"Configs    : {len(configs)}")
    print()

    for cfg in configs:
        print(f" - {cfg.name}")
    print()

    results = []

    for cfg in configs:
        res = run_one(sim_path, cfg, args.device)
        results.append(res)

        if not res["ok"] and not args.continue_on_error:
            print(f"\nStopping on first failure: {cfg.name}")
            break

    print_summary(results)

    if all(r["ok"] for r in results):
        sys.exit(0)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()