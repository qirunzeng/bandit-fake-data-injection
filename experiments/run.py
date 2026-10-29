#!/usr/bin/env python3
"""Build, validate, and run the corrected experiments using Python and a C++17 compiler."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(output, mode):
    with (output / "runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    sources = ["experiments/simulator.cpp", "experiments/prepare_data.py", "experiments/run.py", "experiments/report.py"]
    record = {
        "schema_version": 2,
        "repository": "qirunzeng/bandit-fake-data-injection",
        "upstream_repository": "richardhoff88/online_learning_torank",
        "upstream_commit": "dc2d3df799ec11d5d6111a5d8986c6d0dda41550",
        "mode": mode,
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
        "result_sha256": {name: sha256(output / name) for name in ("runs.csv", "trajectories.csv")},
        "data": json.loads((ROOT / "experiments/data/provenance.json").read_text()),
        "rows": len(rows),
        "configuration": {
            learner: {
                "trials": len({r["trial"] for r in rows if r["learner"] == learner}),
                "sigma": sorted({float(r["sigma"]) for r in rows if r["learner"] == learner}),
                "horizons": sorted({int(r["T"]) for r in rows if r["learner"] == learner}),
                "delta": sorted({float(r["delta"]) for r in rows if r["learner"] == learner}),
                "fake_lower_bounds": sorted({float(r["r_min"]) for r in rows if r["learner"] == learner}),
                "batch_sizes": sorted({int(r["batch"]) for r in rows if r["learner"] == learner}),
            } for learner in ("ucb", "ts")
        },
        "protocol": {
            "curves": "Cost and target ratio use the same endpoint runs at each varying horizon T." if {r["grid"] for r in rows} == {"horizon"} else "Legacy sweeps or parameter probe.",
            "clock": "Each genuine or injected sample consumes one of T records, including initialization.",
            "cost": "Sum of abs(fake_reward - true_arm_mean), for every processed fake record.",
            "ucb_rewards": "Bernoulli draws at the selected MovieLens column mean.",
            "ucb_index": "empirical_mean + 3*sigma*sqrt(log(next_record)/sample_count), sigma=0.5.",
            "ts_rewards": "Gaussian(mean, 0.05^2) throughout, including initialization.",
            "ts_posterior": "Independent Gaussian(empirical_mean, 1/sample_count).",
            "randomness": "SplitMix64 and Box-Muller; arm-wise reward streams and posterior streams are paired across methods.",
            "pbi": "Reject nonpositive signed gaps or cooldowns below one; record infeasible arms without excluding trials. Recertify delayed batches before execution.",
            "heuristic": "Randomly place SBI's realized per-arm fake budget after initialization; exactly match counts and true-mean cost.",
            "li": "Unbounded reference; not subject to the bounded methods' reward interval.",
        },
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Recorded source/data/result hashes for {len(rows)} runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=ROOT / "results/horizon")
    parser.add_argument("--quick", action="store_true", help="Small pilot; never used as the publication run")
    parser.add_argument("--lower-bound", type=float, default=-64, help="Shared fixed fake reward for SBI, PBI, and the matched heuristic")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=3141592653)
    parser.add_argument("--ratio-only", action="store_true", help="Parameter probe at T=100000")
    parser.add_argument("--sweeps", action="store_true", help="Run the archived horizon, trajectory, and sensitivity grids")
    parser.add_argument("--manifest-only", action="store_true", help="Record provenance for an already completed run")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.manifest_only:
        compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            raise SystemExit("A C++17 compiler is required (clang++ or g++).")
        subprocess.run([sys.executable, str(ROOT / "experiments/prepare_data.py")], check=True)
        build = ROOT / "build"
        build.mkdir(exist_ok=True)
        executable = build / "simulator"
        subprocess.run([compiler, "-std=c++17", "-O3", "-Wall", "-Wextra", "-Wpedantic", "-pthread",
                        str(ROOT / "experiments/simulator.cpp"), "-o", str(executable)], check=True)
        subprocess.run([str(executable), "--self-test"], check=True)
        command = [str(executable), "--means", str(ROOT / "experiments/data/movielens-means.csv"),
                   "--threads", str(args.threads), "--output", str(output),
                   "--lower-bound", str(args.lower_bound), "--batch-size", str(args.batch_size),
                   "--seed-base", str(args.seed_base)]
        if args.sweeps:
            command.append("--sweeps")
        if args.quick:
            command.append("--quick")
        if args.ratio_only:
            command.append("--ratio-only")
        subprocess.run(command, check=True)
    manifest(output, "probe" if args.ratio_only else "pilot" if args.quick else "full")


if __name__ == "__main__":
    main()
