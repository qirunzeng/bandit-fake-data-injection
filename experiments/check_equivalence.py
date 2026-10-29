#!/usr/bin/env python3
"""Compare two simulator builds byte for byte on the same platform.

Use --baseline to supply a simulator compiled before a behavior-preserving change.
The candidate defaults to build/simulator. This check does not rewrite saved results.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "small-horizons": ["--quick"],
    "sensitivity-grids": ["--quick", "--sweeps"],
    "small-batches": ["--quick", "--sweeps", "--batch-size", "1", "--lower-bound", "-64"],
    "recorded-trajectories": ["--quick", "--ratio-only", "--lower-bound", "-36", "--seed-base", "2026102901"],
    "million-records": ["--benchmark"],
}


def compare(baseline, candidate):
    with tempfile.TemporaryDirectory(prefix="bandit-equivalence-") as temporary:
        temporary = Path(temporary)
        for case, arguments in CASES.items():
            outputs = []
            for label, executable, threads in (("baseline", baseline, 1), ("candidate", candidate, 3)):
                output = temporary / case / label
                subprocess.run([str(executable), "--threads", str(threads),
                                "--means", str(ROOT / "experiments/data/movielens-means.csv"),
                                "--output", str(output), *arguments], check=True, capture_output=True)
                outputs.append(output)
            for name in ("runs.csv", "trajectories.csv"):
                digests = [hashlib.sha256((output / name).read_bytes()).digest() for output in outputs]
                if digests[0] != digests[1]:
                    raise AssertionError(f"Result changed in {case}: {name}")
            print(f"PASS: {case}, both CSV files byte-identical", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=ROOT / "build/simulator")
    args = parser.parse_args()
    compare(args.baseline.resolve(), args.candidate.resolve())
