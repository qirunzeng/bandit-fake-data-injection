#!/usr/bin/env python3
"""Check that scheduling worker threads does not change any recorded results."""
import hashlib
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="bandit-reproducibility-") as temporary:
        outputs = []
        for threads in (1, 3):
            output = Path(temporary) / str(threads)
            subprocess.run([str(ROOT / "build/simulator"), "--quick", "--threads", str(threads),
                            "--means", str(ROOT / "experiments/data/movielens-means.csv"),
                            "--output", str(output)], check=True, capture_output=True)
            outputs.append(output)
        for name in ("runs.csv", "trajectories.csv"):
            digests = [hashlib.sha256((output / name).read_bytes()).hexdigest() for output in outputs]
            if len(set(digests)) != 1:
                raise AssertionError(f"Results depend on worker scheduling: {name}")
    print("PASS: single-worker and three-worker results are byte-identical")


if __name__ == "__main__":
    main()
