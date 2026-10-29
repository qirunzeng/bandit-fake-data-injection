#!/usr/bin/env python3
"""Extract Bernoulli arm means from the pinned upstream MovieLens matrix."""
import argparse
import array
import ast
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "richardhoff88/online_learning_torank"
COMMIT = "dc2d3df799ec11d5d6111a5d8986c6d0dda41550"
BLOB = "2ea89402667bebf82affdd724d854fc081cba877"


def prepare(source, target):
    raw = source.read_bytes()
    digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if digest != BLOB:
        raise ValueError("The input is not the pinned upstream MovieLens matrix")
    if raw[:8] != b"\x93NUMPY\x01\x00":
        raise ValueError("Expected NumPy v1 format")
    size = struct.unpack("<H", raw[8:10])[0]
    header = ast.literal_eval(raw[10:10 + size].decode().strip())
    if header["descr"] != "<f8" or header["fortran_order"]:
        raise ValueError("Expected a row-major little-endian float64 matrix")
    users, movies = header["shape"]
    values = array.array("d")
    values.frombytes(raw[10 + size:])
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != users * movies or any(x not in (0, 1) for x in values):
        raise ValueError("Expected a binary user-item matrix")
    means = [sum(values[i * movies + j] for i in range(users)) / users for j in range(movies)]
    target.mkdir(parents=True, exist_ok=True)
    with (target / "movielens-means.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("column", "interaction_probability"))
        writer.writerows(enumerate(means))
    metadata = {
        "repository": UPSTREAM, "commit": COMMIT,
        "path": "dataset/movielens.npy", "git_blob_sha1": digest,
        "sha256": hashlib.sha256(raw).hexdigest(), "shape": list(header["shape"]),
        "extraction": "Column means of the binary matrix; genuine rewards are sampled as Bernoulli(p).",
    }
    (target / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Extracted {movies} movie probabilities from {users} users")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/movielens.npy")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/data")
    args = parser.parse_args()
    prepare(args.source, args.output)
