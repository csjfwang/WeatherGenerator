#!/usr/bin/env python3

import argparse
import glob
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert row indices into the raw concatenated candidate TAROT feature matrix "
            "back to idx_time_ns / idx_global / readable timestamps."
        )
    )
    parser.add_argument(
        "--row-indices",
        required=True,
        help="Path to .npy file containing row indices into the concatenated candidate feature bank.",
    )
    parser.add_argument(
        "--candidate-pattern",
        required=True,
        help="Glob for candidate TAROT feature files, in the exact order used to build the candidate matrix.",
    )
    parser.add_argument(
        "--output-time-ns",
        required=True,
        help="Output .npy path for idx_time_ns values.",
    )
    parser.add_argument(
        "--output-global-idx",
        required=True,
        help="Output .npy path for idx_global values.",
    )
    parser.add_argument(
        "--output-timestamps-txt",
        default=None,
        help="Optional text file with ISO timestamps, one per line.",
    )
    parser.add_argument(
        "--global-index-origin",
        default="1979-01-01",
        help="Origin used to convert absolute time to global time-step index.",
    )
    parser.add_argument(
        "--global-index-step-hours",
        type=int,
        default=6,
        help="Step size in hours for global time-step indexing.",
    )
    return parser.parse_args()


def _load_candidate_time_bank(pattern: str) -> np.ndarray:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No candidate feature files matched pattern: {pattern}")

    chunks = []
    for path in paths:
        with np.load(path) as data:
            if "idx_time_ns" not in data:
                raise KeyError(f"Missing 'idx_time_ns' in feature file: {path}")
            chunks.append(np.asarray(data["idx_time_ns"], dtype=np.int64).reshape(-1))
    return np.concatenate(chunks, axis=0)


def _time_ns_to_global_idx(idx_time_ns: np.ndarray, origin: str, step_hours: int) -> np.ndarray:
    origin_ns = np.datetime64(origin).astype("datetime64[ns]").astype(np.int64)
    step_ns = np.int64(step_hours) * np.int64(3600 * 10**9)
    delta = idx_time_ns.astype(np.int64) - origin_ns
    if np.any(delta < 0):
        raise ValueError("Found timestamps earlier than --global-index-origin.")
    if np.any(delta % step_ns != 0):
        raise ValueError("Found timestamps not aligned with --global-index-step-hours.")
    return (delta // step_ns).astype(np.int64)


def main() -> None:
    args = _parse_args()

    row_indices_path = Path(args.row_indices)
    row_indices = np.load(row_indices_path)
    row_indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)

    idx_time_bank = _load_candidate_time_bank(args.candidate_pattern)
    n_rows = idx_time_bank.shape[0]

    if row_indices.size == 0:
        raise ValueError("Input row-indices array is empty.")
    if np.any(row_indices < 0) or np.any(row_indices >= n_rows):
        raise ValueError(
            f"Row indices out of valid range [0, {n_rows}). "
            f"Observed min={int(row_indices.min())}, max={int(row_indices.max())}."
        )

    idx_time_ns = idx_time_bank[row_indices]
    idx_global = _time_ns_to_global_idx(
        idx_time_ns,
        origin=args.global_index_origin,
        step_hours=args.global_index_step_hours,
    )

    output_time_ns = Path(args.output_time_ns)
    output_global_idx = Path(args.output_global_idx)
    output_time_ns.parent.mkdir(parents=True, exist_ok=True)
    output_global_idx.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_time_ns, idx_time_ns)
    np.save(output_global_idx, idx_global)

    if args.output_timestamps_txt is not None:
        timestamps = idx_time_ns.astype("datetime64[ns]")
        out_txt = Path(args.output_timestamps_txt)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with out_txt.open("w", encoding="utf-8") as f:
            for ts in timestamps:
                f.write(f"{ts}\n")

    ts_min = str(idx_time_ns.min().astype("datetime64[ns]"))
    ts_max = str(idx_time_ns.max().astype("datetime64[ns]"))
    print(f"Loaded {row_indices.size} row indices from {row_indices_path}")
    print(f"Candidate matrix rows: {n_rows}")
    print(f"Saved idx_time_ns to {output_time_ns}")
    print(f"Saved idx_global to {output_global_idx}")
    if args.output_timestamps_txt is not None:
        print(f"Saved readable timestamps to {args.output_timestamps_txt}")
    print(f"Time range: {ts_min} .. {ts_max}")
    print(f"Global idx range: [{int(idx_global.min())}, {int(idx_global.max())}]")


if __name__ == "__main__":
    main()
