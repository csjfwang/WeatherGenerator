#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np


def load_timestamps(csv_path: str) -> np.ndarray:
    arr = np.loadtxt(csv_path, dtype=str, delimiter=',')
    if arr.ndim == 0:
        arr = np.array([str(arr)])
    values = [str(x).strip() for x in arr.tolist() if str(x).strip()]
    if not values:
        raise ValueError(f'No timestamps found in {csv_path}')
    return np.array([v.replace(' ', 'T') for v in values], dtype='datetime64[ns]')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert a one-column timestamp CSV into TAROT/global index .npy file.'
    )
    parser.add_argument('--input-csv', required=True, help='CSV with one timestamp per line.')
    parser.add_argument('--output', required=True, help='Output .npy path for unique global indices.')
    parser.add_argument(
        '--origin',
        default='1979-01-01',
        help='Global index origin datetime, default: 1979-01-01',
    )
    parser.add_argument(
        '--step-hours',
        type=int,
        default=6,
        help='Step size in hours, default: 6',
    )
    args = parser.parse_args()

    if args.step_hours <= 0:
        raise ValueError('--step-hours must be > 0')

    timestamps = load_timestamps(args.input_csv)
    origin = np.datetime64(args.origin, 'ns')
    step_ns = np.int64(args.step_hours) * np.int64(3600 * 10**9)
    delta = timestamps.astype(np.int64) - origin.astype(np.int64)

    if np.any(delta < 0):
        raise ValueError('Some timestamps are earlier than --origin.')
    if np.any(delta % step_ns != 0):
        bad = timestamps[(delta % step_ns) != 0][:10]
        raise ValueError(
            'Some timestamps are not aligned to the requested step size. '
            f'First few misaligned values: {bad}'
        )

    global_idx = np.unique((delta // step_ns).astype(np.int64))
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, global_idx)

    print(f'Saved {len(global_idx)} global indices to {out_path}')
    print(f'Min timestamp: {timestamps.min()}')
    print(f'Max timestamp: {timestamps.max()}')
    print(f'Min global_idx: {global_idx.min()}')
    print(f'Max global_idx: {global_idx.max()}')


if __name__ == '__main__':
    main()
