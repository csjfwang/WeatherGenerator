#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np


def load_unique_indices(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    return np.unique(arr)


def save_timestamps(indices: np.ndarray, path: str, origin: str, step_hours: int) -> None:
    origin_dt = np.datetime64(origin, 'ns')
    times = origin_dt + indices.astype(np.int64) * np.timedelta64(step_hours, 'h')
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for t in times:
            f.write(str(t).replace('T', ' ') + '\\n')
            

def main() -> None:
    parser = argparse.ArgumentParser(description='Compare overlap between two global-index .npy files.')
    parser.add_argument('--a', required=True, help='First .npy file of global indices.')
    parser.add_argument('--b', required=True, help='Second .npy file of global indices.')
    parser.add_argument('--label-a', default='A', help='Label for first set.')
    parser.add_argument('--label-b', default='B', help='Label for second set.')
    parser.add_argument('--save-overlap', default=None, help='Optional output .npy path for overlap indices.')
    parser.add_argument('--save-overlap-timestamps', default=None, help='Optional output text path for overlap timestamps.')
    parser.add_argument('--origin', default='1979-01-01', help='Origin datetime for timestamp export.')
    parser.add_argument('--step-hours', type=int, default=6, help='Step size in hours for timestamp export.')
    args = parser.parse_args()

    if args.step_hours <= 0:
        raise ValueError('--step-hours must be > 0')

    a = load_unique_indices(args.a)
    b = load_unique_indices(args.b)
    overlap = np.intersect1d(a, b)
    union = np.union1d(a, b)
    only_a = np.setdiff1d(a, b)
    only_b = np.setdiff1d(b, a)

    print(f'{args.label_a} size: {len(a)}')
    print(f'{args.label_b} size: {len(b)}')
    print(f'overlap size: {len(overlap)}')
    print(f'overlap / {args.label_a}: {len(overlap) / len(a):.6f}' if len(a) else f'overlap / {args.label_a}: nan')
    print(f'overlap / {args.label_b}: {len(overlap) / len(b):.6f}' if len(b) else f'overlap / {args.label_b}: nan')
    print(f'jaccard: {len(overlap) / len(union):.6f}' if len(union) else 'jaccard: nan')
    print(f'only {args.label_a}: {len(only_a)}')
    print(f'only {args.label_b}: {len(only_b)}')

    if args.save_overlap:
        out = pathlib.Path(args.save_overlap)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, overlap)
        print(f'Saved overlap indices to {out}')

    if args.save_overlap_timestamps:
        save_timestamps(overlap, args.save_overlap_timestamps, args.origin, args.step_hours)
        print(f'Saved overlap timestamps to {args.save_overlap_timestamps}')


if __name__ == '__main__':
    main()
