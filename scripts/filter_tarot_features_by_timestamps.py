#!/usr/bin/env python3
import argparse
import glob
import pathlib

import numpy as np


def load_csv_timestamps(path: str) -> np.ndarray:
    arr = np.loadtxt(path, dtype=str, delimiter=',')
    if arr.ndim == 0:
        arr = np.array([str(arr)])
    arr = np.array([s.strip().replace(" ", "T") for s in arr if str(s).strip()], dtype="datetime64[ns]")
    return arr.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter TAROT feature files by an explicit timestamp list.")
    parser.add_argument('--input', required=True, help='Glob for input npz files.')
    parser.add_argument('--timestamps-csv', required=True, help='CSV with one timestamp per line.')
    parser.add_argument('--output', required=True, help='Output filtered npz path.')
    parser.add_argument('--global-index-origin', default='1979-01-01', help='Origin for optional idx_global computation.')
    parser.add_argument('--global-index-step-hours', type=int, default=6, help='Step hours for optional idx_global computation.')
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input))
    if not paths:
        raise FileNotFoundError(f'No input files matched: {args.input}')

    wanted_ns = load_csv_timestamps(args.timestamps_csv)
    wanted_set = set(wanted_ns.tolist())

    idx_all = []
    idx_time_all = []
    feat_all = []
    source_path_all = []
    source_row_all = []

    for path in paths:
        with np.load(path) as data:
            if 'idx' not in data or 'idx_time_ns' not in data or 'feat_grad_projected' not in data:
                raise KeyError(f'Missing required keys in {path}; found {data.files}')
            idx = np.asarray(data['idx'], dtype=np.int64).reshape(-1)
            idx_time_ns = np.asarray(data['idx_time_ns'], dtype=np.int64).reshape(-1)
            feat = np.asarray(data['feat_grad_projected'], dtype=np.float32)
            mask = np.isin(idx_time_ns, wanted_ns)
            if not np.any(mask):
                continue
            rows = np.nonzero(mask)[0].astype(np.int64)
            idx_all.append(idx[mask])
            idx_time_all.append(idx_time_ns[mask])
            feat_all.append(feat[mask])
            source_path_all.append(np.full(rows.shape[0], path, dtype=object))
            source_row_all.append(rows)

    if not idx_all:
        raise ValueError('No timestamps from CSV were found in the input feature files.')

    idx = np.concatenate(idx_all)
    idx_time_ns = np.concatenate(idx_time_all)
    feat = np.concatenate(feat_all)
    source_path = np.concatenate(source_path_all)
    source_row = np.concatenate(source_row_all)

    order = np.argsort(idx_time_ns, kind='stable')
    idx = idx[order]
    idx_time_ns = idx_time_ns[order]
    feat = feat[order]
    source_path = source_path[order]
    source_row = source_row[order]

    unique_time_ns, unique_pos = np.unique(idx_time_ns, return_index=True)
    if unique_time_ns.shape[0] != idx_time_ns.shape[0]:
        idx = idx[unique_pos]
        idx_time_ns = idx_time_ns[unique_pos]
        feat = feat[unique_pos]
        source_path = source_path[unique_pos]
        source_row = source_row[unique_pos]

    missing = sorted(wanted_set - set(idx_time_ns.tolist()))
    if missing:
        missing_dt = np.asarray(missing, dtype='datetime64[ns]')
        raise ValueError(f'Missing {len(missing)} timestamps from CSV in the input feature files; first few: {missing_dt[:10]}')

    origin_ns = np.datetime64(args.global_index_origin).astype('datetime64[ns]').astype(np.int64)
    step_ns = np.int64(args.global_index_step_hours) * np.int64(3600 * 10**9)
    delta = idx_time_ns - origin_ns
    if np.any(delta < 0) or np.any(delta % step_ns != 0):
        raise ValueError('Timestamps are not aligned with the provided global index origin/step.')
    idx_global = (delta // step_ns).astype(np.int64)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        idx=idx,
        idx_time_ns=idx_time_ns,
        idx_global=idx_global,
        feat_grad_projected=feat,
        source_path=source_path,
        source_row=source_row,
    )
    print(f'Saved filtered target features to {out_path} with {idx.shape[0]} timestamps.')


if __name__ == '__main__':
    main()
