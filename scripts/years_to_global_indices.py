#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a TAROT-compatible global index file by selecting all time steps "
            "whose timestamps fall in the requested calendar years."
        )
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Calendar years to include, e.g. --years 2022 2021 2020 2019 2018 2017 2014 2011 2009.",
    )
    parser.add_argument(
        "--start-date",
        default="1979-01-01T00:00",
        help="Inclusive start of the full training time grid.",
    )
    parser.add_argument(
        "--end-date",
        default="2022-12-31T00:00",
        help="Exclusive end of the full training time grid.",
    )
    parser.add_argument(
        "--global-index-origin",
        default="1979-01-01",
        help="Origin used for global index numbering.",
    )
    parser.add_argument(
        "--step-hours",
        type=int,
        default=6,
        help="Temporal step size in hours.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npy file for global indices.",
    )
    parser.add_argument(
        "--output-timestamps-txt",
        default=None,
        help="Optional text file with ISO timestamps, one per line.",
    )
    return parser.parse_args()


def _year_from_ns(ts_ns: np.ndarray) -> np.ndarray:
    return ts_ns.astype("datetime64[ns]").astype("datetime64[Y]").astype(np.int64) + 1970


def main() -> None:
    args = _parse_args()
    if args.step_hours <= 0:
        raise ValueError("--step-hours must be > 0")

    years = np.array(sorted(set(args.years)), dtype=np.int64)
    start_ns = np.datetime64(args.start_date).astype("datetime64[ns]").astype(np.int64)
    end_ns = np.datetime64(args.end_date).astype("datetime64[ns]").astype(np.int64)
    origin_ns = np.datetime64(args.global_index_origin).astype("datetime64[ns]").astype(np.int64)
    step_ns = np.int64(args.step_hours) * np.int64(3600 * 10**9)

    if end_ns <= start_ns:
        raise ValueError("--end-date must be later than --start-date")
    if start_ns < origin_ns:
        raise ValueError("--start-date must be >= --global-index-origin")
    if (start_ns - origin_ns) % step_ns != 0:
        raise ValueError("--start-date is not aligned with --step-hours relative to --global-index-origin")
    if (end_ns - origin_ns) % step_ns != 0:
        raise ValueError("--end-date is not aligned with --step-hours relative to --global-index-origin")

    all_time_ns = np.arange(start_ns, end_ns, step_ns, dtype=np.int64)
    all_years = _year_from_ns(all_time_ns)
    mask = np.isin(all_years, years)
    selected_time_ns = all_time_ns[mask]
    selected_idx_global = ((selected_time_ns - origin_ns) // step_ns).astype(np.int64)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, selected_idx_global)

    if args.output_timestamps_txt is not None:
        out_txt = Path(args.output_timestamps_txt)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with out_txt.open("w", encoding="utf-8") as f:
            for ts in selected_time_ns.astype("datetime64[ns]"):
                f.write(f"{ts}\n")

    print(f"Selected years: {years.tolist()}")
    print(f"Full grid steps: {all_time_ns.size}")
    print(f"Selected steps: {selected_idx_global.size}")
    print(f"Saved global indices to {output}")
    if args.output_timestamps_txt is not None:
        print(f"Saved timestamps to {args.output_timestamps_txt}")
    print("Per-year counts:")
    for year in years:
        count = int(np.sum(all_years[mask] == year))
        print(f"  {year}: {count}")
    if selected_idx_global.size > 0:
        print(f"Global idx range: [{int(selected_idx_global.min())}, {int(selected_idx_global.max())}]")
        print(
            "Timestamp range: "
            f"{selected_time_ns.min().astype('datetime64[ns]')} .. {selected_time_ns.max().astype('datetime64[ns]')}"
        )


if __name__ == "__main__":
    main()
