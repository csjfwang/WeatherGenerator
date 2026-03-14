#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from weathergen.datasets.data_reader_base import str_to_datetime64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize temporal selection distributions (year/month/season) "
            "for TAROT/random/stratified/full baselines."
        )
    )
    parser.add_argument(
        "--config",
        default="config/default_config.yml",
        help="Config file containing start_date/end_date/step_hrs.",
    )
    parser.add_argument("--tarot", default=None, help="Path to TAROT selected indices (.npy/.npz).")
    parser.add_argument("--random", default=None, help="Path to random selected indices (.npy/.npz).")
    parser.add_argument(
        "--stratified", default=None, help="Path to stratified selected indices (.npy/.npz)."
    )
    parser.add_argument(
        "--output-dir",
        default="plots/selection_temporal",
        help="Directory for output figures and CSV summaries.",
    )
    parser.add_argument(
        "--rolling-window-days",
        type=int,
        default=90,
        help="Rolling window size (in days) for temporal density curve.",
    )
    parser.add_argument(
        "--val-anchor-year",
        type=int,
        default=None,
        help=(
            "Validation anchor year for |train_year - anchor| plots. "
            "If omitted, uses start_date_val year from config."
        ),
    )
    parser.add_argument(
        "--feature-pattern",
        default=None,
        help=(
            "Optional glob for exported feature npz files used for subset similarity "
            "analysis (e.g., '/path/to/tarot_features_chkpt*_rank*.npz')."
        ),
    )
    parser.add_argument(
        "--feature-key",
        default="feat_grad_projected",
        help="Feature key inside npz files (e.g., feat_grad_projected, feat_mean_std).",
    )
    parser.add_argument(
        "--feature-index-key",
        default="idx",
        choices=["idx", "idx_global", "idx_time_ns"],
        help="Index key in feature npz used to map features to subset indices.",
    )
    parser.add_argument(
        "--similarity-max-samples-per-method",
        type=int,
        default=20000,
        help="Max subset size per method used in similarity analysis (randomly downsampled).",
    )
    parser.add_argument(
        "--similarity-max-pairs",
        type=int,
        default=200000,
        help="Max number of random pairs for pairwise similarity distribution per method.",
    )
    return parser.parse_args()


def _load_indices(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"indices file does not exist: {p}")
    if p.suffix.lower() == ".npz":
        with np.load(p) as loaded:
            if "indices" in loaded:
                idx = np.asarray(loaded["indices"], dtype=np.int64).reshape(-1)
            elif len(loaded.files) == 1:
                idx = np.asarray(loaded[loaded.files[0]], dtype=np.int64).reshape(-1)
            else:
                raise ValueError(
                    f"For .npz index file {p}, provide key 'indices' or a single-array archive."
                )
    else:
        idx = np.asarray(np.load(p), dtype=np.int64).reshape(-1)
    return np.unique(idx)


def _validate_index_range(name: str, idx: np.ndarray, total_steps: int) -> None:
    if idx.size == 0:
        raise ValueError(f"{name}: index file is empty.")
    out_of_range = (idx < 0) | (idx >= total_steps)
    if np.any(out_of_range):
        bad_vals = idx[out_of_range]
        preview = np.unique(bad_vals)[:10]
        raise ValueError(
            f"{name}: found {bad_vals.size} out-of-range indices for [0, {total_steps}). "
            f"examples={preview.tolist()}"
        )


def _get_time_grid(cfg_path: str) -> tuple[pd.Timestamp, pd.Timestamp, int, np.ndarray, int]:
    cfg = OmegaConf.load(cfg_path)
    start = str_to_datetime64(cfg.start_date)
    end = str_to_datetime64(cfg.end_date)
    if cfg.get("start_date_val", None) is None:
        raise ValueError("Config must provide start_date_val for validation anchor year inference.")
    val_start = str_to_datetime64(cfg.start_date_val)
    val_anchor_year = pd.Timestamp(str(val_start)).year
    step_hours = int(cfg.step_hrs)
    total_steps = int((end - start) // np.timedelta64(step_hours, "h"))
    full_idx = np.arange(total_steps, dtype=np.int64)
    start_ts = pd.Timestamp(str(start))
    end_ts = pd.Timestamp(str(end))
    return start_ts, end_ts, step_hours, full_idx, int(val_anchor_year)


def _indices_to_time_df(name: str, idx: np.ndarray, start_ts: pd.Timestamp, step_hours: int) -> pd.DataFrame:
    ts = start_ts + pd.to_timedelta(idx * step_hours, unit="h")
    df = pd.DataFrame({"method": name, "idx": idx, "time": ts})
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    month = df["month"]
    df["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        ["DJF", "MAM", "JJA", "SON"],
        default="UNK",
    )
    return df


def _normalize_rows(count_df: pd.DataFrame) -> pd.DataFrame:
    share = count_df.copy()
    totals = share.sum(axis=1).replace(0, np.nan)
    share = share.div(totals, axis=0).fillna(0.0)
    return share


def _plot_grouped_bar(df: pd.DataFrame, title: str, ylabel: str, out_path: Path) -> None:
    ax = df.T.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _compute_year_month_lift(
    all_df: pd.DataFrame,
    method_order: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    year_month_counts = (
        all_df.groupby(["method", "month", "year"]).size().unstack(fill_value=0).sort_index(axis=1)
    )
    year_month_counts = year_month_counts.reindex(columns=sorted(year_month_counts.columns), fill_value=0)

    grids: dict[str, pd.DataFrame] = {}
    for method in method_order:
        method_df = year_month_counts.loc[method].copy()
        method_df = method_df.reindex(index=range(1, 13), fill_value=0)
        grids[method] = method_df

    full_grid = grids["full"].astype(np.float64)
    full_share = full_grid / max(float(full_grid.values.sum()), 1.0)

    lift_grids: dict[str, pd.DataFrame] = {}
    for method in method_order:
        if method == "full":
            continue
        grid = grids[method].astype(np.float64)
        method_share = grid / max(float(grid.values.sum()), 1.0)
        # Lift > 1 means this year-month cell is overrepresented relative to full-data frequency.
        lift = method_share / full_share.replace(0.0, np.nan)
        lift_grids[method] = lift.replace([np.inf, -np.inf], np.nan)

    return lift_grids, full_share


def _plot_year_month_lift_heatmaps(
    lift_grids: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    if len(lift_grids) == 0:
        return
    methods = list(lift_grids.keys())
    ncols = len(methods)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols + 1.2, 7), squeeze=False)
    finite_vals = np.concatenate(
        [np.ravel(grid.to_numpy(dtype=float)) for grid in lift_grids.values()]
    )
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    vmax = float(np.nanquantile(finite_vals, 0.98)) if finite_vals.size > 0 else 1.0
    vmax = max(vmax, 1.0)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    im = None
    for ax, method in zip(axes[0], methods, strict=True):
        grid = lift_grids[method]
        im = ax.imshow(
            grid.to_numpy(dtype=float),
            aspect="auto",
            origin="lower",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"{method} selection lift")
        ax.set_xlabel("year")
        ax.set_ylabel("month")
        ax.set_xticks(np.arange(grid.shape[1]))
        ax.set_xticklabels([str(c) for c in grid.columns], rotation=90)
        ax.set_yticks(np.arange(12))
        ax.set_yticklabels(month_labels)
    if im is not None:
        # Reserve a dedicated colorbar axis on the far right so it does not overlap the last panel.
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="selection lift vs full")
    fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])
    plt.savefig(out_path, dpi=180)
    plt.close()


def _compute_rolling_density(
    all_df: pd.DataFrame,
    method_order: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    if window_days <= 0:
        raise ValueError("--rolling-window-days must be > 0")

    day_index = pd.date_range(start=start_ts.floor("D"), end=end_ts.floor("D"), freq="D")
    out = pd.DataFrame(index=day_index)
    for method in method_order:
        method_df = all_df[all_df["method"] == method]
        daily = method_df.groupby(method_df["time"].dt.floor("D")).size().reindex(day_index, fill_value=0)
        total = float(daily.sum())
        if total > 0:
            density = daily / total
        else:
            density = daily.astype(float)
        out[method] = density.rolling(window=window_days, min_periods=1, center=True).mean()
    return out


def _plot_rolling_density(rolling_density: pd.DataFrame, out_path: Path, window_days: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in rolling_density.columns:
        ax.plot(rolling_density.index, rolling_density[col], label=col, linewidth=2.0)
    ax.set_title(f"Temporal Selection Density (Rolling {window_days}-Day Window)")
    ax.set_ylabel("density (smoothed share per day)")
    ax.set_xlabel("time")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _run_lengths(idx: np.ndarray) -> np.ndarray:
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    idx_sorted = np.sort(np.unique(idx))
    gaps = np.diff(idx_sorted)
    split_points = np.where(gaps != 1)[0] + 1
    runs = np.split(idx_sorted, split_points)
    return np.asarray([len(r) for r in runs], dtype=np.int64)


def _run_length_distribution(method_indices: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_counts: list[pd.Series] = []
    rows_share: list[pd.Series] = []
    for method, idx in method_indices.items():
        rl = _run_lengths(idx)
        if rl.size == 0:
            rows_counts.append(pd.Series(name=method, dtype=float))
            rows_share.append(pd.Series(name=method, dtype=float))
            continue
        counts = pd.Series(rl).value_counts().sort_index()
        counts.name = method
        share = counts / counts.sum()
        share.name = method
        rows_counts.append(counts)
        rows_share.append(share)
    counts_df = pd.DataFrame(rows_counts).fillna(0.0).sort_index(axis=1)
    share_df = pd.DataFrame(rows_share).fillna(0.0).sort_index(axis=1)
    counts_df.index.name = "method"
    share_df.index.name = "method"
    return counts_df, share_df


def _effective_window_metrics(method_indices: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for method, idx in method_indices.items():
        selected_n = int(np.unique(idx).size)
        rl = _run_lengths(idx)
        window_n = int(rl.size)
        avg_run = float(rl.mean()) if window_n > 0 else 0.0
        eff_ratio = float(window_n / selected_n) if selected_n > 0 else 0.0
        rows.append(
            {
                "method": method,
                "selected_samples": selected_n,
                "effective_windows": window_n,
                "avg_run_length": avg_run,
                "effective_ratio": eff_ratio,
            }
        )
    return pd.DataFrame(rows).set_index("method")


def _plot_run_length_share(share_df: pd.DataFrame, out_path: Path) -> None:
    # Keep chart readable: exact lengths 1..10, and aggregate the rest into >10.
    cols = list(share_df.columns)
    small_cols = [c for c in cols if int(c) <= 10]
    large_cols = [c for c in cols if int(c) > 10]
    disp = share_df.copy()
    disp = disp[small_cols].copy() if small_cols else pd.DataFrame(index=share_df.index)
    if large_cols:
        disp[">10"] = share_df[large_cols].sum(axis=1)
    if disp.shape[1] == 0:
        return
    ax = disp.T.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Consecutive Window Length Share")
    ax.set_ylabel("share")
    ax.set_xlabel("run length")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_effective_windows(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(metrics_df.shape[0])
    width = 0.38
    ax.bar(x - width / 2, metrics_df["selected_samples"].values, width=width, label="selected_samples")
    ax.bar(
        x + width / 2,
        metrics_df["effective_windows"].values,
        width=width,
        label="effective_windows",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index.tolist())
    ax.set_title("Selected Samples vs Effective Windows")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _compute_year_distance_share(all_df: pd.DataFrame, anchor_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = all_df.copy()
    df["year_distance"] = (df["year"] - int(anchor_year)).abs()
    dist_counts = (
        df.groupby(["method", "year_distance"]).size().unstack(fill_value=0).sort_index(axis=1)
    )
    dist_share = _normalize_rows(dist_counts)
    return dist_counts, dist_share


def _plot_year_distance_share(dist_share: pd.DataFrame, out_path: Path, anchor_year: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for method in dist_share.index:
        x = dist_share.columns.to_numpy(dtype=int)
        y = dist_share.loc[method].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, label=method)
    ax.set_title(f"Selection Probability vs |Year - {anchor_year}|")
    ax.set_xlabel(f"|train_year - {anchor_year}|")
    ax.set_ylabel("selection probability (share)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _load_feature_bank(
    pattern: str,
    feature_key: str,
    feature_index_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    import glob

    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    if len(paths) == 0:
        raise FileNotFoundError(f"No feature files matched --feature-pattern: {pattern}")

    idx_all = []
    feat_all = []
    for p in paths:
        with np.load(p) as data:
            if feature_index_key not in data:
                raise KeyError(f"Missing '{feature_index_key}' in feature file: {p}")
            if feature_key not in data:
                raise KeyError(f"Missing '{feature_key}' in feature file: {p}")
            idx = np.asarray(data[feature_index_key], dtype=np.int64).reshape(-1)
            feat = np.asarray(data[feature_key], dtype=np.float32)
            if feat.shape[0] != idx.shape[0]:
                raise ValueError(
                    f"Feature/sample length mismatch in {p}: idx={idx.shape[0]} feat={feat.shape[0]}"
                )
            idx_all.append(idx)
            feat_all.append(feat)

    idx_all_cat = np.concatenate(idx_all, axis=0)
    feat_all_cat = np.concatenate(feat_all, axis=0)

    uniq_idx, inverse, counts = np.unique(idx_all_cat, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        d = feat_all_cat.shape[1]
        feat_sum = np.zeros((uniq_idx.shape[0], d), dtype=np.float64)
        np.add.at(feat_sum, inverse, feat_all_cat.astype(np.float64))
        feat_uniq = (feat_sum / counts[:, None]).astype(np.float32)
    else:
        feat_uniq = feat_all_cat.astype(np.float32, copy=False)
    return uniq_idx.astype(np.int64), feat_uniq


def _subset_features(
    subset_idx: np.ndarray,
    bank_idx: np.ndarray,
    bank_feat: np.ndarray,
) -> np.ndarray:
    target = np.unique(subset_idx.astype(np.int64))
    pos = np.searchsorted(bank_idx, target)
    within = pos < bank_idx.shape[0]
    pos_in = pos[within]
    target_in = target[within]
    exact = bank_idx[pos_in] == target_in
    pos = pos_in[exact]
    if pos.size == 0:
        return np.zeros((0, bank_feat.shape[1]), dtype=np.float32)
    return bank_feat[pos]


def _pairwise_cosine_sample(
    feat: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = feat.shape[0]
    if n < 2:
        return np.zeros(0, dtype=np.float32)
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    feat = feat / np.clip(norms, 1e-12, None)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        sims = []
        for i in range(n):
            v = feat[i : i + 1]
            sims.append((v @ feat[i + 1 :].T).ravel())
        if len(sims) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(sims, axis=0).astype(np.float32)

    i = rng.integers(0, n, size=max_pairs, dtype=np.int64)
    j = rng.integers(0, n, size=max_pairs, dtype=np.int64)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(np.sum(same)), dtype=np.int64)
        same = i == j
    sims = np.sum(feat[i] * feat[j], axis=1)
    return sims.astype(np.float32)


def _plot_similarity_hist(sim_by_method: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for method, sims in sim_by_method.items():
        if sims.size == 0:
            continue
        ax.hist(sims, bins=60, density=True, histtype="step", linewidth=2.0, label=method)
    ax.set_title("Subset Internal Pairwise Cosine Similarity Distribution")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_similarity_box(sim_by_method: dict[str, np.ndarray], out_path: Path) -> None:
    methods = [m for m, v in sim_by_method.items() if v.size > 0]
    if not methods:
        return
    data = [sim_by_method[m] for m in methods]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=methods, showfliers=False)
    ax.set_title("Subset Internal Pairwise Cosine Similarity (Boxplot)")
    ax.set_ylabel("cosine similarity")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ts, end_ts, step_hours, full_idx, cfg_anchor_year = _get_time_grid(args.config)
    anchor_year = int(args.val_anchor_year) if args.val_anchor_year is not None else cfg_anchor_year
    total_steps = int(full_idx.size)

    method_indices: dict[str, np.ndarray] = {"full": full_idx}
    tarot_idx = _load_indices(args.tarot)
    random_idx = _load_indices(args.random)
    stratified_idx = _load_indices(args.stratified)
    if tarot_idx is not None:
        method_indices["tarot"] = tarot_idx
    if random_idx is not None:
        method_indices["random"] = random_idx
    if stratified_idx is not None:
        method_indices["stratified"] = stratified_idx

    for name, idx in method_indices.items():
        _validate_index_range(name, idx, total_steps)

    method_order = list(method_indices.keys())
    frames = [
        _indices_to_time_df(name, idx, start_ts, step_hours) for name, idx in method_indices.items()
    ]
    all_df = pd.concat(frames, ignore_index=True)

    year_counts = (
        all_df.groupby(["method", "year"]).size().unstack(fill_value=0).sort_index(axis=1)
    )
    month_counts = (
        all_df.groupby(["method", "month"]).size().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
    )
    season_order = ["DJF", "MAM", "JJA", "SON"]
    season_counts = (
        all_df.groupby(["method", "season"]).size().unstack(fill_value=0).reindex(columns=season_order, fill_value=0)
    )

    year_share = _normalize_rows(year_counts)
    month_share = _normalize_rows(month_counts)
    season_share = _normalize_rows(season_counts)
    rolling_density = _compute_rolling_density(
        all_df=all_df,
        method_order=method_order,
        start_ts=start_ts,
        end_ts=end_ts,
        window_days=int(args.rolling_window_days),
    )
    run_length_counts, run_length_share = _run_length_distribution(method_indices)
    effective_metrics = _effective_window_metrics(method_indices)
    year_distance_counts, year_distance_share = _compute_year_distance_share(all_df, anchor_year)
    year_month_lift_grids, full_year_month_share = _compute_year_month_lift(all_df, method_order)

    year_counts.to_csv(output_dir / "year_counts.csv")
    month_counts.to_csv(output_dir / "month_counts.csv")
    season_counts.to_csv(output_dir / "season_counts.csv")
    year_share.to_csv(output_dir / "year_share.csv")
    month_share.to_csv(output_dir / "month_share.csv")
    season_share.to_csv(output_dir / "season_share.csv")
    rolling_density.to_csv(output_dir / "rolling_density.csv", index_label="date")
    run_length_counts.to_csv(output_dir / "run_length_counts.csv")
    run_length_share.to_csv(output_dir / "run_length_share.csv")
    effective_metrics.to_csv(output_dir / "effective_windows_metrics.csv")
    year_distance_counts.to_csv(output_dir / "year_distance_counts.csv")
    year_distance_share.to_csv(output_dir / "year_distance_share.csv")
    full_year_month_share.to_csv(output_dir / "year_month_full_share.csv")
    for method, grid in year_month_lift_grids.items():
        grid.to_csv(output_dir / f"year_month_lift_{method}.csv")

    _plot_grouped_bar(
        year_counts,
        "Selected Sample Counts by Year",
        "count",
        output_dir / "year_counts.png",
    )
    _plot_grouped_bar(
        year_share,
        "Selected Sample Share by Year",
        "share",
        output_dir / "year_share.png",
    )
    _plot_grouped_bar(
        month_share,
        "Selected Sample Share by Month",
        "share",
        output_dir / "month_share.png",
    )
    _plot_grouped_bar(
        season_share,
        "Selected Sample Share by Season",
        "share",
        output_dir / "season_share.png",
    )
    _plot_rolling_density(
        rolling_density,
        output_dir / "rolling_density.png",
        window_days=int(args.rolling_window_days),
    )
    _plot_run_length_share(run_length_share, output_dir / "run_length_share.png")
    _plot_effective_windows(effective_metrics, output_dir / "effective_windows.png")
    _plot_year_distance_share(
        year_distance_share,
        output_dir / "year_distance_share.png",
        anchor_year=anchor_year,
    )
    _plot_year_month_lift_heatmaps(
        year_month_lift_grids,
        output_dir / "year_month_lift_heatmap.png",
    )

    if args.feature_pattern is not None:
        bank_idx, bank_feat = _load_feature_bank(
            pattern=str(args.feature_pattern),
            feature_key=str(args.feature_key),
            feature_index_key=str(args.feature_index_key),
        )
        rng = np.random.default_rng(42)
        max_samples = int(args.similarity_max_samples_per_method)
        max_pairs = int(args.similarity_max_pairs)
        sim_by_method: dict[str, np.ndarray] = {}
        summary_rows = []
        for method, idx in method_indices.items():
            feat_subset = _subset_features(idx, bank_idx, bank_feat)
            if feat_subset.shape[0] > max_samples:
                sel = rng.choice(feat_subset.shape[0], size=max_samples, replace=False)
                feat_subset = feat_subset[np.sort(sel)]
            sims = _pairwise_cosine_sample(feat_subset, max_pairs=max_pairs, rng=rng)
            sim_by_method[method] = sims
            if sims.size > 0:
                summary_rows.append(
                    {
                        "method": method,
                        "num_samples_used": int(feat_subset.shape[0]),
                        "num_pairs_used": int(sims.shape[0]),
                        "sim_mean": float(np.mean(sims)),
                        "sim_std": float(np.std(sims)),
                        "sim_q25": float(np.quantile(sims, 0.25)),
                        "sim_median": float(np.quantile(sims, 0.50)),
                        "sim_q75": float(np.quantile(sims, 0.75)),
                    }
                )
            else:
                summary_rows.append(
                    {
                        "method": method,
                        "num_samples_used": int(feat_subset.shape[0]),
                        "num_pairs_used": 0,
                        "sim_mean": np.nan,
                        "sim_std": np.nan,
                        "sim_q25": np.nan,
                        "sim_median": np.nan,
                        "sim_q75": np.nan,
                    }
                )

        sim_summary = pd.DataFrame(summary_rows).set_index("method")
        sim_summary.to_csv(output_dir / "subset_similarity_summary.csv")
        _plot_similarity_hist(sim_by_method, output_dir / "subset_similarity_hist.png")
        _plot_similarity_box(sim_by_method, output_dir / "subset_similarity_boxplot.png")

    print(f"Saved plots and CSV summaries to: {output_dir}")
    print(f"Methods included: {list(method_indices.keys())}")
    print(f"Validation anchor year: {anchor_year}")


if __name__ == "__main__":
    main()
