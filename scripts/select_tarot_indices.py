#!/usr/bin/env python3

import argparse
import glob
import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LoadedFeatures:
    ids: np.ndarray
    feat: np.ndarray
    idx_local: np.ndarray
    idx_time_ns: np.ndarray
    idx_global: np.ndarray
    file_id: np.ndarray
    row_in_file: np.ndarray
    paths: list[str]
    id_mode: str


def _choose_id_mode(requested_mode: str, paths: list[str], has_global: bool, has_time: bool) -> str:
    if requested_mode != "auto":
        if requested_mode == "idx_global" and not has_global:
            raise KeyError("Requested --id-mode idx_global but some files are missing 'idx_global'.")
        if requested_mode == "idx_time_ns" and not has_time:
            raise KeyError("Requested --id-mode idx_time_ns but some files are missing 'idx_time_ns'.")
        return requested_mode

    if has_global:
        return "idx_global"
    if has_time:
        return "idx_time_ns"

    parent_dirs = {str(pathlib.Path(p).parent) for p in paths}
    # Single run (possibly multi-rank): local idx is usually sufficient.
    if len(parent_dirs) == 1:
        return "idx"
    # Multiple run shards: local idx collides across shards, so include file identity.
    return "shard_local"


def _make_ids(id_mode: str, idx_local: np.ndarray, idx_time_ns: np.ndarray, idx_global: np.ndarray, file_id: np.ndarray) -> np.ndarray:
    if id_mode == "idx":
        return idx_local.astype(np.int64, copy=False)
    if id_mode == "idx_global":
        return idx_global.astype(np.int64, copy=False)
    if id_mode == "idx_time_ns":
        return idx_time_ns.astype(np.int64, copy=False)
    if id_mode == "shard_local":
        return (file_id.astype(np.int64) << np.int64(32)) | (
            idx_local.astype(np.int64) & np.int64(0xFFFFFFFF)
        )
    raise ValueError(f"Unknown id_mode: {id_mode}")


def _load_rank_files(pattern: str, feature_key: str, id_mode: str) -> LoadedFeatures:
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No feature files matched pattern: {pattern}")

    idx_local_all = []
    idx_time_ns_all = []
    idx_global_all = []
    feat_all = []
    file_id_all = []
    row_in_file_all = []
    has_global_all = True
    has_time_all = True

    for fid, path in enumerate(paths):
        with np.load(path) as data:
            if "idx" not in data:
                raise KeyError(f"Missing 'idx' in feature file: {path}")
            if feature_key not in data:
                raise KeyError(f"Missing '{feature_key}' in feature file: {path}")
            idx_local = np.asarray(data["idx"], dtype=np.int64).reshape(-1)
            feat = np.asarray(data[feature_key], dtype=np.float32)
            if feat.shape[0] != idx_local.shape[0]:
                raise ValueError(
                    f"Feature/sample length mismatch in {path}: idx={idx_local.shape[0]} feat={feat.shape[0]}"
                )
            has_time = "idx_time_ns" in data
            has_global = "idx_global" in data
            has_time_all = has_time_all and has_time
            has_global_all = has_global_all and has_global
            if has_time:
                idx_time_ns = np.asarray(data["idx_time_ns"], dtype=np.int64).reshape(-1)
            else:
                idx_time_ns = np.full(idx_local.shape[0], -1, dtype=np.int64)
            if has_global:
                idx_global = np.asarray(data["idx_global"], dtype=np.int64).reshape(-1)
            else:
                idx_global = np.full(idx_local.shape[0], -1, dtype=np.int64)

        idx_local_all.append(idx_local)
        idx_time_ns_all.append(idx_time_ns)
        idx_global_all.append(idx_global)
        feat_all.append(feat)
        file_id_all.append(np.full(idx_local.shape[0], fid, dtype=np.int64))
        row_in_file_all.append(np.arange(idx_local.shape[0], dtype=np.int64))

    idx_local = np.concatenate(idx_local_all, axis=0)
    idx_time_ns = np.concatenate(idx_time_ns_all, axis=0)
    idx_global = np.concatenate(idx_global_all, axis=0)
    feat = np.concatenate(feat_all, axis=0)
    file_id = np.concatenate(file_id_all, axis=0)
    row_in_file = np.concatenate(row_in_file_all, axis=0)
    id_mode_used = _choose_id_mode(id_mode, paths, has_global_all, has_time_all)
    ids = _make_ids(id_mode_used, idx_local, idx_time_ns, idx_global, file_id)
    return LoadedFeatures(
        ids=ids,
        feat=feat,
        idx_local=idx_local,
        idx_time_ns=idx_time_ns,
        idx_global=idx_global,
        file_id=file_id,
        row_in_file=row_in_file,
        paths=paths,
        id_mode=id_mode_used,
    )


def _deduplicate_by_id(loaded: LoadedFeatures, policy: str) -> tuple[LoadedFeatures, int]:
    unique_ids, inverse, counts = np.unique(loaded.ids, return_inverse=True, return_counts=True)
    dup_count = int(np.sum(counts > 1))
    if dup_count == 0:
        # Preserve original ordering when there are no duplicates.
        return loaded, 0

    if policy == "error":
        raise ValueError(
            f"Found {dup_count} duplicated IDs across feature files. "
            "Use --dedup-policy mean/first to resolve explicitly."
        )

    first_positions = np.full(unique_ids.shape[0], -1, dtype=np.int64)
    for i, uid in enumerate(inverse):
        if first_positions[uid] == -1:
            first_positions[uid] = i

    if policy == "first":
        return (
            LoadedFeatures(
                ids=unique_ids,
                feat=loaded.feat[first_positions],
                idx_local=loaded.idx_local[first_positions],
                idx_time_ns=loaded.idx_time_ns[first_positions],
                idx_global=loaded.idx_global[first_positions],
                file_id=loaded.file_id[first_positions],
                row_in_file=loaded.row_in_file[first_positions],
                paths=loaded.paths,
                id_mode=loaded.id_mode,
            ),
            dup_count,
        )

    if policy == "mean":
        d = loaded.feat.shape[1]
        feat_sum = np.zeros((unique_ids.shape[0], d), dtype=np.float64)
        np.add.at(feat_sum, inverse, loaded.feat.astype(np.float64))
        feat_mean = feat_sum / counts[:, None]
        return (
            LoadedFeatures(
                ids=unique_ids,
                feat=feat_mean.astype(np.float32),
                idx_local=loaded.idx_local[first_positions],
                idx_time_ns=loaded.idx_time_ns[first_positions],
                idx_global=loaded.idx_global[first_positions],
                file_id=loaded.file_id[first_positions],
                row_in_file=loaded.row_in_file[first_positions],
                paths=loaded.paths,
                id_mode=loaded.id_mode,
            ),
            dup_count,
        )

    raise ValueError(f"Unknown dedup policy: {policy}")


def _normalize_features(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def _whiten_features(
    cand_feat: np.ndarray, tgt_feat: np.ndarray, reg: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """Joint Cholesky whitening of candidate and target features.

    Mirrors the whitening step in TAROT's WFDEstimator.get_wfd():
      1. Concatenate and center (joint mean subtraction).
      2. Compute biased covariance: X^T X / N.
      3. Regularize: add reg * I.
      4. Cholesky-factor and invert: W = inv(L).T.
      5. Apply: X_whitened = X_centered @ W.

    The caller is responsible for L2-normalizing the returned features
    before passing them to DataSelector or the score matrix function.
    """
    n_cand = cand_feat.shape[0]
    all_feat = np.concatenate([cand_feat, tgt_feat], axis=0).astype(np.float64)
    all_feat -= all_feat.mean(axis=0, keepdims=True)
    n = all_feat.shape[0]
    xtx = (all_feat.T @ all_feat) / n
    xtx += np.eye(xtx.shape[0]) * reg
    L = np.linalg.cholesky(xtx)
    W = np.linalg.inv(L).T
    all_feat = (all_feat @ W).astype(np.float32)
    return all_feat[:n_cand], all_feat[n_cand:]


def _subsample_loaded(loaded: LoadedFeatures, max_samples: int, seed: int) -> LoadedFeatures:
    if max_samples <= 0 or loaded.ids.shape[0] <= max_samples:
        return loaded
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(loaded.ids.shape[0], size=max_samples, replace=False))
    return LoadedFeatures(
        ids=loaded.ids[sel],
        feat=loaded.feat[sel],
        idx_local=loaded.idx_local[sel],
        idx_time_ns=loaded.idx_time_ns[sel],
        idx_global=loaded.idx_global[sel],
        file_id=loaded.file_id[sel],
        row_in_file=loaded.row_in_file[sel],
        paths=loaded.paths,
        id_mode=loaded.id_mode,
    )


def _time_ns_to_global_idx(time_ns: np.ndarray, origin: str, step_hours: int) -> np.ndarray:
    if step_hours <= 0:
        raise ValueError("--global-index-step-hours must be > 0")
    origin_ns = np.datetime64(origin).astype("datetime64[ns]").astype(np.int64)
    step_ns = np.int64(step_hours) * np.int64(3600 * 10**9)
    delta = time_ns.astype(np.int64) - origin_ns
    if np.any(delta < 0):
        raise ValueError("Some selected samples are earlier than --global-index-origin.")
    if np.any(delta % step_ns != 0):
        raise ValueError(
            "Some selected samples are not aligned to --global-index-step-hours; "
            "cannot convert to integer global indices."
        )
    return (delta // step_ns).astype(np.int64)


def _shard_local_to_global_idx(
    idx_local: np.ndarray,
    file_id: np.ndarray,
    shard_start_dates: str,
    origin: str,
    step_hours: int,
    num_candidate_files: int,
) -> np.ndarray:
    starts = [s.strip() for s in shard_start_dates.split(",") if s.strip()]
    if len(starts) != num_candidate_files:
        raise ValueError(
            "Length mismatch: --candidate-shard-start-dates must provide one datetime per "
            f"matched candidate file. got={len(starts)} expected={num_candidate_files}"
        )
    offsets = _time_ns_to_global_idx(
        np.asarray(starts, dtype="datetime64[ns]").astype(np.int64),
        origin=origin,
        step_hours=step_hours,
    )
    return offsets[file_id.astype(np.int64)] + idx_local.astype(np.int64)


def _infomax_select(
    cand_feat_norm: np.ndarray,
    score: np.ndarray,
    n_select: int,
    n_neighbor: int = 10,
    gamma: float = -1.0,
    mis_ratio: float = 0.0,
    importance_agg: str = "mean",
    n_stratas: int = 1,
    n_importance_iter: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Hybrid InfoMax + D2Pruning coreset selection.

    Implements graph-density sampling from https://arxiv.org/abs/2506.01701 with
    an optional D2Pruning pre-filter that removes the least relevant candidates.

    Returns (selected_positions, importance_weights) where positions index into the
    original cand_feat_norm array.
    """
    n_cand = cand_feat_norm.shape[0]
    d = cand_feat_norm.shape[1]
    if gamma < 0.0:
        gamma = 1.0 / d

    # Per-candidate importance: aggregate cosine similarity to all targets.
    if importance_agg == "mean":
        importance = score.mean(axis=1).astype(np.float64)
    elif importance_agg == "max":
        importance = score.max(axis=1).astype(np.float64)
    else:
        importance = score.sum(axis=1).astype(np.float64)

    # D2Pruning: discard bottom mis_ratio fraction (least relevant to target distribution).
    original_indices = np.arange(n_cand, dtype=np.int64)
    if mis_ratio > 0.0:
        n_prune = max(1, int(mis_ratio * n_cand))
        prune_idx = np.argpartition(importance, n_prune)[:n_prune]
        keep_mask = np.ones(n_cand, dtype=bool)
        keep_mask[prune_idx] = False
        original_indices = np.where(keep_mask)[0].astype(np.int64)
        cand_feat_norm = cand_feat_norm[original_indices]
        importance = importance[original_indices]
        print(f"  D2Pruning: removed {n_prune} low-relevance candidates, {len(original_indices)} remain.")

    n_pool = len(original_indices)
    n_select = min(n_select, n_pool)
    k = min(n_neighbor, n_pool - 1)

    if n_stratas > 1:
        return _infomax_stratified(
            cand_feat_norm, importance, original_indices, n_select, k, gamma, n_stratas,
            n_importance_iter,
        )

    return _infomax_graph_select(
        cand_feat_norm, importance, original_indices, n_select, k, gamma, n_importance_iter
    )


def _build_knn_graph(feat: np.ndarray, k: int, block_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """k-NN graph via dot product on L2-normalised features (cosine similarity).

    Uses cand @ cand.T in blocks so no O(n²) memory spike.
    Returns (cosine_distances, nn_indices) where cosine_distance = 1 - cos_sim.
    """
    n = feat.shape[0]
    k = min(k, n - 1)
    feat_t = torch.from_numpy(feat)          # already L2-normalised

    nn_dists = np.empty((n, k), dtype=np.float32)
    nn_idxs  = np.empty((n, k), dtype=np.int64)

    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        sim = feat_t[i0:i1] @ feat_t.T      # (block, n) cosine similarities
        dist = 1.0 - sim                     # cosine distance in [0, 2]
        # Exclude self by setting its distance to a large value before top-k.
        self_idx = torch.arange(i0, i1, device=dist.device).unsqueeze(1)
        dist.scatter_(1, self_idx, 2.0)
        vals, idx = torch.topk(dist, k, largest=False)
        nn_dists[i0:i1] = vals.float().numpy()
        nn_idxs[i0:i1]  = idx.numpy()

    return nn_dists, nn_idxs


def _graph_density_greedy(
    feat: np.ndarray,
    importance: np.ndarray,
    n_select: int,
    k: int,
    gamma: float,
    n_importance_iter: int = 1,
) -> np.ndarray:
    """Greedy InfoMax graph-density selection on a pool of candidates.

    Graph density for node i = Σ_j (1 - exp(-dist(i,j))) * importance[j]
    After selecting node s, reduce each neighbor j's density by
    exp(-dist(s,j) * gamma) * density[s] to penalise redundancy.

    n_importance_iter > 1 refines importance scores by propagating graph density
    back as the importance signal for the next iteration (GCCG-style).
    """
    n_pool = feat.shape[0]
    print(f"    building k={k} NN graph on {n_pool} samples (dim={feat.shape[1]})...")
    nn_dists, nn_idxs = _build_knn_graph(feat, k)

    epsilon = 1e-7
    importance = np.maximum(importance.copy(), epsilon)
    for it in range(n_importance_iter):
        neighbor_imp = importance[nn_idxs]                     # (n, k)
        edge_w = (1.0 - np.exp(-nn_dists)) * neighbor_imp     # (n, k)
        new_importance = edge_w.sum(axis=1).astype(np.float64)
        max_imp = new_importance.max()
        if max_imp > 0:
            new_importance /= max_imp
        importance = np.maximum(new_importance, epsilon)
        if n_importance_iter > 1:
            print(
                f"    importance iter {it + 1}/{n_importance_iter}: "
                f"min={importance.min():.4f} max={importance.max():.4f}"
            )

    neighbor_imp = importance[nn_idxs]                        # (n, k)
    edge_w = (1.0 - np.exp(-nn_dists)) * neighbor_imp        # (n, k)
    graph_density = edge_w.sum(axis=1).astype(np.float64)

    available = np.ones(n_pool, dtype=bool)
    selected = np.empty(n_select, dtype=np.int64)

    for step in range(n_select):
        sel = int(np.argmax(np.where(available, graph_density, -np.inf)))
        selected[step] = sel
        available[sel] = False

        nbs = nn_idxs[sel]
        decay = np.exp(-nn_dists[sel] * gamma) * graph_density[sel]
        graph_density[nbs] = np.maximum(0.0, graph_density[nbs] - decay)

    return selected


def _infomax_graph_select(
    feat: np.ndarray,
    importance: np.ndarray,
    original_indices: np.ndarray,
    n_select: int,
    k: int,
    gamma: float,
    n_importance_iter: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"  InfoMax: selecting {n_select} from {len(original_indices)} candidates...")
    local_sel = _graph_density_greedy(feat, importance, n_select, k, gamma, n_importance_iter)
    return original_indices[local_sel], importance[local_sel].astype(np.float32)


def _infomax_stratified(
    feat: np.ndarray,
    importance: np.ndarray,
    original_indices: np.ndarray,
    n_select: int,
    k: int,
    gamma: float,
    n_stratas: int,
    n_importance_iter: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified InfoMax: divide by importance bins, run graph selection per stratum."""
    n_pool = len(original_indices)
    print(
        f"  InfoMax stratified: {n_stratas} strata, selecting {n_select} from {n_pool} candidates..."
    )

    # Assign each candidate to a stratum by importance rank.
    rank = np.argsort(np.argsort(importance))   # rank[i] in [0, n_pool)
    stratum_id = (rank * n_stratas // n_pool).clip(0, n_stratas - 1)

    # Uniform budget across strata; distribute remainder to top strata.
    base_budget = n_select // n_stratas
    remainder = n_select - base_budget * n_stratas
    budgets = np.full(n_stratas, base_budget, dtype=np.int64)
    if remainder > 0:
        budgets[-remainder:] += 1  # give extra slots to highest-importance strata

    all_sel = []
    all_imp = []
    for s in range(n_stratas):
        mask = stratum_id == s
        if not mask.any():
            continue
        pool_idx = np.where(mask)[0]
        budget = int(budgets[s])
        if budget <= 0:
            continue
        budget = min(budget, len(pool_idx))
        local_sel = _graph_density_greedy(
            feat[pool_idx], importance[pool_idx], budget, k, gamma, n_importance_iter
        )
        all_sel.append(original_indices[pool_idx[local_sel]])
        all_imp.append(importance[pool_idx[local_sel]])

    sel_pos = np.concatenate(all_sel).astype(np.int64)
    sel_imp = np.concatenate(all_imp).astype(np.float32)
    return sel_pos, sel_imp


def _cosine_score_matrix(
    candidate: np.ndarray,
    target: np.ndarray,
    block_size: int,
    score_memmap_path: str | None,
) -> np.ndarray:
    candidate_norm = _normalize_features(candidate.astype(np.float32, copy=False))
    target_norm = _normalize_features(target.astype(np.float32, copy=False))
    n_cand = candidate_norm.shape[0]
    n_tgt = target_norm.shape[0]

    if score_memmap_path is None:
        score = np.empty((n_cand, n_tgt), dtype=np.float32)
    else:
        score_path = pathlib.Path(score_memmap_path)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score = np.memmap(score_path, dtype=np.float32, mode="w+", shape=(n_cand, n_tgt))

    for i0 in range(0, n_cand, block_size):
        i1 = min(i0 + block_size, n_cand)
        score[i0:i1] = candidate_norm[i0:i1] @ target_norm.T

    return score


def main():
    parser = argparse.ArgumentParser(description="Select TAROT indices from exported WG features.")
    parser.add_argument("--candidate", required=True, help="Glob for candidate npz files.")
    parser.add_argument("--target", required=True, help="Glob for target npz files.")
    parser.add_argument(
        "--feature",
        default="mean_std",
        choices=["mean", "mean_std", "grad_projected"],
        help="Feature key to use from exported npz files.",
    )
    parser.add_argument("--ratio", type=float, default=0.2, help="Selection ratio in (0,1].")
    parser.add_argument("--output", required=True, help="Output .npy path for selected IDs.")
    parser.add_argument(
        "--method",
        default="fixed_size",
        choices=["fixed_size", "random", "dsdm", "less", "otm", "infomax"],
        help="Selection method. 'infomax' uses graph-density sampling (InfoMax + optional D2Pruning).",
    )
    parser.add_argument(
        "--infomax-n-neighbor",
        type=int,
        default=10,
        help="[infomax] Number of nearest neighbours for the k-NN graph.",
    )
    parser.add_argument(
        "--infomax-gamma",
        type=float,
        default=-1.0,
        help="[infomax] RBF decay factor for density update. Negative → auto (1/feature_dim).",
    )
    parser.add_argument(
        "--infomax-mis-ratio",
        type=float,
        default=0.0,
        help=(
            "[infomax] D2Pruning pre-filter ratio in [0, 1). "
            "Remove this fraction of candidates with the lowest mean cosine score to target before selection."
        ),
    )
    parser.add_argument(
        "--infomax-importance",
        default="mean",
        choices=["mean", "max", "sum"],
        help="[infomax] How to aggregate per-candidate cosine scores into a single importance value.",
    )
    parser.add_argument(
        "--infomax-stratas",
        type=int,
        default=1,
        help=(
            "[infomax] Number of importance strata for stratified selection. "
            "1 = no stratification (pure graph-density greed). "
            ">1 = divide candidates into this many importance bins and run graph selection per bin."
        ),
    )
    parser.add_argument(
        "--infomax-importance-iter",
        type=int,
        default=1,
        help=(
            "[infomax] Number of graph-density propagation iterations to refine importance scores "
            "before greedy selection. 1 = single pass (original behaviour). "
            ">1 = iteratively update importance[i] = Σ_j (1-exp(-dist)) * importance[j], "
            "normalised, so local target-relevance diffuses through the k-NN graph (GCCG-style)."
        ),
    )
    parser.add_argument(
        "--dedup-policy",
        default="mean",
        choices=["mean", "first", "error"],
        help="How to handle duplicated IDs across matched feature files.",
    )
    parser.add_argument(
        "--id-mode",
        default="auto",
        choices=["auto", "idx", "idx_global", "idx_time_ns", "shard_local"],
        help=(
            "Identity key used for dedup/output. "
            "auto: idx_global -> idx_time_ns -> idx(single-run) -> shard_local(multi-run)."
        ),
    )
    parser.add_argument(
        "--output-global-idx",
        default=None,
        help="Optional output .npy path for training-ready global indices.",
    )
    parser.add_argument(
        "--global-index-origin",
        default=None,
        help="Datetime origin (e.g. 1979-01-01) for time->global index conversion.",
    )
    parser.add_argument(
        "--global-index-step-hours",
        type=int,
        default=6,
        help="Step size in hours for time->global index conversion.",
    )
    parser.add_argument(
        "--candidate-shard-start-dates",
        default=None,
        help=(
            "Comma-separated shard start datetimes aligned with sorted candidate files. "
            "Used to convert shard_local IDs to global indices for legacy exports."
        ),
    )
    parser.add_argument(
        "--target-max-samples",
        type=int,
        default=0,
        help="If >0, subsample target features to this many samples before score computation.",
    )
    parser.add_argument(
        "--target-subsample-seed",
        type=int,
        default=42,
        help="Random seed for target subsampling.",
    )
    parser.add_argument(
        "--score-block-size",
        type=int,
        default=4096,
        help="Block size along candidate axis for cosine score matrix computation.",
    )
    parser.add_argument(
        "--score-memmap-path",
        default=None,
        help=(
            "Optional path to store score matrix as memmap (reduces RAM pressure, uses disk). "
            "If omitted, score is kept in RAM."
        ),
    )
    parser.add_argument(
        "--save-score-analysis",
        action="store_true",
        default=False,
        help="Save score analysis sidecar file with features, aggregated scores, and OT weights.",
    )
    parser.add_argument(
        "--save-all-candidate-feat",
        action="store_true",
        default=False,
        help="Save whitened+normalized features for ALL candidates (not just selected).",
    )
    args = parser.parse_args()

    if not (0.0 < args.ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {args.ratio}")
    if args.score_block_size <= 0:
        raise ValueError("--score-block-size must be > 0")
    if not (0.0 <= args.infomax_mis_ratio < 1.0):
        raise ValueError(f"--infomax-mis-ratio must be in [0, 1), got {args.infomax_mis_ratio}")
    if args.infomax_stratas < 1:
        raise ValueError(f"--infomax-stratas must be >= 1, got {args.infomax_stratas}")
    if args.infomax_importance_iter < 1:
        raise ValueError(f"--infomax-importance-iter must be >= 1, got {args.infomax_importance_iter}")

    if args.method != "infomax":
        tarot_root = pathlib.Path(__file__).resolve().parent.parent / "TAROT"
        sys.path.insert(0, str(tarot_root))
        from tarot.data_selector import DataSelector

    feature_key = f"feat_{args.feature}"
    cand_loaded = _load_rank_files(args.candidate, feature_key, args.id_mode)
    tgt_loaded = _load_rank_files(args.target, feature_key, args.id_mode)
    print(
        f"ID mode: candidate={cand_loaded.id_mode} target={tgt_loaded.id_mode} "
        f"(requested={args.id_mode})"
    )
    if args.id_mode == "auto" and cand_loaded.id_mode != tgt_loaded.id_mode:
        print(
            "WARNING: candidate and target resolved different ID modes under auto mode. "
            "Selection remains valid, but selected IDs are in candidate ID space."
        )
    if cand_loaded.id_mode not in {"idx", "idx_global"} and args.output_global_idx is None:
        raise ValueError(
            "--output stores IDs (not directly training indices) for id_mode "
            f"'{cand_loaded.id_mode}'. Please also pass --output-global-idx."
        )
    cand_loaded, cand_dups = _deduplicate_by_id(cand_loaded, args.dedup_policy)
    tgt_loaded, tgt_dups = _deduplicate_by_id(tgt_loaded, args.dedup_policy)
    if cand_dups > 0 or tgt_dups > 0:
        print(
            f"Deduplicated repeated IDs with policy={args.dedup_policy}: "
            f"candidate_duplicates={cand_dups}, target_duplicates={tgt_dups}"
        )
    tgt_loaded = _subsample_loaded(tgt_loaded, args.target_max_samples, args.target_subsample_seed)
    if args.target_max_samples > 0:
        print(f"Target subsampling active: using {tgt_loaded.ids.shape[0]} target samples.")

    # Joint Cholesky whitening (mirrors WFDEstimator.get_wfd).
    # After whitening, apply per-row L2 normalization so that the score matrix
    # equals the cosine similarity in the whitened space, and the OT distance
    # in DataSelector's cosine_L2 is the correct sqrt(2 - 2·cos) metric.
    print(
        "Whitening features: "
        f"candidate={cand_loaded.feat.shape}, target={tgt_loaded.feat.shape}"
    )
    cand_feat, tgt_feat = _whiten_features(cand_loaded.feat, tgt_loaded.feat)
    cand_feat_norm = _normalize_features(cand_feat)
    tgt_feat_norm = _normalize_features(tgt_feat)
    print("Whitening done.")

    score_memmap_path = args.score_memmap_path
    tmp_memmap = None
    if score_memmap_path is None and cand_feat.shape[0] * tgt_feat.shape[0] > 40_000_000:
        tmp_memmap = tempfile.NamedTemporaryFile(
            prefix="tarot_score_", suffix=".mmap", delete=False
        )
        tmp_memmap.close()
        score_memmap_path = tmp_memmap.name
        print(
            f"Large score matrix detected ({cand_feat.shape[0]} x {tgt_feat.shape[0]}). "
            f"Using temporary memmap file: {score_memmap_path}"
        )

    score = None
    score_stats = None
    ot_weights = None
    try:
        # cand_feat_norm / tgt_feat_norm are whitened + L2-normalized,
        # matching the features returned by WFDEstimator.get_wfd().
        score = _cosine_score_matrix(
            cand_feat_norm,
            tgt_feat_norm,
            block_size=args.score_block_size,
            score_memmap_path=score_memmap_path,
        )

        if args.method == "infomax":
            n_select = max(1, int(args.ratio * cand_loaded.ids.shape[0]))
            print(
                f"InfoMax selection: ratio={args.ratio}, n_select={n_select}, "
                f"n_neighbor={args.infomax_n_neighbor}, gamma={args.infomax_gamma}, "
                f"mis_ratio={args.infomax_mis_ratio}, importance={args.infomax_importance}, "
                f"stratas={args.infomax_stratas}, importance_iter={args.infomax_importance_iter}"
            )
            selected_pos, ot_weights = _infomax_select(
                cand_feat_norm,
                score,
                n_select=n_select,
                n_neighbor=args.infomax_n_neighbor,
                gamma=args.infomax_gamma,
                mis_ratio=args.infomax_mis_ratio,
                importance_agg=args.infomax_importance,
                n_stratas=args.infomax_stratas,
                n_importance_iter=args.infomax_importance_iter,
            )
        else:
            cfg = {
                "device": "cpu",
                "selection_method": args.method,
                "selection_ratio": args.ratio,
                "k_fold_splits": 10,
                "merge_target_data": False,
                "data_weighting": False,
            }
            selector = DataSelector(cfg)
            selected_pos, ot_weights = selector.select_data(
                score, torch.from_numpy(cand_feat_norm), torch.from_numpy(tgt_feat_norm)
            )
            ot_weights = np.asarray(ot_weights, dtype=np.float32)

        selected_pos = np.asarray(selected_pos, dtype=np.int64)
        selected_ids_raw = cand_loaded.ids[selected_pos]
        selected_ids, unique_pos = np.unique(selected_ids_raw, return_index=True)
        ot_weights = ot_weights[unique_pos]
        selected_pos_unique = selected_pos[unique_pos]
        if args.save_score_analysis and score is not None:
            # Force a copy so score stats remain available after memmap cleanup.
            selected_scores = np.asarray(score[selected_pos_unique])
            score_stats = {
                "score_mean_to_target": selected_scores.mean(axis=1),
                "score_max_to_target": selected_scores.max(axis=1),
                "score_min_to_target": selected_scores.min(axis=1),
            }
            del selected_scores
        selected_idx_local = cand_loaded.idx_local[selected_pos_unique]
        selected_idx_time_ns = cand_loaded.idx_time_ns[selected_pos_unique]
        selected_idx_global = cand_loaded.idx_global[selected_pos_unique]
        selected_file_id = cand_loaded.file_id[selected_pos_unique]
        selected_row_in_file = cand_loaded.row_in_file[selected_pos_unique]
    finally:
        if tmp_memmap is not None:
            if score is not None:
                del score
            if score_memmap_path is not None and os.path.exists(score_memmap_path):
                os.remove(score_memmap_path)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, selected_ids)

    # Always persist sidecar metadata so selected IDs are auditable and mappable.
    meta_out_path = out_path.with_suffix(".meta.npz")
    np.savez(
        meta_out_path,
        selected_id=selected_ids,
        selected_idx_local=selected_idx_local,
        selected_idx_time_ns=selected_idx_time_ns,
        selected_idx_global=selected_idx_global,
        selected_file_id=selected_file_id,
        selected_row_in_file=selected_row_in_file,
        candidate_paths=np.asarray(cand_loaded.paths, dtype=str),
        id_mode=np.asarray([cand_loaded.id_mode]),
    )

    if args.save_score_analysis:
        analysis_path = out_path.with_suffix(".score_analysis.npz")
        analysis_data = {
            "selected_id": selected_ids,
            "selected_feat": cand_feat_norm[selected_pos_unique],
            "target_id": tgt_loaded.ids,
            "target_feat": tgt_feat_norm,
            "ot_weights": ot_weights,
        }
        if score_stats is not None:
            analysis_data.update(score_stats)
        np.savez(analysis_path, **analysis_data)
        print(f"Saved score analysis to {analysis_path}.")

    if args.save_all_candidate_feat:
        all_feat_path = out_path.with_suffix(".all_candidate_feat.npz")
        np.savez(
            all_feat_path,
            candidate_id=cand_loaded.ids,
            candidate_feat=cand_feat_norm,
        )
        print(f"Saved all candidate features ({cand_feat_norm.shape}) to {all_feat_path}.")

    # Optional: produce training-ready global indices.
    if args.output_global_idx is not None:
        if np.all(selected_idx_global >= 0):
            output_global_idx = selected_idx_global.astype(np.int64, copy=False)
        elif np.all(selected_idx_time_ns >= 0) and args.global_index_origin is not None:
            output_global_idx = _time_ns_to_global_idx(
                selected_idx_time_ns, args.global_index_origin, args.global_index_step_hours
            )
        elif (
            cand_loaded.id_mode == "shard_local"
            and args.global_index_origin is not None
            and args.candidate_shard_start_dates is not None
        ):
            output_global_idx = _shard_local_to_global_idx(
                idx_local=selected_idx_local,
                file_id=selected_file_id,
                shard_start_dates=args.candidate_shard_start_dates,
                origin=args.global_index_origin,
                step_hours=args.global_index_step_hours,
                num_candidate_files=len(cand_loaded.paths),
            )
        else:
            raise ValueError(
                "Cannot generate global indices. Need either exported idx_global, or "
                "idx_time_ns plus --global-index-origin, or shard_local plus "
                "--candidate-shard-start-dates and --global-index-origin."
            )
        output_global_idx = np.unique(output_global_idx)
        out_global = pathlib.Path(args.output_global_idx)
        out_global.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_global, output_global_idx)
        print(f"Saved {len(output_global_idx)} global indices to {out_global}.")

    print(
        f"Saved {len(selected_ids)} selected IDs to {out_path}. "
        f"candidate={len(cand_loaded.ids)} target={len(tgt_loaded.ids)} feature={args.feature}"
    )
    print(f"Saved selection metadata to {meta_out_path}.")


if __name__ == "__main__":
    main()
