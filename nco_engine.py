"""
scripts/nco_engine.py

Implements the full NCO pipeline with randomised stratified block
partitioning and ensemble averaging across M randomisations.

Pipeline per rebalance date × lookback window
---------------------------------------------
For m = 1 .. M:
    1. Stratified random partition → B blocks of ~n assets each
    2. Per block:
        a. Estimate V_block (n×n) with analytical Ledoit-Wolf
        b. De-noise V_block via Marchenko-Pastur eigenvalue clipping
        c. NCO within block:
              - K-Means clustering on correlation distance matrix
              - Intra-cluster : min variance   (no μ needed)
              - Inter-cluster : risk parity
        d. Scale weights by 1/B → universe-scaled weights
    3. Concatenate B blocks → full weight vector (N×1, sums to 1)

ω* = (1/M) Σ_m  ω^m   (universe-scaled weights already, mean preserves sum=1)

Diagnostics stored per asset:
    weight_mean   final weight
    weight_std    std across M randomisations
    weight_cv     coefficient of variation  (std / mean)

Usage
-----
python scripts/nco_engine.py [--run-id <id>] [--lookback 6m,4m,3m]
                              [--n-randomisations 50] [--n-blocks 20]
                              [--max-clusters 10] [--seed 42]
"""

import argparse
import os
import sys
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import silhouette_samples

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quant.utils.db import RunRegistry, get_connection

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_N_BLOCKS          = 15     # B  — number of blocks per randomisation
DEFAULT_N_RANDOMISATIONS  = 30     # M  — ensemble size
DEFAULT_MAX_CLUSTERS      = 10     # max K considered by silhouette search
DEFAULT_KMEANS_N_INIT     = 1      # K-Means restarts per K candidate
DEFAULT_SEED              = 42
DEFAULT_LOOKBACKS         = ["6m", "4m", "3m"]

# ── Asset class registry (matches pull_market_data.py) ────────────────────────

ASSET_CLASSES = [
    "crypto", "commodities", "indices", "forex",
    "sector_etf", "country_etf", "factor_etf",
    "fixed_income_etf", "commodity_etf", "equities",
    "leveraged_etf", "algo",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Covariance estimation
# ══════════════════════════════════════════════════════════════════════════════

def ledoit_wolf_cov(X: np.ndarray) -> np.ndarray:
    """
    Analytical Ledoit-Wolf shrinkage estimator.
    X : (T × n) returns matrix, no NaNs.
    Returns an (n × n) positive-definite covariance matrix.
    """
    lw = LedoitWolf(assume_centered=False)
    lw.fit(X)
    return lw.covariance_


def cov2corr(cov: np.ndarray) -> np.ndarray:
    """Covariance → correlation matrix."""
    std  = np.sqrt(np.diag(cov))
    # Guard against zero-variance columns (degenerate assets)
    std  = np.where(std < 1e-12, 1e-12, std)
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def corr2cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Correlation matrix + std vector → covariance matrix."""
    return corr * np.outer(std, std)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  De-noising via Marchenko-Pastur
# ══════════════════════════════════════════════════════════════════════════════

def _mp_pdf(var: float, q: float, pts: int = 1000) -> pd.Series:
    """Marchenko-Pastur PDF.  q = T/n."""
    e_min = var * (1.0 - q**-0.5) ** 2
    e_max = var * (1.0 + q**-0.5) ** 2
    e_val = np.linspace(e_min, e_max, pts)
    pdf   = (q / (2 * np.pi * var * e_val) *
             np.sqrt(np.maximum((e_max - e_val) * (e_val - e_min), 0)))
    return pd.Series(pdf, index=e_val)



def denoise_cov(cov: np.ndarray, q: float) -> np.ndarray:
    """
    De-noise a covariance matrix by clipping noise eigenvalues.

    Eigenvalues below the MP upper bound are replaced by their mean
    (preserving trace), leaving signal eigenvalues untouched.

    q = T/n ratio for this block.
    """
    corr          = cov2corr(cov)
    std           = np.sqrt(np.diag(cov))
    e_vals, e_vecs = np.linalg.eigh(corr)

    # eigh returns ascending order — reverse to descending
    e_vals  = e_vals[::-1]
    e_vecs  = e_vecs[:, ::-1]

    # σ²=1 exactly for a correlation matrix, so the theoretical
    # MP bound is used directly — no KDE fitting needed.
    e_max = (1.0 + q**-0.5) ** 2

    # Number of signal factors: eigenvalues strictly above e_max
    n_signal = int(np.sum(e_vals > e_max))
    n_signal = max(n_signal, 1)   # keep at least 1 signal factor

    # Clip: replace noise eigenvalues with their mean
    e_clip            = e_vals.copy()
    noise_mean        = e_clip[n_signal:].mean()
    e_clip[n_signal:] = noise_mean

    # Reconstruct correlation matrix
    corr_dn = e_vecs @ np.diag(e_clip) @ e_vecs.T
    corr_dn = cov2corr(corr_dn)   # re-normalise numerical drift

    return corr2cov(corr_dn, std), n_signal


# ══════════════════════════════════════════════════════════════════════════════
# 3.  NCO sub-routines
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_kmeans(corr: np.ndarray,
                    max_k: int,
                    n_init: int = DEFAULT_KMEANS_N_INIT,
                    rng_seed: int = 0) -> tuple[np.ndarray, int]:
    """
    Find the optimal K-Means partition of assets using silhouette score.

    Distance matrix: d_ij = sqrt(0.5 * (1 - ρ_ij))   (correlation distance)

    Returns
    -------
    labels   : (n,) integer cluster assignment
    best_k   : chosen number of clusters
    """
    n     = corr.shape[0]
    max_k = max(2, min(max_k, n - 1))

    if n < 3:
        return np.zeros(n, dtype=int), 1

    dist   = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))

    best_score  = -np.inf
    best_labels = np.zeros(n, dtype=int)
    best_k      = 2
    no_improve  = 0

    for k in range(2, max_k + 1):
        km = KMeans(
            n_clusters   = k,
            n_init       = n_init,
            random_state = rng_seed,
        )
        km.fit(dist)
        if len(np.unique(km.labels_)) < k:
            continue   # degenerate — fewer clusters than requested

        silh  = silhouette_samples(dist, km.labels_, metric="precomputed")
        score = silh.mean() / (silh.std() + 1e-10)

        if score > best_score:
            best_score  = score
            best_labels = km.labels_.copy()
            best_k      = k
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= 2:
                break

    return best_labels, best_k


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """
    Minimum variance portfolio weights.
    ω* = V⁻¹ · 1 / (1' · V⁻¹ · 1)
    Uses pseudo-inverse for robustness in near-singular cases.
    """
    ones = np.ones(cov.shape[0])
    try:
        w = np.linalg.solve(cov, ones)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(cov, ones, rcond=None)[0]
    w = w / w.sum()
    return w


def _risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """
    Inverse-volatility weights across clusters.

    At the inter-cluster level the K cluster portfolios are already
    nearly uncorrelated (NCO property), so inverse-vol converges to
    the risk-parity solution without requiring iterative optimisation.
    """
    vol = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    w   = (1.0 / vol) / (1.0 / vol).sum()
    return w


def nco_block(
    cov: np.ndarray,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    kmeans_seed: int  = 0,
) -> np.ndarray:
    """
    NCO algorithm on a single block covariance matrix.

    Parameters
    ----------
    cov          : (n × n) covariance matrix (Ledoit-Wolf + de-noised)
    max_clusters : upper bound on K for silhouette search
    kmeans_seed  : seed for K-Means reproducibility within this call

    Returns
    -------
    weights : (n,) array summing to 1.0
    k       : number of clusters chosen
    """
    n    = cov.shape[0]
    corr = cov2corr(cov)

    # ── Step 1: cluster assets within block ───────────────────────────────────
    labels, k = _cluster_kmeans(corr, max_clusters, rng_seed=kmeans_seed)
    clusters  = {c: np.where(labels == c)[0] for c in range(k)}

    # ── Step 2: intra-cluster min variance weights ────────────────────────────
    w_intra = np.zeros((n, k))

    for c_idx, asset_idx in clusters.items():
        if len(asset_idx) == 1:
            # Singleton cluster — full weight on that asset
            w_intra[asset_idx[0], c_idx] = 1.0
        else:
            cov_c = cov[np.ix_(asset_idx, asset_idx)]
            w_c   = _min_variance_weights(cov_c)
            w_intra[asset_idx, c_idx] = w_c

    # ── Step 3: collapse to K×K covariance ───────────────────────────────────
    # V_reduced = W_intra' · V · W_intra
    cov_reduced = w_intra.T @ cov @ w_intra

    # ── Step 4: inter-cluster risk parity weights ─────────────────────────────
    w_inter = _risk_parity_weights(cov_reduced)   # (K,)

    # ── Step 5: final weights = intra · inter ────────────────────────────────
    # w_intra : (n × K),  w_inter : (K,)
    weights = (w_intra * w_inter[None, :]).sum(axis=1)
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()

    return weights, k


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Stratified random partitioning
# ══════════════════════════════════════════════════════════════════════════════

def stratified_partition(
    tickers: list[str],
    asset_class_map: dict[str, str],
    n_blocks: int,
    rng: np.random.Generator,
) -> list[list[str]]:
    """
    Partition `tickers` into `n_blocks` blocks such that each block
    contains a proportional cross-section of every asset class.

    Assets within each class are shuffled randomly before assignment.
    Remaining assets (when N is not divisible by n_blocks) are
    distributed round-robin across the first few blocks.

    Returns
    -------
    blocks : list of n_blocks lists, each containing ticker strings
    """
    # Group tickers by asset class
    class_groups: dict[str, list[str]] = {}
    for t in tickers:
        cls = asset_class_map.get(t, "unknown")
        class_groups.setdefault(cls, []).append(t)

    # Shuffle within each class
    for cls in class_groups:
        arr = class_groups[cls]
        rng.shuffle(arr)

    # Initialise empty blocks
    blocks: list[list[str]] = [[] for _ in range(n_blocks)]

    # Distribute each class proportionally across blocks
    for cls, members in class_groups.items():
        n_cls      = len(members)
        base_count = n_cls // n_blocks
        remainder  = n_cls % n_blocks

        pos = 0
        for b in range(n_blocks):
            count      = base_count + (1 if b < remainder else 0)
            blocks[b] += members[pos : pos + count]
            pos        += count

    # Final shuffle within each block (so asset class ordering isn't
    # preserved within the block, which could bias K-Means)
    for b in blocks:
        rng.shuffle(b)

    return blocks


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Parallel workers
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_randomisation(
    m_idx: int,
    returns_np: np.ndarray,        # (T × N) float64, no NaNs
    tickers: list,
    asset_class_map: dict,
    n_blocks: int,
    max_clusters: int,
    base_seed: int,
) -> np.ndarray:
    """
    Module-level worker for joblib. Accepts numpy arrays so loky can
    serialise arguments cheaply across processes.

    Returns (N,) weight array summing to 1.0.
    """
    rng = np.random.default_rng(base_seed + m_idx)
    T, N = returns_np.shape

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    blocks        = stratified_partition(tickers, asset_class_map, n_blocks, rng)

    all_weights = np.zeros(N)

    for b_idx, block_tickers in enumerate(blocks):
        n_b = len(block_tickers)
        if n_b < 2:
            for t in block_tickers:
                all_weights[ticker_to_idx[t]] = 1.0 / n_blocks
            continue

        block_idx = [ticker_to_idx[t] for t in block_tickers]
        X_b       = returns_np[:, block_idx]          # (T × n_b)
        q_b       = T / n_b

        cov_b            = ledoit_wolf_cov(X_b)
        cov_b, _         = denoise_cov(cov_b, q=q_b)
        w_b, _           = nco_block(
            cov_b,
            max_clusters = max_clusters,
            kmeans_seed  = m_idx * 1000 + b_idx,
        )

        for local_i, t in enumerate(block_tickers):
            all_weights[ticker_to_idx[t]] = w_b[local_i] / n_blocks

    total = all_weights.sum()
    if total > 1e-10:
        all_weights = all_weights / total

    return all_weights


def _run_one_window(
    returns_np: np.ndarray,
    tickers: list[str],
    asset_class_map: dict[str, str],
    n_blocks: int,
    n_randomisations: int,
    max_clusters: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Module-level worker: runs all M randomisations for one rebalance window
    and returns ensemble-averaged weights + diagnostics.

    Returns (w_mean, w_std, w_cv) each of shape (N,).
    """
    N = len(tickers)
    weight_runs = np.empty((n_randomisations, N))

    for m in range(n_randomisations):
        weight_runs[m] = _run_one_randomisation(
            m, returns_np, tickers, asset_class_map,
            n_blocks, max_clusters, base_seed,
        )

    w_mean = weight_runs.mean(axis=0)
    total  = w_mean.sum()
    if total > 1e-10:
        w_mean /= total
    w_std = weight_runs.std(axis=0)
    w_cv  = np.full_like(w_mean, np.nan)
    mask  = w_mean > 1e-10
    w_cv[mask] = w_std[mask] / w_mean[mask]

    return w_mean, w_std, w_cv


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main — iterate over rebalance schedule
# ══════════════════════════════════════════════════════════════════════════════

def main(
    run_id:            Optional[int]   = None,
    lookbacks:         list[str]       = DEFAULT_LOOKBACKS,
    n_blocks:          int             = DEFAULT_N_BLOCKS,
    n_randomisations:  int             = DEFAULT_N_RANDOMISATIONS,
    max_clusters:      int             = DEFAULT_MAX_CLUSTERS,
    seed:              int             = DEFAULT_SEED,
):
    registry = RunRegistry()
    con      = get_connection()

    # ── Resolve run_id ────────────────────────────────────────────────────────
    if run_id is None:
        row = con.execute("""
            SELECT run_id FROM runs
            WHERE json_extract_string(labels, '$.type') = 'data_pull'
            ORDER BY created_at DESC
            LIMIT 1
        """).fetchone()
        if row is None:
            raise RuntimeError("No data_pull run found.")
        run_id = row[0]

    print(f"\n── NCO Engine ───────────────────────────────────────────────────────")
    print(f"  run_id           : {run_id}")
    print(f"  lookbacks        : {lookbacks}")
    print(f"  n_blocks (B)     : {n_blocks}")
    print(f"  n_randomisations : {n_randomisations}")
    print(f"  max_clusters (K) : {max_clusters}")
    print(f"  seed             : {seed}")

    # ── Load returns (wide format) ────────────────────────────────────────────

    print(f"\n  Loading returns...", flush=True)

    # Market returns (sparsified)
    mkt = con.execute("""
        SELECT date, ticker, log_return
        FROM market_returns_sparse
        WHERE run_id = ?
    """, [run_id]).df()

    mkt_wide = (
        mkt.pivot(index="date", columns="ticker", values="log_return")
           .sort_index()
    )
    mkt_wide.index = pd.to_datetime(mkt_wide.index)

    # Algo returns
    algo = con.execute("""
        SELECT date, algo AS ticker, log_return
        FROM algo_returns
        WHERE run_id = ?
    """, [run_id]).df()

    algo_wide = (
        algo.pivot(index="date", columns="ticker", values="log_return")
            .sort_index()
    )
    algo_wide.index = pd.to_datetime(algo_wide.index)

    # Combined returns panel
    returns_all = pd.concat([mkt_wide, algo_wide], axis=1).sort_index()

    # Asset class map
    ticker_meta = con.execute("""
        SELECT ticker, asset_class FROM ticker_meta WHERE run_id = ?
    """, [run_id]).df()

    asset_class_map = dict(zip(ticker_meta["ticker"], ticker_meta["asset_class"]))
    for t in algo_wide.columns:
        asset_class_map[t] = "algo"

    tickers_all = returns_all.columns.tolist()
    N           = len(tickers_all)
    print(f"  N total          : {N}  ({mkt_wide.shape[1]} market + {algo_wide.shape[1]} algo)")

    # ── Load rebalance schedule ───────────────────────────────────────────────

    schedule = con.execute("""
        SELECT DISTINCT lookback, rebalance_date, window_start, window_end, n_days
        FROM rebalance_schedule
        WHERE run_id = ?
        ORDER BY lookback, rebalance_date
    """, [run_id]).df()

    schedule = schedule[schedule["lookback"].isin(lookbacks)].copy()
    schedule["rebalance_date"] = pd.to_datetime(schedule["rebalance_date"])
    schedule["window_start"]   = pd.to_datetime(schedule["window_start"])
    schedule["window_end"]     = pd.to_datetime(schedule["window_end"])

    total_rebs = len(schedule)
    print(f"  Total rebalances : {total_rebs}  across {len(lookbacks)} windows")

    # ── NCO run ───────────────────────────────────────────────────────────────

    nco_run_id = registry.start_run(
        type             = "nco",
        parent_run_id    = run_id,
        n_blocks         = n_blocks,
        n_randomisations = n_randomisations,
        max_clusters     = max_clusters,
        seed             = seed,
        lookbacks        = ",".join(lookbacks),
        notes            = "Stratified random block NCO with ensemble averaging",
    )

    all_weight_rows  = []
    all_summary_rows = []

    t0 = time.time()

    # ── Pre-slice all windows and build job list ──────────────────────────────
    jobs     = []
    job_meta = []  # parallel list: (lb, reb_dt, tickers, active_class_map, T, N, n_dropped)

    for _, row in schedule.iterrows():
        lb, reb_dt = row["lookback"], row["rebalance_date"]
        mask = (
            (returns_all.index >= row["window_start"]) &
            (returns_all.index <= row["window_end"])
        )
        R = returns_all.loc[mask]

        active_cols = R.columns[R.notna().any(axis=0)]
        n_dropped   = len(R.columns) - len(active_cols)
        R           = R[active_cols].fillna(0.0)
        T_w, N_w    = R.shape

        if T_w < 20:
            print(f"  SKIP {lb} {reb_dt.date()} — only {T_w} days in window")
            continue

        active_class_map = {t: asset_class_map.get(t, "unknown") for t in active_cols}
        jobs.append(delayed(_run_one_window)(
            R.to_numpy(dtype=np.float64),
            active_cols.tolist(),
            active_class_map,
            n_blocks, n_randomisations, max_clusters, seed,
        ))
        job_meta.append((lb, reb_dt, active_cols.tolist(), active_class_map, T_w, N_w, n_dropped))

    print(f"  Dispatching {len(jobs)} windows × {n_randomisations} randomisations...", flush=True)

    raw_results = Parallel(n_jobs=-1, backend="loky")(
        tqdm(jobs, total=len(jobs), desc="rebalances")
    )

    # ── Collect results ───────────────────────────────────────────────────────
    for (lb, reb_dt, tickers_w, active_class_map, T_w, N_w, n_dropped), (w_mean, w_std, w_cv) in zip(job_meta, raw_results):

        for ticker, wm, ws, wc in zip(tickers_w, w_mean, w_std, w_cv):
            all_weight_rows.append({
                "nco_run_id":     nco_run_id,
                "lookback":       lb,
                "rebalance_date": reb_dt,
                "ticker":         ticker,
                "weight":         float(wm),
                "weight_std":     float(ws),
                "weight_cv":      float(wc),
                "asset_class":    active_class_map.get(ticker, "unknown"),
            })

        all_summary_rows.append({
            "nco_run_id":     nco_run_id,
            "lookback":       lb,
            "rebalance_date": reb_dt,
            "n_active":       N_w,
            "n_dropped":      n_dropped,
            "t_days":         T_w,
            "t_over_n":       round(T_w / N_w, 3),
            "weight_sum":     round(float(w_mean.sum()), 6),
            "mean_cv":        round(float(np.nanmean(w_cv)), 4),
            "max_weight":     round(float(w_mean.max()), 6),
            "min_weight":     round(float(w_mean.min()), 6),
        })

    # ── Store to DuckDB ───────────────────────────────────────────────────────

    print(f"\n── Storing results ──────────────────────────────────────────────────")

    weights_store = pd.DataFrame(all_weight_rows)
    summary_store = pd.DataFrame(all_summary_rows)

    registry.log_dataframe(nco_run_id, table="nco_weights",  df=weights_store)
    registry.log_dataframe(nco_run_id, table="nco_run_summary", df=summary_store)

    registry.log_metrics(
        nco_run_id,
        total_rebalances  = len(all_summary_rows),
        mean_t_over_n     = round(summary_store["t_over_n"].mean(), 3),
        mean_cv_overall   = round(summary_store["mean_cv"].mean(), 4),
        total_weight_rows = len(weights_store),
    )

    registry.end_run(nco_run_id)

    total_time = time.time() - t0
    print(f"  nco_run_id       : {nco_run_id}")
    print(f"  Weight rows      : {len(weights_store):,}")
    print(f"  Summary rows     : {len(summary_store)}")
    print(f"  Total time       : {total_time/60:.1f} min")
    print(f"\n── Complete ─────────────────────────────────────────────────────────")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",            type=int,   default=None)
    parser.add_argument("--lookback",          type=str,   default="6m,4m,3m",
                        help="Comma-separated lookback labels e.g. '6m,4m'")
    parser.add_argument("--n-randomisations",  type=int,   default=DEFAULT_N_RANDOMISATIONS)
    parser.add_argument("--n-blocks",          type=int,   default=DEFAULT_N_BLOCKS)
    parser.add_argument("--max-clusters",      type=int,   default=DEFAULT_MAX_CLUSTERS)
    parser.add_argument("--seed",              type=int,   default=DEFAULT_SEED)
    args = parser.parse_args()

    main(
        run_id           = args.run_id,
        lookbacks        = args.run_id and args.lookback.split(",") or DEFAULT_LOOKBACKS,
        n_blocks         = args.n_blocks,
        n_randomisations = args.n_randomisations,
        max_clusters     = args.max_clusters,
        seed             = args.seed,
    )
