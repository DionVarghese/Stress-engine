"""
run_stress_test.py
==================
Phase 1 (default): Diagnostic plots — one subplot per anomaly type.

  Univariate / Regime / Pattern : equity curve  (original vs perturbed)
  Cross-curve                   : rolling 20-day pairwise correlation
                                  between the 2 most affected assets

Phase 2 (--run-model): Run model_fn wrappers against the full scenario bank.

Usage
-----
python scripts/run_stress_test.py
    [--engine-run-id <uuid>]
    [--out-dir plots/stress_test]
    [--single-type <anomaly_type>]
    [--run-model]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from quant.utils.db import RunRegistry, get_connection
from scenario_engine import ANOMALY_REGISTRY, LAYER_ORDER


# ── Layout ────────────────────────────────────────────────────────────────────

PLOT_ORDER = [
    "vol_spike",           "vol_cluster_burst",    "persistent_vol_shift", "artificial_drawdown",
    "drift_injection",     "trend_reversal",       "heavy_tail_sub",       "ar1_injection",
    "merton_jump",         "decorrelation",        "contagion",            "sync_drawdown_recovery",
    "vol_regime_swap",     "drawdown_recovery_var","regime_persistence",   "oscillating_pattern",
]

PLOT_GRID_ROWS = 4
PLOT_GRID_COLS = 4

LAYER_COLOURS = {
    "univariate":  "#2196F3",
    "cross_curve": "#FF9800",
    "regime":      "#9C27B0",
    "pattern":     "#4CAF50",
}

# Which plot type to use per layer.
# Pattern uses equity (cumulative), not daily-return bars: a smooth
# sinusoidal/persistent-drift mean is visually indistinguishable from noisy
# alternating bars in a daily-return view, but shows up clearly as a wave or
# trend once cumulated.
LAYER_PLOT_TYPE = {
    "univariate":  "equity",
    "regime":      "equity",
    "cross_curve": "correlation",
    "pattern":     "equity",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_engine_run_id(con, engine_run_id: str | None) -> str:
    if engine_run_id:
        return engine_run_id
    row = con.execute("""
        SELECT run_id FROM runs
        WHERE json_extract_string(labels, '$.type') = 'scenario_engine'
          AND status = 'done'
        ORDER BY created_at DESC LIMIT 1
    """).fetchone()
    if row is None:
        raise RuntimeError("No completed scenario_engine run found.")
    return row[0]


def _get_data_run_id(con, engine_run_id: str) -> str:
    row = con.execute(
        "SELECT json_extract_string(labels, '$.parent_run_id') FROM runs WHERE run_id = ?",
        [engine_run_id]
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(f"No parent_run_id for engine run {engine_run_id}")
    return row[0]


def _load_returns(con, data_run_id: str) -> pd.DataFrame:
    mkt = con.execute(
        "SELECT date, ticker, log_return FROM market_returns_sparse WHERE run_id = ?",
        [data_run_id]
    ).df()
    mkt_wide = (mkt.pivot(index="date", columns="ticker", values="log_return")
                   .sort_index())
    mkt_wide.index = pd.to_datetime(mkt_wide.index)

    algo = con.execute(
        "SELECT date, algo AS ticker, log_return FROM algo_returns WHERE run_id = ?",
        [data_run_id]
    ).df()
    algo_wide = (algo.pivot(index="date", columns="ticker", values="log_return")
                    .sort_index())
    algo_wide.index = pd.to_datetime(algo_wide.index)

    return pd.concat([mkt_wide, algo_wide], axis=1).sort_index().fillna(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO QUERIES
# ══════════════════════════════════════════════════════════════════════════════

_CROSS_CURVE_TYPES = {"decorrelation", "contagion", "sync_drawdown_recovery"}


def _most_affected_scenario(con, anomaly_type: str,
                             engine_run_id: str) -> dict | None:
    # Prefer single-anomaly scenarios so the stored delta is purely from
    # this anomaly type. Multi-anomaly deltas include contributions from
    # other injections which compress the y-axis and hide the pattern.
    #
    # For cross-curve types the correlation plot needs ≥2 snapshot-bearing
    # curves, so add a HAVING guard.
    #
    # Scoped to one engine_run_id: scenario_snapshots/anomaly_instances
    # accumulate across every past run (CREATE TABLE IF NOT EXISTS never
    # clears old rows), so without this filter the pick would come from the
    # table's entire history, not this run.
    #
    # Picks the example closest to the *median* impact for this type, not
    # the single most extreme one. This overview grid is meant to show what
    # a typical injection looks like; ORDER BY max_impact DESC always landed
    # on the ~99.9th-percentile tail case, which misrepresents calibration
    # even when the bulk of generated scenarios are reasonable. Extreme
    # examples are still available via --single-type, which intentionally
    # ranks by impact for edge-case review.
    having = "HAVING COUNT(DISTINCT aac.curve_idx) >= 2" if anomaly_type in _CROSS_CURVE_TYPES else ""
    df = con.execute(f"""
        WITH candidates AS (
            SELECT ai.anomaly_id, ai.scenario_id, ai.window_start, ai.window_end,
                   ai.params, MAX(ABS(ss.vol_ratio - 1)) AS max_impact,
                   s.n_anomalies
            FROM anomaly_instances       ai
            JOIN anomaly_affected_curves aac ON aac.anomaly_id  = ai.anomaly_id
            JOIN scenario_snapshots      ss  ON ss.scenario_id  = ai.scenario_id
                                            AND ss.curve_idx    = aac.curve_idx
            JOIN scenarios               s   ON s.scenario_id   = ai.scenario_id
            WHERE ai.anomaly_type = ?
              AND ss.original_vol > 1e-4
              AND s.engine_run_id = ?
            GROUP BY ai.anomaly_id, ai.scenario_id, ai.window_start, ai.window_end,
                     ai.params, s.n_anomalies
            {having}
        )
        SELECT c.*
        FROM candidates c, (SELECT MEDIAN(max_impact) AS med FROM candidates) m
        ORDER BY c.n_anomalies ASC, ABS(c.max_impact - m.med) ASC
        LIMIT 1
    """, [anomaly_type, engine_run_id]).df()
    if df.empty:
        return None
    r = df.iloc[0]
    return {"anomaly_id":   r["anomaly_id"],
            "scenario_id":  r["scenario_id"],
            "window_start": int(r["window_start"]),
            "window_end":   int(r["window_end"]),
            "params":       r["params"],
            "max_impact":   float(r["max_impact"])}


def _most_affected_ticker(con, scenario_id: str, anomaly_id: str) -> dict | None:
    df = con.execute("""
        SELECT ss.ticker, ss.curve_idx, ss.vol_ratio
        FROM scenario_snapshots      ss
        JOIN anomaly_affected_curves aac ON aac.curve_idx = ss.curve_idx
        WHERE ss.scenario_id = ?
          AND aac.anomaly_id = ?
          AND ss.original_vol > 1e-4
        ORDER BY ABS(ss.vol_ratio - 1) DESC
        LIMIT 1
    """, [scenario_id, anomaly_id]).df()
    return None if df.empty else df.iloc[0].to_dict()


def _two_most_affected_curves(con, scenario_id: str,
                               anomaly_type: str) -> pd.DataFrame:
    """For cross_curve anomalies: the 2 curves with highest |vol_ratio-1|
    that were both affected by the same anomaly instance and have snapshots."""
    return con.execute("""
        SELECT aac.curve_idx, ss.ticker, ss.vol_ratio
        FROM anomaly_affected_curves aac
        JOIN anomaly_instances   ai ON aac.anomaly_id  = ai.anomaly_id
        JOIN scenario_snapshots  ss ON ss.scenario_id  = ai.scenario_id
                                   AND ss.curve_idx    = aac.curve_idx
        WHERE ai.scenario_id  = ?
          AND ai.anomaly_type  = ?
          AND ss.ticker IS NOT NULL
        ORDER BY ABS(ss.vol_ratio - 1) DESC
        LIMIT 2
    """, [scenario_id, anomaly_type]).df()


def _load_curve_delta(con, scenario_id: str, curve_idx: int) -> dict:
    df = con.execute("""
        SELECT t_idx, delta_value
        FROM scenario_deltas
        WHERE scenario_id = ? AND curve_idx = ?
    """, [scenario_id, int(curve_idx)]).df()
    return dict(zip(df["t_idx"].astype(int), df["delta_value"].astype(float)))


# ══════════════════════════════════════════════════════════════════════════════
# CURVE RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _reconstruct(returns_np, delta, col_idx, start, end):
    """Slice [start:end] of one column, apply sparse delta, return (orig, pert)."""
    orig = returns_np[start:end, col_idx].copy()
    pert = orig.copy()
    for t, dv in delta.items():
        if start <= t < end:
            pert[t - start] += dv
    return orig, pert


def _equity_curves(orig, pert):
    return np.cumprod(1 + orig), np.cumprod(1 + pert)


def _rolling_corr(a: np.ndarray, b: np.ndarray, window: int = 15) -> np.ndarray:
    n = len(a)
    corrs = np.full(n, np.nan)
    for i in range(window - 1, n):
        sa, sb = a[i-window+1:i+1], b[i-window+1:i+1]
        if np.std(sa) > 1e-8 and np.std(sb) > 1e-8:
            corrs[i] = float(np.corrcoef(sa, sb)[0, 1])
    return corrs


# ══════════════════════════════════════════════════════════════════════════════
# PER-SUBPLOT DRAWERS
# ══════════════════════════════════════════════════════════════════════════════

def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


def _shade_window(ax, w_start, w_end, p_start, colour):
    s = w_start - p_start
    e = w_end   - p_start
    ax.axvspan(s, e, alpha=0.15, color=colour, lw=0)
    ax.axvline(s, color=colour, lw=0.7, ls=":")
    ax.axvline(e, color=colour, lw=0.7, ls=":")


def _draw_equity(ax, con, returns_np, ticker_idx, T,
                 meta, snap, colour,
                 buf_pre=10, buf_post=10):
    """Equity curve: original (solid blue) vs perturbed (dashed red)."""
    col = ticker_idx.get(snap["ticker"])
    if col is None:
        ax.text(0.5, 0.5, f"ticker missing", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="grey")
        return

    p0 = max(0, meta["window_start"] - buf_pre)
    p1 = min(T, meta["window_end"]   + buf_post)

    delta         = _load_curve_delta(con, meta["scenario_id"], int(snap["curve_idx"]))
    orig, pert    = _reconstruct(returns_np, delta, col, p0, p1)
    cum_o, cum_p  = _equity_curves(orig, pert)
    x             = np.arange(p1 - p0)

    ax.plot(x, cum_o, lw=1.4, color="#1565C0", label="original")
    ax.plot(x, cum_p, lw=1.4, color="#C62828", ls="--", label="perturbed")
    ax.axhline(1.0, color="grey", lw=0.5, ls=":")
    _shade_window(ax, meta["window_start"], meta["window_end"], p0, colour)
    ax.set_xlabel("trading days", fontsize=6)
    ax.set_ylabel("cumulative return", fontsize=6)


def _draw_correlation(ax, con, returns_np, ticker_idx, T,
                      meta, anomaly_type, colour,
                      buf_pre=20, buf_post=15, roll_win=15):
    """
    Rolling pairwise correlation between the 2 most affected curves.
    Shows how the cross_curve anomaly changes the correlation structure —
    something an equity curve of a single asset cannot reveal.
    """
    curves_df = _two_most_affected_curves(con, meta["scenario_id"], anomaly_type)
    if len(curves_df) < 2:
        ax.text(0.5, 0.5, "need ≥2 affected\ncurves in snapshots",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="grey")
        return

    p0 = max(0, meta["window_start"] - buf_pre)
    p1 = min(T, meta["window_end"]   + buf_post)

    col_a = ticker_idx.get(str(curves_df.iloc[0]["ticker"]))
    col_b = ticker_idx.get(str(curves_df.iloc[1]["ticker"]))
    if col_a is None or col_b is None:
        ax.text(0.5, 0.5, "tickers missing", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="grey")
        return

    delta_a = _load_curve_delta(con, meta["scenario_id"], int(curves_df.iloc[0]["curve_idx"]))
    delta_b = _load_curve_delta(con, meta["scenario_id"], int(curves_df.iloc[1]["curve_idx"]))

    orig_a, pert_a = _reconstruct(returns_np, delta_a, col_a, p0, p1)
    orig_b, pert_b = _reconstruct(returns_np, delta_b, col_b, p0, p1)

    corr_orig = _rolling_corr(orig_a, orig_b, roll_win)
    corr_pert = _rolling_corr(pert_a, pert_b, roll_win)
    x = np.arange(p1 - p0)

    ax.plot(x, corr_orig, lw=1.4, color="#1565C0", label="original")
    ax.plot(x, corr_pert, lw=1.4, color="#C62828", ls="--", label="perturbed")
    ax.axhline(0.0, color="grey", lw=0.5, ls=":")
    ax.set_ylim(-1.05, 1.05)
    _shade_window(ax, meta["window_start"], meta["window_end"], p0, colour)

    t_a = curves_df.iloc[0]["ticker"]
    t_b = curves_df.iloc[1]["ticker"]
    ax.set_xlabel("trading days", fontsize=6)
    ax.set_ylabel(f"{roll_win}d rolling corr\n{t_a} vs {t_b}", fontsize=5.5)


def _draw_daily_returns(ax, con, returns_np, ticker_idx, T,
                        meta, snap, colour,
                        buf_pre=5, buf_post=5):
    """
    Daily return bar chart inside the injection window ± small buffer.
    The alternation / bimodal shape that equity curves cannot show is
    immediately visible in individual bar heights.
    """
    col = ticker_idx.get(snap["ticker"])
    if col is None:
        ax.text(0.5, 0.5, "ticker missing", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="grey")
        return

    p0 = max(0, meta["window_start"] - buf_pre)
    p1 = min(T, meta["window_end"]   + buf_post)

    delta      = _load_curve_delta(con, meta["scenario_id"], int(snap["curve_idx"]))
    orig, pert = _reconstruct(returns_np, delta, col, p0, p1)
    x          = np.arange(p1 - p0)
    width      = 0.4

    # Original bars (blue)
    ax.bar(x - width/2, orig, width=width, color="#1565C0",
           alpha=0.7, label="original")
    # Perturbed bars (red)
    ax.bar(x + width/2, pert, width=width, color="#C62828",
           alpha=0.7, label="perturbed")

    ax.axhline(0.0, color="grey", lw=0.5)
    _shade_window(ax, meta["window_start"], meta["window_end"], p0, colour)
    ax.set_xlabel("trading days", fontsize=6)
    ax.set_ylabel("daily log-return", fontsize=6)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN OVERVIEW GRID
# ══════════════════════════════════════════════════════════════════════════════

def plot_anomaly_diagnostics(con, returns: pd.DataFrame, out_path: str,
                              engine_run_id: str):
    """
    Grid — one subplot per anomaly type.

    Univariate / Regime / Pattern → equity curve
    Cross-curve                  → rolling 20-day pairwise correlation
    """
    returns_np = returns.to_numpy().astype(np.float64)
    ticker_idx = {t: i for i, t in enumerate(returns.columns)}
    T          = returns_np.shape[0]

    fig, axes = plt.subplots(PLOT_GRID_ROWS, PLOT_GRID_COLS,
                             figsize=(22, 4 * PLOT_GRID_ROWS))
    fig.subplots_adjust(hspace=0.60, wspace=0.38)
    fig.suptitle(
        "Scenario Engine — Original vs Perturbed\n"
        "(equity curve | rolling correlation | daily returns  —  per anomaly type)",
        fontsize=11, fontweight="bold", y=0.99,
    )

    for ax_idx, anomaly_type in enumerate(PLOT_ORDER):
        ax        = axes[ax_idx // PLOT_GRID_COLS][ax_idx % PLOT_GRID_COLS]
        layer     = ANOMALY_REGISTRY[anomaly_type].layer
        colour    = LAYER_COLOURS.get(layer, "#999999")
        plot_type = LAYER_PLOT_TYPE[layer]

        meta = _most_affected_scenario(con, anomaly_type, engine_run_id)
        if meta is None:
            ax.text(0.5, 0.5, "no scenarios", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="grey")
            ax.set_title(f"{anomaly_type}\n[{layer}]", fontsize=8,
                         color=colour, fontweight="bold")
            _clean(ax)
            continue

        snap = _most_affected_ticker(con, meta["scenario_id"], meta["anomaly_id"])

        # ── Choose visualisation by layer ─────────────────────────────────────
        if plot_type == "equity":
            if snap is None:
                ax.text(0.5, 0.5, "no snapshot", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="grey")
            else:
                _draw_equity(ax, con, returns_np, ticker_idx, T,
                             meta, snap, colour)
                vr = float(snap["vol_ratio"])
                ax.set_title(
                    f"{anomaly_type}\n[{layer}]  {snap['ticker']}  vol×{vr:.2f}",
                    fontsize=7.5, color=colour, fontweight="bold"
                )

        elif plot_type == "correlation":
            _draw_correlation(ax, con, returns_np, ticker_idx, T,
                              meta, anomaly_type, colour)
            ax.set_title(
                f"{anomaly_type}\n[{layer}]  rolling corr",
                fontsize=7.5, color=colour, fontweight="bold"
            )

        elif plot_type == "daily_returns":
            if snap is None:
                ax.text(0.5, 0.5, "no snapshot", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="grey")
            else:
                _draw_daily_returns(ax, con, returns_np, ticker_idx, T,
                                    meta, snap, colour)
                vr = float(snap["vol_ratio"])
                ax.set_title(
                    f"{anomaly_type}\n[{layer}]  {snap['ticker']}  vol×{vr:.2f}",
                    fontsize=7.5, color=colour, fontweight="bold"
                )

        _clean(ax)

    # Hide any grid slots past the last anomaly type (grid size need not be
    # an exact multiple of len(PLOT_ORDER)).
    for empty_idx in range(len(PLOT_ORDER), PLOT_GRID_ROWS * PLOT_GRID_COLS):
        axes[empty_idx // PLOT_GRID_COLS][empty_idx % PLOT_GRID_COLS].axis("off")

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color="#1565C0", lw=1.4, label="original"),
        plt.Line2D([0], [0], color="#C62828", lw=1.4, ls="--",
                   label="perturbed / rolling corr"),
    ] + [
        mpatches.Patch(color=c, alpha=0.4,
                       label=f"{layer}  ({LAYER_PLOT_TYPE[layer]})")
        for layer, c in LAYER_COLOURS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.002))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-TYPE DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

def plot_single_anomaly_type(con, returns: pd.DataFrame,
                              anomaly_type: str, out_path: str,
                              engine_run_id: str, n_examples: int = 6):
    returns_np = returns.to_numpy().astype(np.float64)
    ticker_idx = {t: i for i, t in enumerate(returns.columns)}
    T          = returns_np.shape[0]
    layer      = ANOMALY_REGISTRY[anomaly_type].layer
    colour     = LAYER_COLOURS.get(layer, "#999999")
    plot_type  = LAYER_PLOT_TYPE[layer]

    # Scoped to one engine_run_id — see _most_affected_scenario for why.
    df = con.execute("""
        SELECT ai.anomaly_id, ai.scenario_id, ai.window_start, ai.window_end,
               MAX(ABS(ss.vol_ratio - 1)) AS max_impact,
               s.n_anomalies
        FROM anomaly_instances       ai
        JOIN anomaly_affected_curves aac ON aac.anomaly_id  = ai.anomaly_id
        JOIN scenario_snapshots      ss  ON ss.scenario_id  = ai.scenario_id
                                        AND ss.curve_idx    = aac.curve_idx
        JOIN scenarios               s   ON s.scenario_id   = ai.scenario_id
        WHERE ai.anomaly_type = ?
          AND ss.original_vol > 1e-4
          AND s.engine_run_id = ?
        GROUP BY ai.anomaly_id, ai.scenario_id, ai.window_start, ai.window_end,
                 s.n_anomalies
        ORDER BY s.n_anomalies ASC, max_impact DESC
        LIMIT ?
    """, [anomaly_type, engine_run_id, n_examples]).df()

    if df.empty:
        print(f"  No scenarios for {anomaly_type!r}.")
        return

    n    = len(df)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{anomaly_type}  [{layer}]  —  {plot_type}",
                 fontsize=12, fontweight="bold")

    for i, (_, row) in enumerate(df.iterrows()):
        ax   = axes[i]
        meta = {"anomaly_id":   row["anomaly_id"],
                "scenario_id":  row["scenario_id"],
                "window_start": int(row["window_start"]),
                "window_end":   int(row["window_end"]),
                "max_impact":   float(row["max_impact"])}

        if plot_type == "equity":
            snap = _most_affected_ticker(con, meta["scenario_id"], meta["anomaly_id"])
            if snap:
                _draw_equity(ax, con, returns_np, ticker_idx, T,
                             meta, snap, colour)
                ax.set_title(
                    f"#{i+1}  {snap['ticker']}  vol×{snap['vol_ratio']:.2f}\n"
                    f"impact={row['max_impact']:.3f}",
                    fontsize=8
                )
        elif plot_type == "correlation":
            _draw_correlation(ax, con, returns_np, ticker_idx, T,
                              meta, anomaly_type, colour)
            ax.set_title(f"#{i+1}  impact={row['max_impact']:.3f}", fontsize=8)
        elif plot_type == "daily_returns":
            snap = _most_affected_ticker(con, meta["scenario_id"], meta["anomaly_id"])
            if snap:
                _draw_daily_returns(ax, con, returns_np, ticker_idx, T,
                                    meta, snap, colour)
                ax.set_title(
                    f"#{i+1}  {snap['ticker']}  vol×{snap['vol_ratio']:.2f}",
                    fontsize=8
                )
        if i == 0:
            ax.legend(fontsize=7)
        _clean(ax)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL_FN WRAPPERS  (stubs — implement for Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

def make_nco_dwf_fn(dates_index, tickers, asset_class_map,
                    w1_days, w2_days, min_sharpe_w2, min_sortino_w2,
                    min_sharpe_w1, min_sortino_w1, nco_days, seed):
    """
    Factory for the NCO dual-window filter model_fn.
    (perturbed, original, rebalance_dates) → dict[metric, float]
    TODO: implement for Phase 2.
    """
    def model_fn(perturbed, original, rebalance_dates):
        raise NotImplementedError("make_nco_dwf_fn stub — implement for Phase 2.")
    return model_fn


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(engine_run_id: str = None, out_dir: str = "plots/stress_test",
         run_model: bool = False, single_type: str = None):

    con           = get_connection()
    engine_run_id = _resolve_engine_run_id(con, engine_run_id)
    data_run_id   = _get_data_run_id(con, engine_run_id)

    print(f"\n── Stress Test ──────────────────────────────────────────────────")
    print(f"  engine_run_id : {engine_run_id}")
    print(f"  data_run_id   : {data_run_id}")
    print(f"  out_dir       : {out_dir}")

    print("\n  Loading returns...", flush=True)
    returns = _load_returns(con, data_run_id)
    print(f"  Shape : {returns.shape}")

    if single_type:
        out_path = os.path.join(out_dir, f"anomaly_{single_type}.png")
        print(f"\n  Plotting {single_type} deep-dive...")
        plot_single_anomaly_type(con, returns, single_type, out_path, engine_run_id)
    else:
        out_path = os.path.join(out_dir, "00_anomaly_overview.png")
        print(f"\n  Plotting {PLOT_GRID_ROWS}x{PLOT_GRID_COLS} overview...")
        plot_anomaly_diagnostics(con, returns, out_path, engine_run_id)

    if run_model:
        print("\n  Phase 2 — implement make_nco_dwf_fn() first.")

    print("\n── Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-run-id", type=str, default=None)
    parser.add_argument("--out-dir",       type=str, default="plots/stress_test")
    parser.add_argument("--run-model",     action="store_true")
    parser.add_argument("--single-type",   type=str, default=None)
    args = parser.parse_args()
    main(engine_run_id=args.engine_run_id, out_dir=args.out_dir,
         run_model=args.run_model, single_type=args.single_type)
