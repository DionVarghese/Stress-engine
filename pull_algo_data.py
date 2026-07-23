"""
pull_algo_data.py
=================
Ingest raw MT5-style backtest deal logs into DuckDB as the per-curve daily
return panel the scenario engine consumes.

One CSV = one curve. Filenames follow  <NNN>_<SYMBOL>_<TF>.csv  (e.g.
179_EURJPY_H1.csv); the stem becomes the curve `key` and the suffix the
timeframe `tf` bucket used for calibration.

Equity / return convention
--------------------------
Each row is one closed deal leg. The running `balance` obeys exactly

    balance_t = balance_{t-1} + profit_t + commission_t + swap_t

(verified against the raw files — zero mismatches). The per-leg PnL is
therefore  pnl = profit + commission + swap.  We aggregate PnL to the calendar
day and express it as a FIXED-NOTIONAL arithmetic return

    ret_day = (Σ pnl on that day) / E_0 ,   E_0 = starting equity (≈100_000)

so that  1 + cumsum(ret)  reconstructs the normalised equity curve. This is the
convention scenario_engine / run_stress_test expect (NOT log-return-on-equity,
which is ill-defined for accounts that can cross zero).

Writes (all stamped with a fresh `algo_data_pull` run_id):
    algo_returns_daily(run_id, "Date", key, ret)
    algo_meta(run_id, key, tf, symbol, e0, n_days, first_day, last_day)

Usage
-----
python pull_algo_data.py
    [--data-dir <path to BacktestData>]
    [--reset]        drop existing panel tables first (clean rebuild)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from quant.utils.db import RunRegistry, get_connection


DEFAULT_DATA_DIR = (
    r"C:\Users\Dionv\Downloads\RawDataToLucsFormat\RawDataToLucsFormat\BacktestData"
)

VALID_TF = {"M15", "H1", "H4"}


# ── parsing ─────────────────────────────────────────────────────────────────

def _parse_key(stem: str) -> tuple[str, str]:
    """(<NNN>_<SYMBOL>_<TF>) -> (tf, symbol). Symbol may itself contain '_'."""
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename shape: {stem!r}")
    tf = parts[-1]
    symbol = "_".join(parts[1:-1])
    return tf, symbol


def _daily_returns(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read one deal log → (DataFrame[Date, ret], meta dict) for that curve."""
    df = pd.read_csv(
        path, sep=";",
        usecols=["time", "commission", "swap", "profit", "balance"],
    )
    # MT5 timestamps: 'YYYY.MM.DD HH:MM:SS'
    ts = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M:%S")
    for col in ("commission", "swap", "profit", "balance"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    pnl = df["profit"] + df["commission"] + df["swap"]

    # Starting equity, inferred from the first leg: E_0 = balance_0 - pnl_0.
    e0 = float(df["balance"].iloc[0] - pnl.iloc[0])
    if e0 <= 0:
        e0 = 100_000.0  # degenerate fallback

    day = ts.dt.normalize()
    daily_pnl = pnl.groupby(day).sum()
    ret = (daily_pnl / e0).rename("ret")

    out = ret.reset_index()
    out.columns = ["Date", "ret"]
    out["Date"] = out["Date"].dt.date

    meta = {
        "e0": e0,
        "n_days": int(len(out)),
        "first_day": out["Date"].iloc[0],
        "last_day": out["Date"].iloc[-1],
    }
    return out, meta


# ── quick-look plotting ──────────────────────────────────────────────────────

def _latest_run_id(con):
    row = con.execute("""
        SELECT run_id FROM runs
        WHERE json_extract_string(labels, '$.type') = 'algo_data_pull'
        ORDER BY created_at DESC LIMIT 1
    """).fetchone()
    return row[0] if row else None


def plot_sample_equity(con, run_id: str, n: int, out_path: str,
                       seed: int = 0) -> None:
    """Overlay a random sample of normalised equity curves (1 + Σret) so you can
    eyeball what the ingested fund data looks like. Fixed-notional, so all curves
    start at 1.0 and are directly comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = con.execute('SELECT key FROM algo_meta WHERE run_id = ? ORDER BY key',
                       [run_id]).df()
    if meta.empty:
        print("  No curves to plot.")
        return

    keys = list(meta["key"])
    rng  = np.random.default_rng(seed)
    n    = min(n, len(keys))
    pick = sorted(rng.choice(keys, size=n, replace=False).tolist())

    fig, ax = plt.subplots(figsize=(12, 6))
    for k in pick:
        r = con.execute(
            'SELECT "Date", ret FROM algo_returns_daily '
            'WHERE run_id = ? AND key = ? ORDER BY "Date"', [run_id, k]
        ).df()
        if r.empty:
            continue
        dates  = pd.to_datetime(r["Date"])
        equity = 1.0 + np.cumsum(r["ret"].to_numpy())
        ax.plot(dates, equity, lw=1.2, label=k)

    ax.axhline(1.0, color="grey", lw=0.6, ls=":")
    ax.set_title(f"Sample equity curves (normalised, 1 + Σret) — "
                 f"{n} of {len(keys)} algo curves")
    ax.set_xlabel("date")
    ax.set_ylabel("normalised equity")
    ax.legend(fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved sample equity plot → {out_path}")
    print(f"    curves: {', '.join(pick)}")


# ── main ────────────────────────────────────────────────────────────────────

def main(data_dir: str = DEFAULT_DATA_DIR, reset: bool = False,
         plot: int = 0, plot_only: bool = False,
         plot_out: str = "plots/algo_equity_sample.png") -> str:
    if plot_only:
        con = get_connection()
        run_id = _latest_run_id(con)
        if run_id is None:
            raise RuntimeError("No algo_data_pull run found to plot. "
                               "Run ingestion first.")
        plot_sample_equity(con, run_id, plot or 6, plot_out)
        return run_id

    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files under {data_path}")

    print(f"\n── Algo data pull ──────────────────────────────────────────────")
    print(f"  data_dir : {data_path}")
    print(f"  files    : {len(files)}")

    registry = RunRegistry()
    con = get_connection()

    if reset:
        con.execute("DROP TABLE IF EXISTS algo_returns_daily")
        con.execute("DROP TABLE IF EXISTS algo_meta")
        print("  reset    : dropped algo_returns_daily / algo_meta")

    run_id = registry.start_run(
        type="algo_data_pull",
        source=str(data_path),
        n_files=len(files),
    )

    returns_parts: list[pd.DataFrame] = []
    meta_rows: list[dict] = []
    skipped: list[str] = []

    for i, f in enumerate(files, 1):
        stem = f.stem
        try:
            tf, symbol = _parse_key(stem)
            if tf not in VALID_TF:
                skipped.append(f"{stem} (tf={tf!r})")
                continue
            ret_df, meta = _daily_returns(f)
        except Exception as e:  # noqa: BLE001 — report and continue
            skipped.append(f"{stem} ({e})")
            continue

        ret_df = ret_df.copy()
        ret_df.insert(0, "key", stem)
        ret_df.insert(0, "run_id", run_id)
        returns_parts.append(ret_df)

        meta_rows.append({
            "run_id": run_id, "key": stem, "tf": tf, "symbol": symbol,
            "e0": meta["e0"], "n_days": meta["n_days"],
            "first_day": meta["first_day"], "last_day": meta["last_day"],
        })

        if i % 50 == 0:
            print(f"    parsed {i}/{len(files)}", flush=True)

    if not returns_parts:
        registry.end_run(run_id, status="failed")
        raise RuntimeError("No curves parsed — nothing written.")

    returns_all = pd.concat(returns_parts, ignore_index=True)
    returns_all = returns_all[["run_id", "Date", "key", "ret"]]
    meta_all = pd.DataFrame(meta_rows)

    # ── write panel + meta ──────────────────────────────────────────────────
    con.register("_ret_tmp", returns_all)
    con.register("_meta_tmp", meta_all)
    con.execute("""
        CREATE TABLE IF NOT EXISTS algo_returns_daily (
            run_id VARCHAR, "Date" DATE, key VARCHAR, ret DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS algo_meta (
            run_id VARCHAR, key VARCHAR, tf VARCHAR, symbol VARCHAR,
            e0 DOUBLE, n_days INTEGER, first_day DATE, last_day DATE
        )
    """)
    con.execute('INSERT INTO algo_returns_daily SELECT run_id, "Date", key, ret FROM _ret_tmp')
    con.execute("INSERT INTO algo_meta SELECT run_id, key, tf, symbol, e0, n_days, first_day, last_day FROM _meta_tmp")
    con.unregister("_ret_tmp")
    con.unregister("_meta_tmp")

    registry.log_metrics(run_id,
                         n_curves=len(meta_rows),
                         n_rows=len(returns_all),
                         n_skipped=len(skipped))
    registry.end_run(run_id)

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n  Curves written : {len(meta_rows)}")
    print(f"  Daily rows      : {len(returns_all):,}")
    print(f"  Date span       : {meta_all['first_day'].min()} → {meta_all['last_day'].max()}")
    print(f"  Timeframes      : "
          + ", ".join(f"{tf}×{(meta_all['tf'] == tf).sum()}"
                      for tf in sorted(meta_all["tf"].unique())))
    if skipped:
        print(f"  Skipped         : {len(skipped)}")
        for s in skipped[:10]:
            print(f"      - {s}")
        if len(skipped) > 10:
            print(f"      … and {len(skipped) - 10} more")
    if plot:
        print()
        plot_sample_equity(con, run_id, plot, plot_out)

    print(f"\n── Done — data_run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest raw backtest deal logs into DuckDB as a daily return panel"
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reset", action="store_true",
                        help="Drop algo_returns_daily / algo_meta before writing.")
    parser.add_argument("--plot", type=int, default=0, metavar="N",
                        help="After ingestion, overlay N sample equity curves.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip ingestion; just plot N sample curves (default 6) "
                             "from the latest algo_data_pull run.")
    parser.add_argument("--plot-out", type=str,
                        default="plots/algo_equity_sample.png")
    args = parser.parse_args()
    main(data_dir=args.data_dir, reset=args.reset, plot=args.plot,
         plot_only=args.plot_only, plot_out=args.plot_out)
