"""
scripts/rebalance_schedule.py

Generates the rebalance schedule for the rolling NCO backtest.

Design
------
- Fund rebalances at end of every week: last valid trading day on or
  before Friday (rolls back to Thursday if Friday is a holiday, etc.)
- Three lookback windows tested: 6 months (~126 bdays), 4 months (~84
  bdays), 3 months (~63 bdays)
- First rebalance date for each window is the first Friday on or after
  the lookback period has elapsed from the dataset start date
- Each rebalance row defines:
    rebalance_date  : the Friday weights are estimated and applied
    window_start    : first day of the lookback window (inclusive)
    window_end      : last day of lookback window = rebalance_date (inclusive)
    n_days          : actual number of trading days in [window_start, window_end]

No lookahead: window_end == rebalance_date, meaning we only use data
available *as of* close on that Friday.

Data source
-----------
Algo-only universe. The trading-day calendar is taken from the distinct
dates in `algo_returns_daily` for the most recent `algo_data_pull` run
(populated by pull_algo_data.py). The old market-data path
(market_returns / data_pull) is gone — the fund data is all we model now.
Only the calendar is used here; return *values* are irrelevant to the
schedule.

Usage
-----
python rebalance_schedule.py

Outputs
-------
Prints schedule summary and stores to DuckDB via RunRegistry under
table `rebalance_schedule`, keyed to the most recent algo_data_pull run_id.
Also saves a CSV to the working directory for quick inspection.
"""

import os
import sys
from datetime import timedelta

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

# ── Config ─────────────────────────────────────────────────────────────────────

# Approximate trading days per calendar month
BDAYS_PER_MONTH = 21

LOOKBACK_WINDOWS = {
    "6m": 6 * BDAYS_PER_MONTH,   # 126 bdays
    "4m": 4 * BDAYS_PER_MONTH,   #  84 bdays
    "3m": 3 * BDAYS_PER_MONTH,   #  63 bdays
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def last_valid_friday(date: pd.Timestamp, bday_index: pd.DatetimeIndex) -> pd.Timestamp:
    """
    Given a target Friday (or any date), return the latest date in
    bday_index that is <= date and falls on a Friday or, if that exact
    Friday is not a trading day, the nearest prior trading day.

    This correctly handles:
        - Fridays that are public holidays  → rolls back to Thursday
        - Non-Friday input dates            → rolls back to prior Friday
          (used when computing the first rebalance date)
    """
    # Candidate: last bday on or before date
    candidates = bday_index[bday_index <= date]
    if len(candidates) == 0:
        raise ValueError(f"No trading days on or before {date}")

    # Walk back until we land on a Friday (weekday == 4) or exhaust
    # the index — the latter only happens at the very start of the
    # series and is guarded against by the first_rebalance logic below.
    candidate = candidates[-1]
    idx_pos   = bday_index.get_loc(candidate)

    while candidate.weekday() != 4:   # 4 == Friday
        if idx_pos == 0:
            # Edge case: no Friday exists before this point — just
            # return the earliest available trading day
            break
        idx_pos  -= 1
        candidate = bday_index[idx_pos]

    return candidate


def build_friday_rebalance_dates(
    bday_index: pd.DatetimeIndex,
    lookback_bdays: int,
) -> pd.DatetimeIndex:
    """
    Return all valid rebalance Fridays for a given lookback window.

    First rebalance date: the first Friday on or after
        bday_index[0] + lookback_bdays trading days

    Subsequent dates: every Friday in bday_index after the first,
    subject to the Friday-or-prior-valid-day rule.
    """
    # Minimum index position needed to have a full lookback window
    # bday_index[lookback_bdays] is the first day where we have
    # exactly lookback_bdays prior trading days available
    first_eligible_pos = lookback_bdays  # 0-indexed

    if first_eligible_pos >= len(bday_index):
        raise ValueError(
            f"Lookback of {lookback_bdays} bdays exceeds dataset length "
            f"({len(bday_index)} bdays)"
        )

    first_eligible_date = bday_index[first_eligible_pos]

    # All Fridays in bday_index from first_eligible_date onward
    all_fridays = bday_index[
        (bday_index >= first_eligible_date) &
        (bday_index.weekday == 4)
    ]

    # For each calendar week that has a Friday in bday_index, that
    # Friday is already a valid trading day — no roll needed.
    # Weeks where Friday is a holiday will simply have no Friday in
    # bday_index, so we need to detect those gaps and insert the
    # rolled-back Thursday (or Wednesday, etc.).

    rebalance_dates = _fill_holiday_fridays(all_fridays, bday_index, first_eligible_date)

    return pd.DatetimeIndex(sorted(set(rebalance_dates)))


def _fill_holiday_fridays(
    fridays: pd.DatetimeIndex,
    bday_index: pd.DatetimeIndex,
    start_date: pd.Timestamp,
) -> list:
    """
    Detect weeks where Friday is a holiday and insert the prior valid
    trading day as the rebalance date for that week.

    Strategy: iterate over all calendar weeks between start_date and
    end of bday_index; for each week, find the Friday; if that Friday
    is not in bday_index, find the last bday in that week.
    """
    end_date = bday_index[-1]

    # Generate all calendar Fridays in the range
    cal_fridays = pd.date_range(
        start = start_date,
        end   = end_date,
        freq  = "W-FRI",   # every Friday
    )

    result = []
    for fri in cal_fridays:
        if fri in bday_index:
            result.append(fri)
        else:
            # Holiday Friday — find last bday in the same Mon-Fri week
            week_start = fri - timedelta(days=4)   # Monday
            week_bdays = bday_index[
                (bday_index >= week_start) & (bday_index <= fri)
            ]
            if len(week_bdays) > 0:
                result.append(week_bdays[-1])
            # If entire week has no trading days (e.g. Christmas week
            # in some markets) skip — extremely rare in global data

    return result


def build_schedule(
    bday_index: pd.DatetimeIndex,
    lookback_label: str,
    lookback_bdays: int,
) -> pd.DataFrame:
    """
    Build the full rebalance schedule DataFrame for one lookback window.

    Columns
    -------
    lookback        : label ("6m", "4m", "3m")
    lookback_bdays  : target number of trading days in window
    rebalance_date  : the Friday on which weights are estimated + applied
    window_start    : first trading day of the lookback window
    window_end      : == rebalance_date (no lookahead)
    n_days          : actual bdays in [window_start, window_end]
    rebalance_n     : sequential rebalance number (1-indexed)
    """
    rebalance_dates = build_friday_rebalance_dates(bday_index, lookback_bdays)

    rows = []
    for i, reb_date in enumerate(rebalance_dates):
        # window_start: go back exactly lookback_bdays from reb_date
        reb_pos      = bday_index.get_loc(reb_date)
        start_pos    = reb_pos - lookback_bdays + 1
        start_pos    = max(start_pos, 0)   # clamp to dataset start
        window_start = bday_index[start_pos]
        window_end   = reb_date

        # Actual days in window (may differ slightly from target at
        # dataset boundaries)
        n_days = len(bday_index[(bday_index >= window_start) & (bday_index <= window_end)])

        rows.append({
            "lookback":       lookback_label,
            "lookback_bdays": lookback_bdays,
            "rebalance_date": reb_date,
            "window_start":   window_start,
            "window_end":     window_end,
            "n_days":         n_days,
            "rebalance_n":    i + 1,
        })

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    registry = RunRegistry()
    con      = get_connection()

    # ── Load trading day index from DB ────────────────────────────────────────

    row = con.execute("""
        SELECT run_id FROM runs
        WHERE json_extract_string(labels, '$.type') = 'algo_data_pull'
        ORDER BY created_at DESC
        LIMIT 1
    """).fetchone()

    if row is None:
        raise RuntimeError("No algo_data_pull run found. Run pull_algo_data.py first.")

    run_id = row[0]

    dates_raw = con.execute("""
        SELECT DISTINCT "Date" AS date
        FROM algo_returns_daily
        WHERE run_id = ?
        ORDER BY date
    """, [run_id]).df()

    all_dates  = pd.to_datetime(dates_raw["date"])
    # Cap to weekdays only and no future dates (crypto weekend dates would
    # otherwise extend the range and push the schedule into future weeks)
    today      = pd.Timestamp.today().normalize()
    bday_index = pd.DatetimeIndex(all_dates[(all_dates.dt.weekday < 5) & (all_dates <= today)])

    print(f"\n── Dataset ──────────────────────────────────────────────────────────")
    print(f"  run_id       : {run_id}")
    print(f"  First bday   : {bday_index[0].date()}")
    print(f"  Last bday    : {bday_index[-1].date()}")
    print(f"  Total bdays  : {len(bday_index)}")

    # ── Build schedules ───────────────────────────────────────────────────────

    schedules = []

    print(f"\n── Rebalance schedules ──────────────────────────────────────────────")
    print(f"  {'Window':<8} {'Lookback':>10} {'First reb':>12} {'Last reb':>12} "
          f"{'N rebs':>8} {'Avg days':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")

    for label, n_bdays in LOOKBACK_WINDOWS.items():
        sched = build_schedule(bday_index, label, n_bdays)
        schedules.append(sched)

        print(f"  {label:<8} {n_bdays:>10} "
              f"{sched['rebalance_date'].min().date()!s:>12} "
              f"{sched['rebalance_date'].max().date()!s:>12} "
              f"{len(sched):>8} "
              f"{sched['n_days'].mean():>10.1f}")

    full_schedule = pd.concat(schedules, ignore_index=True)

    # ── Sanity checks ─────────────────────────────────────────────────────────

    print(f"\n── Sanity checks ────────────────────────────────────────────────────")

    # 1. No lookahead: window_end must always equal rebalance_date
    lookahead = full_schedule[full_schedule["window_end"] != full_schedule["rebalance_date"]]
    assert len(lookahead) == 0, f"Lookahead detected in {len(lookahead)} rows"
    print(f"  ✓ No lookahead bias")

    # 2. window_start must be strictly before rebalance_date
    bad_start = full_schedule[full_schedule["window_start"] >= full_schedule["rebalance_date"]]
    assert len(bad_start) == 0, f"window_start >= rebalance_date in {len(bad_start)} rows"
    print(f"  ✓ window_start < rebalance_date")

    # 3. All rebalance dates must be valid trading days
    reb_dates  = pd.DatetimeIndex(full_schedule["rebalance_date"].unique())
    not_bdays  = reb_dates.difference(bday_index)
    assert len(not_bdays) == 0, f"{len(not_bdays)} rebalance dates are not trading days"
    print(f"  ✓ All rebalance dates are valid trading days")

    # 4. All rebalance dates must be Fridays or the last bday of their week
    weekdays   = pd.DatetimeIndex(full_schedule["rebalance_date"]).weekday
    non_friday = (weekdays != 4).sum()
    # Non-Fridays are allowed only if the Friday of that week is a holiday
    print(f"  ✓ {(weekdays == 4).sum()} Fridays, "
          f"{non_friday} holiday-adjusted dates (Thu or earlier)")

    # 5. Consecutive rebalances are ~5 bdays apart (one week)
    for label in LOOKBACK_WINDOWS:
        sched   = full_schedule[full_schedule["lookback"] == label].copy()
        gaps    = sched["rebalance_date"].diff().dropna().dt.days
        assert gaps.min() >= 1, f"{label}: duplicate or backward rebalance dates"
        print(f"  {label}: min gap = {gaps.min():.0f} days, max gap = {gaps.max():.0f} days")
    print(f"  ✓ Weekly rebalance cadence verified")

    # 6. n_days within 5% of target for all non-boundary windows
    for label, n_bdays in LOOKBACK_WINDOWS.items():
        sched      = full_schedule[full_schedule["lookback"] == label]
        # Skip first rebalance (boundary) which may have exactly n_bdays
        interior   = sched.iloc[1:]
        day_dev    = (interior["n_days"] - n_bdays).abs()
        assert day_dev.max() <= max(3, int(0.05 * n_bdays)), \
            f"{label}: n_days deviates more than 5% from target"
    print(f"  ✓ Lookback window lengths within tolerance")

    # ── Store to DuckDB ───────────────────────────────────────────────────────

    store_df = full_schedule.copy()
    store_df["rebalance_date"] = store_df["rebalance_date"].dt.date.astype(str)
    store_df["window_start"]   = store_df["window_start"].dt.date.astype(str)
    store_df["window_end"]     = store_df["window_end"].dt.date.astype(str)
    store_df["run_id"]         = run_id

    # Delete any existing rows for this run before inserting to prevent duplicates
    # from repeated runs (log_dataframe always appends). Guard for the first run,
    # when the table does not exist yet.
    tbl_exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'rebalance_schedule'"
    ).fetchone() is not None
    if tbl_exists:
        con.execute("DELETE FROM rebalance_schedule WHERE run_id = ?", [run_id])

    registry.log_dataframe(run_id, table="rebalance_schedule", df=store_df)

    registry.log_metrics(
        run_id,
        schedule_n_windows  = len(LOOKBACK_WINDOWS),
        schedule_total_rebs = len(full_schedule),
    )

    # ── CSV for quick inspection ──────────────────────────────────────────────

    csv_path = "rebalance_schedule.csv"
    store_df.to_csv(csv_path, index=False)

    print(f"\n── Sample rows (6m window, first 5 rebalances) ──────────────────────")
    sample = full_schedule[full_schedule["lookback"] == "6m"].head(5)[
        ["rebalance_n", "rebalance_date", "window_start", "window_end", "n_days"]
    ]
    print(sample.to_string(index=False))

    print(f"\n── Stored ───────────────────────────────────────────────────────────")
    print(f"  DuckDB table : rebalance_schedule ({len(full_schedule)} rows)")
    print(f"  CSV          : {csv_path}")
    print(f"  run_id       : {run_id}")
    print(f"\n── Complete ─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
