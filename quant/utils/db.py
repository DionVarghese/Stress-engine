"""
quant/utils/db.py

DuckDB-based run registry.

Design
------
Every execution (backtest, allocation, optimisation, scenario run) is
registered as a "run" with a unique ID and arbitrary labels (estimator,
allocator, universe, parameters, etc.). Results are stored as typed tables
linked to run_id.

Core tables
-----------
runs            Registry of all runs with metadata and labels
run_metrics     Flexible key-value store for scalar metrics per run

Additional result tables (e.g. weights, returns) are created dynamically
and always carry a run_id foreign key so they stay queryable by run.

Typical usage
-------------
from quant.utils.db import RunRegistry

registry = RunRegistry()

# Register a new run
run_id = registry.start_run(
    estimator="LedoitWolf",
    allocator="MinVar",
    universe="crypto_commodities",
    notes="baseline run"
)

# Log scalar metrics
registry.log_metrics(run_id, sharpe=1.42, max_drawdown=-0.12, turnover=0.08)

# Store a DataFrame result (e.g. weights time series)
registry.log_dataframe(run_id, table="weights", df=weights_df)

# Query across runs
df = registry.compare_runs(metric="sharpe")
"""

import uuid
import json
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "db" / "stress.duckdb"


# ---------------------------------------------------------------------------
# Low-level connection management
# ---------------------------------------------------------------------------

def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH), read_only=read_only)
    conn.execute("SET threads TO 4")
    return conn


@contextmanager
def managed_connection(read_only: bool = False):
    """
    Context manager — connection closed automatically on exit.

    with managed_connection() as conn:
        conn.execute("SELECT * FROM runs").df()
    """
    conn = get_connection(read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

def _bootstrap(conn: duckdb.DuckDBPyConnection) -> None:
    """Create registry tables if they don't exist."""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id      VARCHAR     PRIMARY KEY,
            created_at  TIMESTAMP   NOT NULL,
            status      VARCHAR     DEFAULT 'running',
            labels      JSON,
            notes       VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_metrics (
            run_id      VARCHAR     NOT NULL,
            metric      VARCHAR     NOT NULL,
            value       DOUBLE      NOT NULL,
            PRIMARY KEY (run_id, metric)
        )
    """)


# ---------------------------------------------------------------------------
# Run Registry
# ---------------------------------------------------------------------------

class RunRegistry:
    """
    Manages the full lifecycle of research runs.

    Parameters
    ----------
    db_path : Path, optional
        Override the default database path.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        with managed_connection() as conn:
            _bootstrap(conn)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, notes: str = "", **labels) -> str:
        """
        Register a new run and return its run_id.

        All keyword arguments become queryable labels.

        Parameters
        ----------
        notes : str
            Free-text description.
        **labels
            Arbitrary key-value metadata, e.g.:
            estimator="LedoitWolf", allocator="MinVar", universe="full"

        Returns
        -------
        str
            Unique run_id (UUID4).

        Example
        -------
        run_id = registry.start_run(
            estimator="LedoitWolf",
            allocator="MinVar",
            universe="crypto_commodities"
        )
        """
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        labels_json = json.dumps(labels)

        with managed_connection() as conn:
            conn.execute(
                "INSERT INTO runs VALUES (?, ?, 'running', ?, ?)",
                [run_id, created_at, labels_json, notes]
            )

        print(f"[run:{run_id[:8]}] started — {labels}")
        return run_id

    def end_run(self, run_id: str, status: str = "done") -> None:
        """
        Mark a run as done or failed.

        Parameters
        ----------
        status : str
            'done' or 'failed'
        """
        with managed_connection() as conn:
            conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?",
                [status, run_id]
            )
        print(f"[run:{run_id[:8]}] {status}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_metrics(self, run_id: str, **metrics) -> None:
        """
        Log scalar metrics for a run.

        Example
        -------
        registry.log_metrics(run_id, sharpe=1.42, max_drawdown=-0.12)
        """
        rows = [(run_id, k, float(v)) for k, v in metrics.items()]
        with managed_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO run_metrics VALUES (?, ?, ?)",
                rows
            )

    def log_dataframe(
        self,
        run_id: str,
        table: str,
        df: pd.DataFrame,
        if_exists: str = "append",
    ) -> None:
        """
        Persist a DataFrame result linked to a run_id.

        Automatically adds a run_id column if not present.
        Creates the table dynamically if it doesn't exist.

        Parameters
        ----------
        run_id : str
            Run identifier from start_run().
        table : str
            Target table name, e.g. 'weights', 'returns', 'positions'.
        df : pd.DataFrame
            Data to store.
        if_exists : str
            'append' (default) or 'replace'.

        Example
        -------
        registry.log_dataframe(run_id, table="weights", df=weights_df)
        """
        df = df.copy()
        if "run_id" not in df.columns:
            df.insert(0, "run_id", run_id)

        with managed_connection() as conn:
            if if_exists == "replace":
                conn.execute(f"DROP TABLE IF EXISTS {table}")

            # Create table from DataFrame schema on first use
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} AS
                SELECT * FROM df WHERE 1=0
            """)

            # Add any columns the table doesn't yet have (schema evolution)
            existing = {
                row[0] for row in conn.execute(f"DESCRIBE {table}").fetchall()
            }
            for col in df.columns:
                if col not in existing:
                    dtype = df[col].dtype
                    if pd.api.types.is_integer_dtype(dtype):
                        sql_type = "BIGINT"
                    elif pd.api.types.is_float_dtype(dtype):
                        sql_type = "DOUBLE"
                    elif pd.api.types.is_bool_dtype(dtype):
                        sql_type = "BOOLEAN"
                    else:
                        sql_type = "VARCHAR"
                    conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" {sql_type}')

            # Insert by column name so missing columns in the df get NULL
            col_list = ", ".join(f'"{c}"' for c in df.columns)
            conn.execute(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM df")

        print(f"[run:{run_id[:8]}] logged {len(df)} rows → {table}")

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_runs(self, **filters) -> pd.DataFrame:
        """
        Return all runs, optionally filtered by label values.

        Example
        -------
        registry.get_runs(allocator="MinVar")
        """
        with managed_connection(read_only=True) as conn:
            df = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC"
            ).df()

        for key, val in filters.items():
            df = df[df["labels"].apply(
                lambda x: json.loads(x).get(key) == val
            )]

        return df

    def get_metrics(self, run_id: str) -> pd.DataFrame:
        """Return all scalar metrics for a single run."""
        with managed_connection(read_only=True) as conn:
            return conn.execute(
                "SELECT metric, value FROM run_metrics WHERE run_id = ?",
                [run_id]
            ).df()

    def compare_runs(self, metric: str, **filters) -> pd.DataFrame:
        """
        Compare a specific metric across all runs, with labels expanded
        into columns for easy reading.

        Parameters
        ----------
        metric : str
            Metric name, e.g. 'sharpe', 'max_drawdown'.
        **filters
            Optional label filters, e.g. allocator="MinVar".

        Returns
        -------
        pd.DataFrame
            One row per run, sorted by metric descending.

        Example
        -------
        registry.compare_runs("sharpe", allocator="MinVar")
        """
        runs_df = self.get_runs(**filters)
        if runs_df.empty:
            return pd.DataFrame()

        with managed_connection(read_only=True) as conn:
            metrics_df = conn.execute(
                "SELECT run_id, value FROM run_metrics WHERE metric = ?",
                [metric]
            ).df().rename(columns={"value": metric})

        result = runs_df.merge(metrics_df, on="run_id", how="left")
        result["labels"] = result["labels"].apply(json.loads)
        labels_expanded = result["labels"].apply(pd.Series)
        result = pd.concat(
            [result[["run_id", "created_at", "status", "notes", metric]],
             labels_expanded],
            axis=1
        )
        return result.sort_values(metric, ascending=False)

    def get_dataframe(self, table: str, run_id: str = None) -> pd.DataFrame:
        """
        Retrieve a stored result table, optionally filtered to one run.

        Example
        -------
        weights = registry.get_dataframe("weights", run_id=run_id)
        """
        with managed_connection(read_only=True) as conn:
            if run_id:
                return conn.execute(
                    f"SELECT * FROM {table} WHERE run_id = ?", [run_id]
                ).df()
            return conn.execute(f"SELECT * FROM {table}").df()
