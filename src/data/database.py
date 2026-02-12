"""DuckDB database management for the demand forecasting system.

Provides a manager class that handles connection lifecycle,
schema initialization, and DataFrame storage/retrieval operations
across raw, cleaned, and features data layers.
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBManager:
    """Manages DuckDB connections and schema operations.

    Creates and maintains a DuckDB database with three schemas:
    ``raw``, ``cleaned``, and ``features`` for data layer separation.

    Attributes:
        db_path: Filesystem path to the DuckDB database file.
        conn: Active DuckDB connection instance.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize database connection and schemas.

        Args:
            db_path: Path where the DuckDB file will be created or opened.
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._init_schemas()
        logger.info("DuckDB connection established at %s", db_path)

    def _init_schemas(self) -> None:
        """Create data layer schemas if they do not exist."""
        for schema in ("raw", "cleaned", "features"):
            self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    def store_dataframe(
        self, df: pd.DataFrame, table_name: str, schema: str = "raw"
    ) -> None:
        """Store a pandas DataFrame as a DuckDB table.

        Args:
            df: DataFrame to store.
            table_name: Name for the target table.
            schema: Target schema (raw, cleaned, or features).
        """
        self.conn.execute(
            f"CREATE OR REPLACE TABLE {schema}.{table_name} AS SELECT * FROM df"
        )
        logger.info("Stored %d rows in %s.%s", len(df), schema, table_name)

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            sql: SQL query string to execute.

        Returns:
            Query results as a pandas DataFrame.
        """
        return self.conn.execute(sql).fetchdf()

    def table_exists(self, table_name: str, schema: str = "raw") -> bool:
        """Check whether a table exists in the given schema.

        Args:
            table_name: Name of the table to check.
            schema: Schema to search in.

        Returns:
            True if the table exists, False otherwise.
        """
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = ? AND table_name = ?",
            [schema, table_name],
        ).fetchone()
        return result[0] > 0 if result else False

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        logger.info("DuckDB connection closed")
