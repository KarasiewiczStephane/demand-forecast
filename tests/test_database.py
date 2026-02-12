"""Tests for the DuckDB database manager."""

import pandas as pd
import pytest

from src.data.database import DuckDBManager


class TestDuckDBManager:
    """Tests for DuckDBManager class."""

    def test_connection(self, tmp_duckdb_path: str) -> None:
        """Manager should establish a valid connection."""
        db = DuckDBManager(tmp_duckdb_path)
        assert db.conn is not None
        db.close()

    def test_schemas_created(self, tmp_duckdb_path: str) -> None:
        """All three schemas should be created on initialization."""
        db = DuckDBManager(tmp_duckdb_path)
        schemas = db.query("SELECT schema_name FROM information_schema.schemata")
        schema_names = schemas["schema_name"].tolist()
        assert "raw" in schema_names
        assert "cleaned" in schema_names
        assert "features" in schema_names
        db.close()

    def test_store_and_query_dataframe(self, tmp_duckdb_path: str) -> None:
        """Storing a DataFrame and querying it should return the same data."""
        db = DuckDBManager(tmp_duckdb_path)
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        db.store_dataframe(df, "test_table", schema="raw")

        result = db.query("SELECT * FROM raw.test_table ORDER BY a")
        assert len(result) == 3
        assert list(result["a"]) == [1, 2, 3]
        db.close()

    def test_table_exists(self, tmp_duckdb_path: str) -> None:
        """table_exists should return True for existing tables."""
        db = DuckDBManager(tmp_duckdb_path)
        df = pd.DataFrame({"x": [1]})
        db.store_dataframe(df, "exists_test")

        assert db.table_exists("exists_test", schema="raw") is True
        assert db.table_exists("no_such_table", schema="raw") is False
        db.close()

    def test_store_to_different_schemas(self, tmp_duckdb_path: str) -> None:
        """DataFrames should be storable in all schemas."""
        db = DuckDBManager(tmp_duckdb_path)
        df = pd.DataFrame({"val": [10, 20]})

        for schema in ("raw", "cleaned", "features"):
            db.store_dataframe(df, "multi_schema", schema=schema)
            assert db.table_exists("multi_schema", schema=schema)

        db.close()

    def test_replace_existing_table(self, tmp_duckdb_path: str) -> None:
        """Storing to the same table should replace its contents."""
        db = DuckDBManager(tmp_duckdb_path)
        df1 = pd.DataFrame({"val": [1, 2]})
        df2 = pd.DataFrame({"val": [10, 20, 30]})

        db.store_dataframe(df1, "replace_test")
        db.store_dataframe(df2, "replace_test")

        result = db.query("SELECT COUNT(*) AS cnt FROM raw.replace_test")
        assert result["cnt"].iloc[0] == 3
        db.close()

    def test_parent_directory_created(self, tmp_path: object) -> None:
        """Manager should create parent directories for db_path."""
        from pathlib import Path

        deep_path = str(Path(str(tmp_path)) / "nested" / "dir" / "test.duckdb")
        db = DuckDBManager(deep_path)
        assert Path(deep_path).parent.exists()
        db.close()

    def test_close(self, tmp_duckdb_path: str) -> None:
        """Closing the manager should invalidate the connection."""
        db = DuckDBManager(tmp_duckdb_path)
        db.close()
        with pytest.raises(Exception):
            db.query("SELECT 1")
