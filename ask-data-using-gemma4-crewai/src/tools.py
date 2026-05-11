import json
import os
import tempfile
import uuid
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from crewai.tools import tool
from tabulate import tabulate


def _detect_file_type(file_path: str) -> str:
    """
    Detect file type by examining content (not extension).
    
    Magic bytes detection:
    - Parquet: starts with 'PAR1' bytes
    - Arrow IPC: starts with 0xFFFFFFFF (4 bytes)
    - CSV: plain text not starting with JSON/Parquet/Arrow markers
    - JSON/NDJSON: starts with '{' or '['
    
    Returns: "csv", "json", "ndjson", "parquet", or "arrow"
    """
    with open(file_path, "rb") as f:
        first_bytes = f.read(100)

    # Check Parquet magic bytes (PAR1)
    if first_bytes.startswith(b"PAR1"):
        return "parquet"

    # Check Arrow IPC magic bytes (0xFFFFFFFF at start)
    if first_bytes.startswith(b"\xff\xff\xff\xff"):
        return "arrow"

    # Check JSON/NDJSON
    first_bytes_stripped = first_bytes.strip()
    if first_bytes_stripped.startswith(b"{") or first_bytes_stripped.startswith(b"["):
        return _distinguish_json_variants(file_path)

    # Default to CSV
    return "csv"


def _distinguish_json_variants(file_path: str) -> str:
    """
    Distinguish between regular JSON and NDJSON (newline-delimited JSON) format.
    
    Returns: "json" or "ndjson"
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                return "json"

            # Try to parse first line
            json.loads(first_line)

            # First line is valid JSON. Check if there's a second line.
            second_line = f.readline().strip()
            if second_line:
                # There's a second line, try to parse it
                try:
                    json.loads(second_line)
                    # Both lines are valid JSON objects → NDJSON
                    return "ndjson"
                except json.JSONDecodeError:
                    # Second line is not valid JSON → regular JSON
                    return "json"

            # Only one line → regular JSON
            return "json"

    except (json.JSONDecodeError, UnicodeDecodeError):
        # If anything fails, default to regular JSON
        # (DuckDB will handle further validation)
        return "json"


def _source_sql(file_type: str, file_path: str) -> tuple[str, list[Any]]:
    if file_type == "csv":
        return "SELECT * FROM read_csv_auto(?)", [file_path]
    if file_type == "json":
        return "SELECT * FROM read_json_auto(?)", [file_path]
    if file_type == "ndjson":
        return "SELECT * FROM read_ndjson(?)", [file_path]
    if file_type == "parquet":
        return "SELECT * FROM read_parquet(?)", [file_path]
    if file_type == "arrow":
        # Arrow is handled specially via _load_arrow_table()
        return None, [file_path]
    raise ValueError(f"Unsupported file type: {file_type}")


def _load_arrow_table(file_path: str):
    """Load an Arrow IPC file and return a PyArrow table."""
    with open(file_path, "rb") as f:
        reader = ipc.RecordBatchStreamReader(f)
        return reader.read_all()


def _format_head_preview(dataframe: pd.DataFrame) -> str:
    """Format the first 10 rows of a DataFrame as a markdown table."""
    preview = dataframe.head(10)
    if preview.empty:
        return "No rows available."
    return tabulate(preview, headers="keys", tablefmt="github", showindex=False)


def _flatten_nested_value(value: Any, parent_key: str, output: dict[str, Any]) -> None:
    """Recursively flatten nested dict/list values into dot-path keys."""
    if isinstance(value, dict):
        if not value and parent_key:
            output[parent_key] = {}
        for key, nested_value in value.items():
            child_key = f"{parent_key}.{key}" if parent_key else str(key)
            _flatten_nested_value(nested_value, child_key, output)
        return

    if isinstance(value, list):
        if not value and parent_key:
            output[parent_key] = []
        for index, nested_value in enumerate(value):
            child_key = f"{parent_key}.{index}" if parent_key else str(index)
            _flatten_nested_value(nested_value, child_key, output)
        return

    output[parent_key] = value


def _flatten_dataframe_dot_paths(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested dataframe cell values (dict/list) into dot-path columns."""
    if dataframe.empty:
        return dataframe

    records = dataframe.to_dict(orient="records")
    flattened_records: list[dict[str, Any]] = []

    for record in records:
        flattened_record: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                _flatten_nested_value(value, str(key), flattened_record)
            else:
                flattened_record[str(key)] = value
        flattened_records.append(flattened_record)

    return pd.DataFrame(flattened_records)


@tool("flatten_nested_json")
def flatten_nested_json(file_path: str) -> str:
    """Flatten deeply nested JSON/NDJSON attributes into dot-path columns and return a preview table.

    Use this tool ONLY when the user explicitly asks to flatten, expand, or see nested attributes
    as separate columns (e.g. 'flatten the JSON', 'show nested fields', 'expand attributes').
    Returns a markdown table of the first 10 rows after flattening, plus the resulting column names.
    """
    file_type = _detect_file_type(file_path)
    if file_type not in {"json", "ndjson"}:
        return f"Flattening is only applicable to JSON or NDJSON files. Detected file type: {file_type}."

    connection = duckdb.connect(database=":memory:")
    try:
        source_sql, params = _source_sql(file_type, file_path)
        connection.execute(f"CREATE TEMP TABLE data AS {source_sql}", params)
        dataframe = connection.execute("SELECT * FROM data").fetchdf()
        flattened = _flatten_dataframe_dot_paths(dataframe)
        return json.dumps(
            {
                "flattened_columns": list(flattened.columns),
                "row_count": len(flattened),
                "preview": _format_head_preview(flattened),
            },
            indent=2,
        )
    finally:
        connection.close()


@tool("inspect_data_file")
def inspect_data_file(file_path: str) -> str:
    """Detect file type (CSV, JSON, NDJSON, Parquet, Arrow) and extract schema for an uploaded data file."""
    file_type = _detect_file_type(file_path)

    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("DROP TABLE IF EXISTS data")
        
        if file_type == "arrow":
            # Handle Arrow IPC format using PyArrow
            arrow_table = _load_arrow_table(file_path)
            connection.register("data", arrow_table)
        else:
            # Handle all other formats using SQL
            source_sql, params = _source_sql(file_type, file_path)
            connection.execute(f"CREATE TEMP TABLE data AS {source_sql}", params)
        
        preview_df = connection.execute("SELECT * FROM data LIMIT 10").fetchdf()
        row_count = int(connection.execute("SELECT COUNT(*) FROM data").fetchone()[0])
        schema_rows = connection.execute("DESCRIBE data").fetchall()
        schema = [
            {
                "column": row[0],
                "type": row[1],
                "nullable": row[2],
            }
            for row in schema_rows
        ]
        return json.dumps(
            {
                "file_type": file_type,
                "row_count": row_count,
                "schema": schema,
                "head": _format_head_preview(preview_df),
            },
            indent=2,
        )
    finally:
        connection.close()


@tool("run_duckdb_sql")
def run_duckdb_sql(file_path: str, sql: str) -> str:
    """Execute DuckDB SQL against uploaded data exposed as table named data."""
    file_type = _detect_file_type(file_path)

    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("DROP TABLE IF EXISTS data")
        
        if file_type == "arrow":
            # Handle Arrow IPC format using PyArrow
            arrow_table = _load_arrow_table(file_path)
            connection.register("data", arrow_table)
        else:
            # Handle all other formats using SQL
            source_sql, params = _source_sql(file_type, file_path)
            connection.execute(f"CREATE TEMP TABLE data AS {source_sql}", params)
        
        dataframe = connection.execute(sql).fetchdf()

        if dataframe.empty:
            return "Query executed successfully. Result set is empty."

        return dataframe.to_markdown(index=False)
    finally:
        connection.close()


@tool("export_duckdb_sql_result")
def export_duckdb_sql_result(file_path: str, sql: str, output_format: str = "csv") -> str:
    """Execute DuckDB SQL against data and export the result set as a downloadable file."""
    normalized_format = output_format.lower().strip()
    if normalized_format not in {"csv", "json"}:
        raise ValueError("output_format must be either 'csv' or 'json'")

    file_type = _detect_file_type(file_path)

    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("DROP TABLE IF EXISTS data")
        
        if file_type == "arrow":
            # Handle Arrow IPC format using PyArrow
            arrow_table = _load_arrow_table(file_path)
            connection.register("data", arrow_table)
        else:
            # Handle all other formats using SQL
            source_sql, params = _source_sql(file_type, file_path)
            connection.execute(f"CREATE TEMP TABLE data AS {source_sql}", params)
        
        dataframe = connection.execute(sql).fetchdf()

        output_path = os.path.join(
            tempfile.gettempdir(),
            f"duckdb_result_{uuid.uuid4().hex}.{normalized_format}",
        )

        if normalized_format == "csv":
            dataframe.to_csv(output_path, index=False)
        else:
            dataframe.to_json(output_path, orient="records", indent=2)

        return json.dumps(
            {
                "output_format": normalized_format,
                "output_path": output_path,
                "row_count": len(dataframe),
                "columns": list(dataframe.columns),
            },
            indent=2,
        )
    finally:
        connection.close()


def _write_dataframe_format(dataframe: pd.DataFrame, output_path: str, output_format: str) -> None:
    """Write a DataFrame to the specified format."""
    if output_format == "csv":
        dataframe.to_csv(output_path, index=False)
    elif output_format == "json":
        dataframe.to_json(output_path, orient="records", indent=2)
    elif output_format == "ndjson":
        with open(output_path, "w") as f:
            for _, row in dataframe.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")
    elif output_format == "parquet":
        dataframe.to_parquet(output_path, index=False)
    elif output_format == "arrow":
        table = pa.Table.from_pandas(dataframe)
        with open(output_path, "wb") as sink:
            writer = ipc.RecordBatchStreamWriter(sink, table.schema)
            writer.write(table)
            writer.close()
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


@tool("convert_file_format")
def convert_file_format(file_path: str, output_format: str) -> str:
    """Convert a data file from one format to another (CSV, JSON, NDJSON, Parquet, Arrow)."""
    normalized_format = output_format.lower().strip()
    if normalized_format not in {"csv", "json", "ndjson", "parquet", "arrow"}:
        raise ValueError("output_format must be one of: 'csv', 'json', 'ndjson', 'parquet', 'arrow'")

    source_file_type = _detect_file_type(file_path)

    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("DROP TABLE IF EXISTS data")
        
        if source_file_type == "arrow":
            # Handle Arrow IPC format using PyArrow
            arrow_table = _load_arrow_table(file_path)
            connection.register("data", arrow_table)
        else:
            # Handle all other formats using SQL
            source_sql, params = _source_sql(source_file_type, file_path)
            connection.execute(f"CREATE TEMP TABLE data AS {source_sql}", params)
        
        dataframe = connection.execute("SELECT * FROM data").fetchdf()

        # Generate output path
        extension_map = {
            "csv": "csv",
            "json": "json",
            "ndjson": "ndjson",
            "parquet": "parquet",
            "arrow": "arrow",
        }
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"converted_{uuid.uuid4().hex}.{extension_map[normalized_format]}",
        )

        # Write the data in the requested format
        _write_dataframe_format(dataframe, output_path, normalized_format)

        return json.dumps(
            {
                "source_format": source_file_type,
                "output_format": normalized_format,
                "output_path": output_path,
                "row_count": len(dataframe),
                "columns": list(dataframe.columns),
                "message": f"Successfully converted from {source_file_type} to {normalized_format}",
            },
            indent=2,
        )
    finally:
        connection.close()
