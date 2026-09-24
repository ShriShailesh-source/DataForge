from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv": "csv", ".xlsx": "excel", ".xls": "excel"}


def _detect_file_type(file_path: str | Path) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    return SUPPORTED_EXTENSIONS[suffix]


def _read_csv(path: str | Path) -> pd.DataFrame:
    candidates = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin-1"},
        {"encoding": "cp1252"},
    ]
    last_error = None
    for options in candidates:
        try:
            return pd.read_csv(path, **options)
        except Exception as exc:  # pragma: no cover - fallback handling
            last_error = exc
    raise ValueError(f"Unable to read CSV file: {last_error}")


def _read_excel(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception as exc:  # pragma: no cover - fallback handling
        raise ValueError(f"Unable to read Excel file: {exc}") from exc


def infer_schema(df: pd.DataFrame) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            inferred_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            inferred_type = "datetime"
        elif pd.api.types.is_bool_dtype(series):
            inferred_type = "boolean"
        elif pd.api.types.is_categorical_dtype(series) or series.dtype == "object":
            inferred_type = "categorical"
        else:
            inferred_type = "text"

        schema[str(column)] = {
            "column": str(column),
            "inferred_type": inferred_type,
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }
    return schema


def load_dataset(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    file_type = _detect_file_type(path)
    if file_type == "csv":
        df = _read_csv(path)
    else:
        df = _read_excel(path)

    if df.empty:
        return {
            "file_name": path.name,
            "file_type": file_type,
            "rows": 0,
            "columns": 0,
            "schema": {},
            "preview": [],
            "status": "empty_dataset",
        }

    if not isinstance(df.columns, pd.Index):
        raise ValueError("Invalid dataset columns.")

    cleaned_columns = []
    for column in df.columns:
        cleaned_columns.append(str(column))
    df = df.rename(columns={old: new for old, new in zip(df.columns, cleaned_columns)})

    return {
        "file_name": path.name,
        "file_type": file_type,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "schema": infer_schema(df),
        "preview": df.head(10).to_dict(orient="records"),
        "status": "loaded",
        "dataframe": df,
    }


def load_dataset_json(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        json.loads(json.dumps(payload))
    except ValueError as exc:  # pragma: no cover - validation path
        raise ValueError(f"Invalid JSON payload: {exc}") from exc
    df = pd.DataFrame(payload.get("data", []))
    return {
        "file_name": payload.get("file_name", "dataset.json"),
        "file_type": "json",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "schema": infer_schema(df),
        "preview": df.head(10).to_dict(orient="records"),
        "status": "loaded",
        "dataframe": df,
    }
