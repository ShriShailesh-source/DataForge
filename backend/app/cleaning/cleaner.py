from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd


def _coerce_datetime(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    converted = pd.to_datetime(series, errors="coerce")
    return converted


def _normalize_categorical(series: pd.Series) -> pd.Series:
    if series.dtype != "object":
        return series
    normalized = series.copy()
    normalized = normalized.str.strip()
    normalized = normalized.str.replace(r"\s+", " ", regex=True)
    return normalized


def clean_dataset(df: pd.DataFrame, *, remove_duplicates: bool = True, fill_missing: bool = True, fill_value: Any = None) -> dict[str, Any]:
    raw_df = df.copy(deep=True)
    working_df = df.copy(deep=True)
    transformation_log: list[str] = []

    if remove_duplicates:
        before_rows = len(working_df)
        working_df = working_df.drop_duplicates().copy()
        removed = before_rows - len(working_df)
        if removed > 0:
            transformation_log.append(f"Removed {removed} exact duplicate rows")

    if fill_missing:
        for column in working_df.columns:
            series = working_df[column]
            if pd.api.types.is_numeric_dtype(series):
                fill = series.median() if fill_value is None else fill_value
                if pd.notna(fill):
                    missing_count = int(series.isna().sum())
                    if missing_count > 0:
                        working_df[column] = series.fillna(fill)
                        transformation_log.append(f"Filled {missing_count} missing values in '{column}' with median {fill}")
            elif pd.api.types.is_datetime64_any_dtype(series):
                if series.isna().any():
                    working_df[column] = series.fillna(pd.Timestamp.now())
                    transformation_log.append(f"Filled missing datetime values in '{column}' with current timestamp")
            else:
                if series.isna().any():
                    fill = "unknown" if fill_value is None else fill_value
                    working_df[column] = series.fillna(fill)
                    transformation_log.append(f"Filled {int(series.isna().sum())} missing values in '{column}' with '{fill}'")

    for column in list(working_df.columns):
        series = working_df[column]
        if pd.api.types.is_object_dtype(series):
            stripped = _normalize_categorical(series)
            if not stripped.equals(series):
                working_df[column] = stripped
                transformation_log.append(f"Normalized text values in '{column}' by trimming whitespace")

    for column in list(working_df.columns):
        series = working_df[column]
        if series.dtype == "object":
            try:
                parsed = _coerce_datetime(series)
                if parsed.notna().sum() > 0 and parsed.notna().sum() >= max(1, len(series) * 0.8):
                    working_df[column] = parsed
                    transformation_log.append(f"Parsed datetime-like values in '{column}'")
            except Exception:
                pass

    return {
        "raw_df": raw_df,
        "cleaned_df": working_df,
        "transformation_log": transformation_log,
        "before_rows": len(raw_df),
        "after_rows": len(working_df),
    }
