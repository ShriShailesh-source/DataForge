from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def describe_numeric_series(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    stats = {
        "mean": float(numeric.mean()) if numeric.notna().any() else None,
        "median": float(numeric.median()) if numeric.notna().any() else None,
        "std": float(numeric.std()) if numeric.notna().any() else None,
        "min": float(numeric.min()) if numeric.notna().any() else None,
        "max": float(numeric.max()) if numeric.notna().any() else None,
        "q25": float(numeric.quantile(0.25)) if numeric.notna().any() else None,
        "q50": float(numeric.quantile(0.50)) if numeric.notna().any() else None,
        "q75": float(numeric.quantile(0.75)) if numeric.notna().any() else None,
        "skewness": float(numeric.skew()) if numeric.notna().any() else None,
    }
    return stats


def describe_categorical_series(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    if non_null.empty:
        return {"most_frequent": None, "frequency": 0, "cardinality": 0}
    counts = non_null.value_counts(dropna=True)
    top = counts.idxmax()
    return {
        "most_frequent": top,
        "frequency": int(counts.iloc[0]),
        "cardinality": int(non_null.nunique(dropna=True)),
    }


def _safe_corr(df: pd.DataFrame, method: str) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return numeric.corr(method=method)
    return numeric.corr(method=method)


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "dataset": {
                "rows": 0,
                "columns": 0,
                "memory_usage_mb": 0.0,
                "duplicate_rows": 0,
                "missing_percentage": 0.0,
            },
            "columns": [],
            "correlations": {"pearson": {}, "spearman": {}},
            "status": "empty",
        }

    rows = len(df)
    columns = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())
    memory_usage_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    missing_total = df.isna().sum().sum()
    missing_percentage = float((missing_total / (rows * columns)) * 100) if rows and columns else 0.0

    column_profiles = []
    for column in df.columns:
        series = df[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        null_count = int(series.isna().sum())
        null_pct = float((null_count / rows) * 100) if rows else 0.0
        unique_count = int(series.nunique(dropna=True))
        unique_pct = float((unique_count / rows) * 100) if rows else 0.0

        column_profile = {
            "column": str(column),
            "dtype": str(series.dtype),
            "inferred_type": "numeric" if is_numeric else "categorical" if series.dtype == "object" else "datetime" if pd.api.types.is_datetime64_any_dtype(series) else "boolean" if pd.api.types.is_bool_dtype(series) else "text",
            "missing_count": null_count,
            "missing_percentage": null_pct,
            "unique_count": unique_count,
            "unique_percentage": unique_pct,
        }

        if is_numeric:
            column_profile.update(describe_numeric_series(series))
        elif pd.api.types.is_datetime64_any_dtype(series):
            column_profile["min"] = pd.to_datetime(series.dropna()).min() if series.notna().any() else None
            column_profile["max"] = pd.to_datetime(series.dropna()).max() if series.notna().any() else None
        else:
            column_profile.update(describe_categorical_series(series))

        column_profiles.append(column_profile)

    numeric_df = df.select_dtypes(include=[np.number])
    corr_pearson = _safe_corr(numeric_df, method="pearson") if not numeric_df.empty else pd.DataFrame()
    corr_spearman = _safe_corr(numeric_df, method="spearman") if not numeric_df.empty else pd.DataFrame()

    return {
        "dataset": {
            "rows": rows,
            "columns": columns,
            "memory_usage_mb": memory_usage_mb,
            "duplicate_rows": duplicate_rows,
            "missing_percentage": missing_percentage,
        },
        "columns": column_profiles,
        "correlations": {
            "pearson": corr_pearson.to_dict() if not corr_pearson.empty else {},
            "spearman": corr_spearman.to_dict() if not corr_spearman.empty else {},
        },
        "status": "ok",
    }
