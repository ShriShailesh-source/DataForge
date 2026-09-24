from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as_percent(value: float, total: float) -> float:
    return float((value / total) * 100) if total else 0.0


def run_quality_checks(df: pd.DataFrame) -> dict[str, Any]:
    issues: dict[str, Any] = {
        "missing_values": [],
        "duplicate_rows": [],
        "constant_columns": [],
        "near_constant_columns": [],
        "outliers": [],
        "suspicious_values": [],
        "high_cardinality": [],
    }

    if df is None or df.empty:
        return issues

    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        if missing_count:
            issues["missing_values"].append(
                {
                    "column": str(column),
                    "severity": "medium" if _as_percent(missing_count, len(df)) > 20 else "low",
                    "affected_rows": missing_count,
                    "percentage": _as_percent(missing_count, len(df)),
                    "details": "Missing values detected in the column.",
                }
            )

        if series.nunique(dropna=True) <= 1:
            issues["constant_columns"].append(
                {
                    "column": str(column),
                    "severity": "medium",
                    "affected_rows": len(df),
                    "percentage": 100.0,
                    "details": "Column contains only one non-null value.",
                }
            )

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any() and numeric.nunique(dropna=True) > 1:
            unique_ratio = numeric.nunique(dropna=True) / numeric.notna().count()
            if unique_ratio < 0.05:
                issues["near_constant_columns"].append(
                    {
                        "column": str(column),
                        "severity": "low",
                        "affected_rows": int(numeric.notna().sum()),
                        "percentage": _as_percent(numeric.notna().sum(), len(df)),
                        "details": "Column has very low variability relative to its size.",
                    }
                )

        if numeric.notna().any():
            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1
            low_bound = q1 - 1.5 * iqr
            high_bound = q3 + 1.5 * iqr
            outliers_mask = (numeric < low_bound) | (numeric > high_bound)
            if outliers_mask.any():
                outlier_count = int(outliers_mask.sum())
                issues["outliers"].append(
                    {
                        "column": str(column),
                        "severity": "medium" if _as_percent(outlier_count, len(df)) > 5 else "low",
                        "affected_rows": outlier_count,
                        "percentage": _as_percent(outlier_count, len(df)),
                        "details": "Values fall outside the interquartile range and may be outliers.",
                    }
                )

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        issues["duplicate_rows"].append(
            {
                "column": "all_columns",
                "severity": "medium",
                "affected_rows": duplicate_rows,
                "percentage": _as_percent(duplicate_rows, len(df)),
                "details": "Exact duplicate rows were found.",
            }
        )

    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            mean = numeric.mean()
            std = numeric.std(ddof=1)
            if pd.notna(std) and std > 0:
                z_scores = np.abs((numeric - mean) / std)
                suspicious = z_scores > 4
                if suspicious.any():
                    issues["suspicious_values"].append(
                        {
                            "column": str(column),
                            "severity": "medium",
                            "affected_rows": int(suspicious.sum()),
                            "percentage": _as_percent(int(suspicious.sum()), len(df)),
                            "details": "Values are more than four standard deviations from the mean.",
                        }
                    )
        elif series.dtype == "object":
            value_counts = series.value_counts(dropna=True)
            if len(value_counts) > 0:
                max_share = value_counts.iloc[0] / value_counts.sum()
                if max_share > 0.9 and len(value_counts) > 1:
                    issues["high_cardinality"].append(
                        {
                            "column": str(column),
                            "severity": "low",
                            "affected_rows": int(value_counts.iloc[0]),
                            "percentage": _as_percent(int(value_counts.iloc[0]), len(df)),
                            "details": "The field is highly skewed toward a dominant category.",
                        }
                    )

    return issues
