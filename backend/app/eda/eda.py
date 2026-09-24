from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _generate_numeric_charts(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    for column in columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            series.hist(ax=axes[0], bins=20)
            axes[0].set_title(f"Histogram: {column}")
            series.plot.box(ax=axes[1])
            axes[1].set_title(f"Boxplot: {column}")
            plt.tight_layout()
            path = f"reports/{column}_dist.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            charts.append({"type": "numeric_distribution", "column": column, "path": path})
    return charts


def _generate_categorical_charts(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    for column in columns:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            value_counts = series.dropna().value_counts().head(10)
            if value_counts.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            value_counts.plot(kind="bar", ax=ax)
            ax.set_title(f"Top categories: {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("count")
            plt.tight_layout()
            path = f"reports/{column}_category.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            charts.append({"type": "categorical_distribution", "column": column, "path": path})
    return charts


def _generate_datetime_charts(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    for column in columns:
        series = pd.to_datetime(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        counts = series.dt.to_period("M").value_counts().sort_index()
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Temporal distribution: {column}")
        plt.tight_layout()
        path = f"reports/{column}_time.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        charts.append({"type": "datetime_distribution", "column": column, "path": path})
    return charts


def generate_eda(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {"charts": [], "summary": {"numeric_columns": [], "categorical_columns": [], "datetime_columns": []}}

    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col])]
    datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

    charts = []
    charts.extend(_generate_numeric_charts(df, numeric_columns[:5]))
    charts.extend(_generate_categorical_charts(df, categorical_columns[:3]))
    charts.extend(_generate_datetime_charts(df, datetime_columns[:2]))

    summary = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
    }
    return {"charts": charts, "summary": summary}
