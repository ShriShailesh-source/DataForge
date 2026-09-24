from __future__ import annotations

from typing import Any

import pandas as pd

from backend.app.cleaning.cleaner import clean_dataset
from backend.app.eda.eda import generate_eda
from backend.app.ingestion.loader import infer_schema
from backend.app.profiling.profile import profile_dataset
from backend.app.quality.quality import run_quality_checks


def generate_report(df: pd.DataFrame) -> dict[str, Any]:
    profile = profile_dataset(df)
    quality = run_quality_checks(df)
    eda = generate_eda(df)
    cleaned = clean_dataset(df, remove_duplicates=True, fill_missing=True)

    dataset_overview = {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_usage_mb": profile["dataset"]["memory_usage_mb"],
        "duplicate_rows": profile["dataset"]["duplicate_rows"],
        "missing_percentage": profile["dataset"]["missing_percentage"],
    }

    key_observations: list[str] = []
    if profile["dataset"]["duplicate_rows"]:
        key_observations.append(f"{profile['dataset']['duplicate_rows']} duplicate rows were detected.")
    if quality["missing_values"]:
        key_observations.append(f"Missing values were found in {len(quality['missing_values'])} column(s).")
    if eda["summary"]["numeric_columns"]:
        key_observations.append(f"{len(eda['summary']['numeric_columns'])} numeric columns were profiled.")
    if not key_observations:
        key_observations.append("No major data-quality issues were detected in the initial scan.")

    report = {
        "dataset_overview": dataset_overview,
        "schema": infer_schema(df),
        "data_quality_summary": quality,
        "missing_values": quality["missing_values"],
        "duplicate_analysis": quality["duplicate_rows"],
        "numerical_statistics": [col for col in profile["columns"] if col["inferred_type"] == "numeric"],
        "categorical_statistics": [col for col in profile["columns"] if col["inferred_type"] != "numeric"],
        "outlier_analysis": quality["outliers"],
        "correlation_analysis": profile["correlations"],
        "eda_visualizations": eda["charts"],
        "cleaning_transformation_log": cleaned["transformation_log"],
        "key_observations": key_observations,
    }
    return report
