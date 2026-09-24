import pandas as pd

from backend.app.cleaning.cleaner import clean_dataset
from backend.app.eda.eda import generate_eda
from backend.app.ingestion.loader import load_dataset
from backend.app.profiling.profile import profile_dataset
from backend.app.quality.quality import run_quality_checks
from backend.app.reports.report import generate_report


def test_csv_ingestion_and_schema_inference(tmp_path):
    dataset = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "score": [10.5, 11.0, 12.5],
            "category": ["a", "b", "a"],
            "signup_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    file_path = tmp_path / "sample.csv"
    dataset.to_csv(file_path, index=False)

    result = load_dataset(file_path)

    assert result["file_type"] == "csv"
    assert result["rows"] == 3
    assert result["columns"] == 4
    assert "score" in result["schema"]
    assert result["schema"]["score"]["inferred_type"] in {"numeric", "number"}


def test_missing_and_duplicate_detection():
    df = pd.DataFrame(
        {
            "a": [1, 1, None, 4],
            "b": ["x", "x", "y", "z"],
            "c": [1, 1, 2, 3],
        }
    )
    profile = profile_dataset(df)
    quality = run_quality_checks(df)

    assert profile["dataset"]["missing_percentage"] >= 0
    assert quality["missing_values"]
    assert quality["duplicate_rows"]


def test_cleaning_transforms_and_preserves_raw():
    df = pd.DataFrame(
        {
            "a": [1, 1, None, 4],
            "b": ["x", "x", "y", "z"],
            "c": ["1", "2", "3", "4"],
        }
    )

    cleaned = clean_dataset(df, remove_duplicates=True, fill_missing=True)

    assert cleaned["cleaned_df"].shape[0] <= df.shape[0]
    assert cleaned["transformation_log"]
    assert cleaned["raw_df"].equals(df)


def test_correlation_and_eda_generation():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [2, 4, 5, 4, 5, 7],
            "group": ["a", "b", "a", "b", "a", "b"],
        }
    )

    corr = profile_dataset(df)["correlations"]
    eda = generate_eda(df)

    assert "pearson" in corr
    assert "charts" in eda
    assert eda["charts"]


def test_report_generation():
    df = pd.DataFrame(
        {
            "amount": [10, 12, 15, 17, 200],
            "category": ["a", "b", "a", "c", "b"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        }
    )

    report = generate_report(df)

    assert report["dataset_overview"]
    assert "data_quality_summary" in report
    assert "key_observations" in report
