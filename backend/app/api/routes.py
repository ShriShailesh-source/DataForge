from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.cleaning.cleaner import clean_dataset
from backend.app.eda.eda import generate_eda
from backend.app.ingestion.loader import load_dataset
from backend.app.profiling.profile import profile_dataset
from backend.app.quality.quality import run_quality_checks
from backend.app.reports.report import generate_report

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_extension = Path(file.filename).suffix.lower()
    allowed = {".csv", ".xlsx", ".xls"}
    if file_extension not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

    storage_dir = Path("datasets")
    storage_dir.mkdir(exist_ok=True)
    upload_path = storage_dir / file.filename
    content = await file.read()
    upload_path.write_bytes(content)

    try:
        loaded = load_dataset(upload_path)
    except Exception as exc:  # pragma: no cover - API validation path
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "file_name": loaded["file_name"],
        "file_type": loaded["file_type"],
        "rows": loaded["rows"],
        "columns": loaded["columns"],
        "schema": loaded["schema"],
        "preview": loaded["preview"],
        "status": loaded["status"],
    }


@router.post("/analyze")
async def analyze_dataset(file_name: str) -> dict[str, Any]:
    dataset_path = Path("datasets") / file_name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    result = load_dataset(dataset_path)
    df = result["dataframe"]
    profile = profile_dataset(df)
    quality = run_quality_checks(df)
    eda = generate_eda(df)
    report = generate_report(df)
    cleaned = clean_dataset(df, remove_duplicates=True, fill_missing=True)

    return {
        "dataset": result,
        "profiling": profile,
        "quality": quality,
        "eda": eda,
        "report": report,
        "cleaned_dataset": {
            "rows": len(cleaned["cleaned_df"]),
            "columns": len(cleaned["cleaned_df"].columns),
            "transformation_log": cleaned["transformation_log"],
        },
    }


@router.get("/report")
async def get_report(file_name: str) -> dict[str, Any]:
    dataset_path = Path("datasets") / file_name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    result = load_dataset(dataset_path)
    return generate_report(result["dataframe"])
