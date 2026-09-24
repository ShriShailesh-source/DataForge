# DataForge

DataForge is a deterministic Python-based data profiling and exploratory analysis engine for Phase 1 of the roadmap.

## What is implemented

- CSV and Excel ingestion
- Schema inference
- Dataset profiling
- Data-quality checks for missingness, duplicates, constant columns, and suspicious numerical values
- Conservative cleaning with transformation logging
- EDA chart generation
- Structured final report output
- FastAPI backend with a minimal browser UI

## Project structure

- backend/app/ - FastAPI app and analysis modules
- backend/tests/ - pytest coverage for Phase 1 behavior
- frontend/ - minimal browser upload interface
- datasets/ - uploaded datasets
- reports/ - generated EDA charts
- README.md - setup and run instructions

## Setup

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the backend

```bash
uvicorn backend.app.main:app --reload
```

Then open:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000 for the root status endpoint

## Browser UI

Open the frontend file directly in a browser or serve it from a simple static server:

```bash
python -m http.server 8001 --directory frontend
```

Then visit:

- http://127.0.0.1:8001/index.html

## Example Python usage

```python
from backend.app.ingestion.loader import load_dataset
from backend.app.profiling.profile import profile_dataset
from backend.app.quality.quality import run_quality_checks
from backend.app.cleaning.cleaner import clean_dataset
from backend.app.eda.eda import generate_eda
from backend.app.reports.report import generate_report

result = load_dataset("datasets/sample.csv")
df = result["dataframe"]
profile = profile_dataset(df)
quality = run_quality_checks(df)
cleaned = clean_dataset(df, remove_duplicates=True, fill_missing=True)
eda = generate_eda(df)
report = generate_report(df)
```

## Tests

Run the test suite:

```bash
pytest backend/tests/test_phase1.py -q
```

## Notes

- Raw uploads are preserved and never overwritten.
- Cleaning is conservative and all changes are recorded in a transformation log.
- This is Phase 1 only: no ML target detection, no model training, and no autonomous agent logic.
