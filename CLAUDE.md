# CLAUDE.md

## Project Overview

Dashboard Performance is a Python ETL + interactive dashboard + automated report generator for Instagram/Meta Business Suite analytics. It reads raw CSV exports, standardizes and cleans the data, computes KPIs and metrics, renders an interactive Streamlit dashboard, and exports results as CSV tables, PNG charts, and DOCX/PDF reports.

- **Language:** Python 3.13
- **Primary framework:** Streamlit 1.42.0
- **UI language:** Portuguese (PT-BR)
- **Target platform:** Windows desktop

## Architecture

The project follows a modular, layer-based architecture with clear separation of concerns:

```
Dashboard_Performance/
├── app/
│   └── dashboard.py          # Main Streamlit application (entry point)
├── src/
│   ├── common/               # Shared utilities
│   │   ├── paths.py          # Centralized path management (ProjectPaths dataclass)
│   │   └── logging_config.py # Rotating file handler + console logging
│   ├── etl/                  # Extract-Transform-Load
│   │   ├── posts_ingest.py   # CSV reading, column aliasing, type coercion
│   │   ├── export.py         # CSV export to output/tables
│   │   └── quality.py        # Data quality checks (QualityReport dataclass)
│   ├── metrics/              # KPI and metric calculations
│   │   ├── posts_metrics.py  # KPIs dataclass, top_posts()
│   │   ├── time_series.py    # Daily/weekly aggregations
│   │   ├── compare.py        # Period-over-period comparison
│   │   └── post_detail.py    # Individual post analysis
│   ├── viz/                  # Visualization generation
│   │   ├── figures.py        # Plotly chart wrappers (go.Figure)
│   │   ├── png_export.py     # Matplotlib PNG generation (9 charts, 160 DPI)
│   │   └── png_worker.py     # PNG export subprocess worker
│   └── report/               # Report generation
│       ├── report_builder.py # DOCX creation with python-docx
│       ├── pdf_export.py     # DOCX-to-PDF via docx2pdf (requires MS Word)
│       └── pdf_worker.py     # PDF conversion subprocess worker
├── input/                    # Raw CSV files (gitignored)
├── output/                   # Generated artifacts (gitignored)
│   ├── logs/                 # Rotating log files
│   ├── tables/               # Exported CSV results
│   ├── charts/               # Exported PNG visualizations
│   └── reports/              # Generated DOCX/PDF reports
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation (Portuguese)
```

**Data flow:** CSV upload -> `etl/posts_ingest.py` (parse, normalize, coerce types) -> `metrics/` (compute KPIs, time series) -> `viz/` (generate charts) -> `report/` (build DOCX/PDF) -> `app/dashboard.py` (render UI)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app/dashboard.py
```

On Windows, a PowerShell script is also available: `.\run.ps1`

## Key Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.42.0 (pinned) | Web dashboard framework |
| pandas | >=2.2.0 | DataFrame manipulation |
| numpy | >=2.0.0 | Numerical operations |
| pyarrow | >=15.0.0 | Columnar data format |
| plotly | >=5.20.0 | Interactive charts |
| kaleido | >=0.2.1 | Plotly static image export |
| matplotlib | >=3.9.0 | Static PNG chart generation |
| python-docx | >=1.1.0 | DOCX report creation |
| docx2pdf | >=0.1.8 | PDF conversion (requires MS Word) |
| chardet | >=5.2.0 | CSV encoding detection |
| python-dateutil | >=2.9.0 | Date parsing |

## Testing

There is no automated test suite. No pytest, unittest, or test files exist in the repository.

## Code Conventions

### Python Style

- All modules use `from __future__ import annotations` (PEP 563 string annotations).
- Type hints are used throughout, including function signatures and return types.
- Functions and variables use `snake_case`. Classes use `PascalCase`.
- Private/internal helper functions are prefixed with `_` (e.g., `_blank_to_na()`, `_coalesce()`).
- Every module defines `logger = logging.getLogger(__name__)` at module level.

### Data Modeling

- Immutable dataclasses with `@dataclass(frozen=True)` for structured records: `KPIs`, `PeriodKpis`, `PostSummary`, `QualityReport`, `ProjectPaths`.
- `pandas.DataFrame` is the primary data structure passed between layers.
- Column names in DataFrames use snake_case English: `post_id`, `created_at`, `caption`, `reach`, `likes`, `comments`, `shares`, `saves`, `views`, `follows`, `engagement`, `engagement_rate`.

### CSV Ingestion

- `posts_ingest.py` handles multiple encodings (utf-8, latin-1) and separators (semicolons, commas).
- A column aliasing system maps 40+ Portuguese and English column names to the canonical schema.
- Column normalization removes accents, lowercases, and converts spaces to underscores.
- Numeric parsing handles formatted values like "1.2k", "3M", "1,234".
- Derived metric: `engagement = likes + comments + shares + saves`.

### Deduplication

- Multi-level fallback: `post_id` -> `permalink` -> `created_at + caption`.
- Ties broken by ranking on reach, then engagement.

### UI Conventions

- All user-facing labels and messages are in Portuguese (PT-BR).
- Sidebar is used for file management and uploads.
- Tab-based navigation via `st.tabs` (Tab 1: single file analysis, Tab 2: multi-file comparison).
- Number formatting through `fmt_int()` and `fmt_pct()` helper functions.

### Performance

- Streamlit caching with `@st.cache_data` using file signatures (mtime) as cache keys.
- Matplotlib backend set to `"Agg"` for headless rendering.
- Rotating file handlers for logs (max 2 MB, 3 backups).

### Path Management

- All paths are centralized in `src/common/paths.py` via the `ProjectPaths` dataclass.
- Root is resolved relative to module location: `Path(__file__).resolve().parents[2]`.
- Directories are auto-created with `mkdir(parents=True, exist_ok=True)`.

### Error Handling

- Explicit `ValueError`/`RuntimeError` with descriptive messages for invalid inputs.
- CSV parsing uses fallback strategies (multiple encodings, separator detection).
- Logging uses `logger.exception()` for tracebacks, `logger.error()` for known errors, `logger.warning()` for non-critical issues.
- Type coercion uses `errors="coerce"` with `fillna` for safe defaults.

## Key Files by Importance

| File | Lines | Purpose |
|------|-------|---------|
| `app/dashboard.py` | ~858 | Main entry point, Streamlit UI with two tabs |
| `src/etl/posts_ingest.py` | ~323 | CSV ingestion, column aliasing, type coercion |
| `src/viz/png_export.py` | ~182 | Matplotlib PNG chart generation |
| `src/report/report_builder.py` | ~144 | DOCX report creation |
| `src/viz/figures.py` | ~103 | Plotly interactive chart wrappers |
| `src/etl/quality.py` | ~88 | Data quality validation |
| `src/metrics/time_series.py` | ~70 | Daily/weekly time series aggregations |
| `src/metrics/post_detail.py` | ~54 | Individual post analysis |
| `src/metrics/compare.py` | ~52 | Period comparison calculations |
| `src/metrics/posts_metrics.py` | ~51 | KPI computation and top posts |

## Guidelines for AI Assistants

- The codebase uses Portuguese for all user-facing strings, comments, and log messages. Maintain this convention when adding UI text.
- Follow the existing layer-based architecture: do not mix ETL logic into visualization modules or vice versa.
- Use frozen dataclasses for any new structured data types.
- Always include `from __future__ import annotations` at the top of new Python files.
- Add `logger = logging.getLogger(__name__)` to new modules.
- New DataFrame columns should use snake_case English names.
- Use `src/common/paths.py` for any new file path references rather than hardcoding paths.
- The project has no automated tests. When making changes, verify manually via the Streamlit dashboard.
- Streamlit is pinned at 1.42.0. Do not upgrade without explicit request.
- PDF export requires Microsoft Word installed. This feature may not work in non-Windows environments.
