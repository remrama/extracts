# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, GitHub Copilot, etc.) when working with code in this repository.

## Project Overview

`extracts` is a small Python helper library for loading tables, figures, and text manually extracted from published journal articles. Each dataset corresponds to a Zenodo deposit; files are downloaded and cached locally via [Pooch](https://www.fatiando.org/pooch/).

## Status

This package is being revived from a several-year-old prototype. A multi-step modernization is in progress — see [TODO.md](TODO.md) for the plan. Steps 1 (infrastructure), 2 (code), 3 (tests), and 4 (docs) are complete; the `liwca.datasets.tables` port and final cleanup remain.

## Commands

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run local tests (network and network_full are skipped by default)
uv run pytest

# Run a single test file or test
uv run pytest tests/test_processor.py
uv run pytest tests/test_processor.py::test_first_call_parses_and_writes_parquet

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy
```

### Test markers

- `uv run pytest` — local tests only (default; `network` and `network_full` skipped).
- `uv run pytest -m network` — remote sample (~15 tests, one canonical table per dataset plus targeted regression tests). CI runs this on push to main.
- `uv run pytest -m network_full` — exhaustive walk over every registered Zenodo table; slow, opt-in only.

### Docs

Docs are Sphinx + `myst-nb`; dataset pages execute code cells at build time so previews reflect the live Zenodo data.

- `uv run sphinx-build -W -b html docs/ docs/_build/html` — full build (warnings are errors).
- Cold builds hit Zenodo for every preview; subsequent builds reuse the Pooch cache plus `docs/.jupyter_cache/` for near-instant rebuilds.

Always use `uv` when running Python scripts or installing dependencies. Never use bare `pip install` or `python`.

## Architecture

### Single source module (`src/extracts/_fetchers.py`)

The whole public API lives in one module:

- **Per-dataset fetchers** (`fetch_barrett2020`, `fetch_cariola2010`, etc.) — each builds a `pooch.Pooch` against a Zenodo DOI via the private `_create_pup(dataset, version)` helper, then routes through a local `_processor(source_path) -> pd.DataFrame` closure that defines that dataset's per-table parse logic (`index_col`, MultiIndex `header`, `skiprows`, etc.). Every fetcher takes `process: bool = True`:
  - `process=True` (default): runs `_processor`, caches the result as parquet next to the raw download via the `CacheParquet` Pooch processor, and returns `pd.read_parquet(cache_path)`. Parquet round-trips MultiIndex columns and dtypes losslessly.
  - `process=False`: skips the processor and returns `pd.read_table(raw_path, **kwargs)` so callers can supply their own read kwargs.
- **`fetch_path(dataset, table, version=None)`** — returns the local `Path` of the downloaded raw file (filename takes `.tsv` if no extension is given).
- **`fetch_text(dataset, ..., process=True)`** — returns the text contents of `text.json`, or its local path when `process=False`.
- **`fetch_reference(dataset, ..., process=True)`** — returns a parsed BibTeX dict (`type`, `key`, `fields`), or the local path when `process=False`.
- **`list_available_datasets()`** / **`list_available_tables(dataset, version=None)`** — discovery utilities.
- **`get_location()`** / **`set_location(path)`** — read/override the cache root. `get_location` resolves `$EXTRACTS_DATA_DIR` if set, else falls back to `pooch.os_cache("extracts")`. `set_location` writes the env var for the current process.

The `DATASETS` dict stores just the numeric Zenodo DOI suffix per version; `_create_pup` prepends the shared `DOI_PREFIX = "10.5281/zenodo."` programmatically.

### Datasets

All remote data lives on Zenodo. Each dataset gets its own subdirectory under `get_location()` (e.g. `~/.cache/extracts/barrett2020/`).

## Code Style

- Ruff: line length 100, target Python 3.13, rules `E, F, I, W, C90, UP, NPY201`
- Docstrings: NumPy convention
- Formatting: double quotes, LF line endings
