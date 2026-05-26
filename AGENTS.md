# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, GitHub Copilot, etc.) when working with code in this repository.

## Project Overview

`extracts` is a small Python helper library for loading tables, figures, and text manually extracted from published journal articles. Each dataset corresponds to a Zenodo deposit; files are downloaded and cached locally via [Pooch](https://www.fatiando.org/pooch/).

## Status

This package is being revived from a several-year-old prototype. A multi-step modernization is in progress — see [TODO.md](TODO.md) for the plan. The current source in `_fetchers.py` is the legacy implementation and will be restructured in upcoming steps (new `_processor()` subfunctions per fetcher, a `fetch_path()` utility, environment-variable-driven cache location, etc.). Treat the current code as transitional.

## Commands

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run all tests with coverage
uv run pytest tests/ --cov=extracts --cov-report=term-missing

# Run a single test file or test
uv run pytest tests/test_fetchers.py
uv run pytest tests/test_fetchers.py::test_name

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy
```

Always use `uv` when running Python scripts or installing dependencies. Never use bare `pip install` or `python`.

## Architecture

### Single source module (`src/extracts/`)

- **`_fetchers.py`** — All per-dataset `fetch_*()` functions. Each builds a `pooch.Pooch` against a Zenodo DOI, downloads the requested table by name, and returns a parsed `pandas.DataFrame`. The `DATASETS` dict maps dataset names to a `{version: doi}` map; the special `"latest"` key resolves to the latest Zenodo Concept DOI.

  Public utilities (also re-exported from `extracts.__init__`):
  - `list_available_datasets()` — sorted list of known dataset keys
  - `list_available_tables(dataset, version)` — registry filenames for a given dataset/version

  Step 2 of the modernization will: remove the static `DATASETS` registry (everything lives in Zenodo now), strip the `10.5281/zenodo.` prefix and add it programmatically, push per-table parsing into `_processor()` subfunctions, and add a `process=True` switch so callers can opt out and load raw DataFrames.

### Datasets

All remote data lives on Zenodo. The local cache defaults to `pooch.os_cache("extracts")` and (after Step 2 modernization) will be overridable via the `EXTRACTS_DATA_DIR` environment variable.

## Code Style

- Ruff: line length 100, target Python 3.13, rules `E, F, I, W, C90, UP, NPY201`
- Docstrings: NumPy convention
- Formatting: double quotes, LF line endings
