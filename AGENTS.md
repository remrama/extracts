# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, GitHub Copilot, etc.) when working with code in this repository.

## Project Overview

`extracts` is a small Python helper library for loading tables, figures, and text manually extracted from published journal articles. Each dataset corresponds to a Zenodo deposit; files are downloaded and cached locally via [Pooch](https://www.fatiando.org/pooch/).

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
- `uv run pytest -m network` — remote sample: one smoke test + the targeted regression tests. CI runs this on push to main.
- `uv run pytest -m network_full` — exhaustive walk over every registered Zenodo table; slow, opt-in only.

### Docs

Docs are Sphinx + `myst-nb`; dataset pages execute code cells at build time so previews reflect the live Zenodo data.

- `uv run sphinx-build -W -b html docs/ docs/_build/html` — full build (warnings are errors).
- Cold builds hit Zenodo for every preview; subsequent builds reuse the Pooch cache plus `docs/.jupyter_cache/` for near-instant rebuilds.

### Dataset doc generation

Dataset pages under `docs/datasets/<name>.md` are **generated from each fetcher's docstring** by `docs/scripts/gen_dataset_pages.py`. The generated `.md` files are committed to git.

After adding, renaming, or editing a dataset:

```bash
uv run python docs/scripts/gen_dataset_pages.py
git status   # review the diff; commit alongside the source change.
```

Always use `uv` when running Python scripts or installing dependencies. Never use bare `pip install` or `python`.

## Architecture

### Two source modules (`src/extracts/`)

- **`_common.py`** — Shared infrastructure: `DATASETS` registry, `DOI_PREFIX`, `get_location` / `set_location`, the `CacheParquet` Pooch processor, the `_create_pup` factory, and the discovery utilities (`list_available_datasets`, `list_available_tables`, `fetch_path`).
- **`_fetchers.py`** — The 16 fetcher functions: 9 per-paper (`fetch_barrett2020`, …, `fetch_paquet2020`), 5 LIWC manuals (`fetch_liwc1999` … `fetch_liwc22`), plus `fetch_text` and `fetch_reference`.

`src/extracts/__init__.py` re-exports the full public API; callers do `import extracts; extracts.fetch_*`.

Every fetcher takes `process: bool = True`:
- `process=True` (default): runs the per-fetcher `_processor` closure, caches the result as parquet next to the raw download via `CacheParquet`, and returns `pd.read_parquet(cache_path)`. Parquet round-trips MultiIndex columns and dtypes losslessly.
- `process=False`: skips the processor and returns `pd.read_table(raw_path, **kwargs)` so callers can supply their own read kwargs.

### Datasets

All remote data lives on Zenodo. Each dataset gets its own subdirectory under `get_location()` (e.g. `~/.cache/extracts/barrett2020/`). The `DATASETS` dict stores just the numeric Zenodo DOI suffix per version; `_create_pup` prepends the shared `DOI_PREFIX = "10.5281/zenodo."` programmatically.

## Docstring template

Every fetcher follows this shape. The `gen_dataset_pages.py` script parses it; deviating from the template will break doc generation. Ruff's `D` rules also enforce the numpy docstring convention.

```python
def fetch_X(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fetch tables from <one-line summary>.

    Citation
    --------
    <verbatim author/year/journal/title>,
    doi:`<doi> <https://doi.org/<doi>>`_

    Table captions
    --------------
    * **table1** — <verbatim caption>.
    * **table2** — <verbatim caption>.

    Notes
    -----
    <optional caveats, errata, parsing quirks>

    Parameters
    ----------
    table : str
        Name of desired table.
    version : str, optional
        Zenodo version key (defaults to ``"latest"``).
    process : bool, default True
        ...
    **kwargs
        Forwarded to :func:`pandas.read_table` when ``process=False``.
    """
```

## Code Style

- License: MIT (see `LICENSE.txt`).
- Ruff: line length 100, target Python 3.13, rules `E, F, I, W, C90, UP, NPY201, D`.
- Docstrings: NumPy convention (enforced via ruff D rules).
- Formatting: double quotes, LF line endings.
