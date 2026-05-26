"""Network-gated tests: fetch real data from Zenodo.

Run with:
    uv run pytest tests/test_remote.py -m network         # default sample
    uv run pytest tests/test_remote.py -m network_full    # exhaustive (opt-in)

The default ``pytest`` invocation skips both markers via the ``-m 'not network
and not network_full'`` filter in pyproject.toml.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

import extracts


@pytest.mark.network
def test_fetch_smoke() -> None:
    """A single fetcher works end-to-end against live Zenodo."""
    df = extracts.fetch_barrett2020("table1")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.network
def test_fetch_path_returns_raw_tsv() -> None:
    path = extracts.fetch_path("barrett2020", "table1")
    assert isinstance(path, Path)
    assert path.exists()
    assert path.suffix == ".tsv"
    assert path.stat().st_size > 0


@pytest.mark.network
def test_process_false_uses_user_kwargs() -> None:
    df = extracts.fetch_barrett2020("table1", process=False, index_col=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.network
def test_multiindex_roundtrip_via_parquet() -> None:
    """hawkins2017 table1 has a 2-level column header; parquet must preserve it."""
    df = extracts.fetch_hawkins2017("table1")
    assert df.columns.nlevels == 2


@pytest.mark.network
def test_niederhoffer2017_per_table_branches() -> None:
    """The fetch_niederhoffer2017 dispatch routes table1/table2/appendixB differently."""
    t1 = extracts.fetch_niederhoffer2017("table1")
    assert t1.columns.nlevels == 2

    t2 = extracts.fetch_niederhoffer2017("table2")
    assert t2.columns.nlevels == 1

    apx_b = extracts.fetch_niederhoffer2017("appendixB")
    # appendixB is read with header=None, so columns are a single-level integer index.
    assert apx_b.columns.nlevels == 1
    assert apx_b.columns.dtype.kind in ("i", "O")


@pytest.mark.network
def test_liwc22_per_table_dispatch() -> None:
    """fetch_liwc22 routes table1, table4, and tableA1 through distinct parsers."""
    t1 = extracts.fetch_liwc22("table1")
    assert isinstance(t1, pd.DataFrame)
    assert t1.index.name == "corpus"

    t4 = extracts.fetch_liwc22("table4")
    assert isinstance(t4, pd.DataFrame)
    assert set(t4.columns) == {
        "liwc22_mean",
        "liwc22_sd",
        "liwc2015_mean",
        "liwc2015_sd",
        "r",
    }

    a1 = extracts.fetch_liwc22("tableA1")
    assert isinstance(a1, pd.DataFrame)
    assert a1.index.name == "corpus"
    assert "description" in a1.columns


@pytest.mark.network
def test_list_available_tables_hits_zenodo() -> None:
    tables = extracts.list_available_tables("barrett2020")
    assert isinstance(tables, list)
    assert "table1.tsv" in tables


@pytest.mark.network
def test_fetch_text_returns_str_then_path() -> None:
    """fetch_text returns content by default and the local Path with process=False."""
    candidate_datasets = extracts.list_available_datasets()
    for dataset in candidate_datasets:
        try:
            registry = extracts.list_available_tables(dataset)
        except Exception:
            continue
        if "text.json" in registry:
            content = extracts.fetch_text(dataset)
            assert isinstance(content, str)
            assert len(content) > 0
            path = extracts.fetch_text(dataset, process=False)
            assert isinstance(path, Path)
            assert path.exists()
            return
    pytest.skip("No dataset in this Zenodo registry currently ships text.json")


@pytest.mark.network
def test_fetch_reference_parses_bibtex() -> None:
    """barrett2020's reference.bib parses into the expected dict shape."""
    entry = extracts.fetch_reference("barrett2020")
    assert isinstance(entry, dict)
    assert set(entry.keys()) >= {"type", "key", "fields"}
    assert entry["type"]  # e.g. "article", "book"
    assert entry["key"]  # citation key
    assert isinstance(entry["fields"], dict)
    # BibTeX entries usually have at least one of these.
    assert any(k in entry["fields"] for k in ("author", "title", "year"))


@pytest.mark.network
def test_fetch_reference_path_with_process_false() -> None:
    path = extracts.fetch_reference("barrett2020", process=False)
    assert isinstance(path, Path)
    assert path.exists()
    assert path.suffix == ".bib"


# ---------------------------------------------------------------------------
# Opt-in exhaustive section. Not run by CI; available for local sanity-checks.
# Run with:  uv run pytest tests/test_remote.py -m network_full
# ---------------------------------------------------------------------------


@pytest.mark.network_full
@pytest.mark.parametrize("dataset", extracts.list_available_datasets())
def test_fetch_every_table(dataset: str) -> None:
    """Walk every .tsv in the dataset's Zenodo registry and load it."""
    fn = getattr(extracts, f"fetch_{dataset}")
    tables = extracts.list_available_tables(dataset)
    tsv_stems = [os.path.splitext(t)[0] for t in tables if t.endswith(".tsv")]
    assert tsv_stems, f"{dataset} has no .tsv files in its registry"
    for stem in tsv_stems:
        df = fn(stem)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, f"{dataset}/{stem} loaded empty"
