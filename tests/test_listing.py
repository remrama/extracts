"""Tests for list_available_datasets and the static DATASETS dict shape."""

from __future__ import annotations

import extracts
from extracts._fetchers import DATASETS


def test_list_available_datasets_is_sorted() -> None:
    names = extracts.list_available_datasets()
    assert names == sorted(names)
    assert names == sorted(DATASETS)


def test_list_includes_known_dataset() -> None:
    assert "barrett2020" in extracts.list_available_datasets()


def test_every_dataset_has_latest_key() -> None:
    for name, versions in DATASETS.items():
        assert "latest" in versions, f"{name} is missing a 'latest' version key"


def test_every_doi_value_is_numeric_string() -> None:
    """Each value should be just the numeric Zenodo deposit id (prefix stripped)."""
    for name, versions in DATASETS.items():
        for version_key, doi_id in versions.items():
            assert isinstance(doi_id, str) and doi_id, f"{name}[{version_key}] is empty"
            assert doi_id.isdigit(), (
                f"{name}[{version_key}] = {doi_id!r} should be digits only "
                "(prefix is added programmatically)"
            )
