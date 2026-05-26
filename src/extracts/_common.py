"""Shared infrastructure for the fetcher modules.

Contains the dataset registry, cache-location helpers, the Pooch processor,
and the Pooch factory. Fetcher functions live in :mod:`extracts._fetchers`
and import the helpers defined here.
"""

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pooch

# All datasets live on Zenodo; the prefix is the same for every deposit.
DOI_PREFIX = "10.5281/zenodo."

# Each dataset maps version keys to the numeric portion of the Zenodo DOI.
# "latest" is the Concept DOI that always resolves to the most recent version.
DATASETS: dict[str, dict[str, str]] = {
    "barrett2020": {
        "latest": "11300322",
        "v1": "11300323",
        "v2": "11355831",
        "v3": "11357746",
    },
    "cariola2010": {
        "latest": "11301890",
        "v1": "11301891",
        "v2": "11356829",
    },
    "cariola2014": {
        "latest": "11301782",
        "v1": "11301783",
        "v2": "11356781",
    },
    "hawkins2017": {
        "latest": "11321093",
        "v1": "11321094",
        "v2": "11356868",
    },
    "liwc1999": {
        "latest": "11397664",
        "v1": "11397665",
    },
    "liwc2001": {
        "latest": "11397687",
        "v1": "11397688",
    },
    "liwc2007": {
        "latest": "11397699",
        "v1": "11397700",
    },
    "liwc2015": {
        "latest": "11397709",
        "v1": "11397710",
    },
    "liwc22": {
        "latest": "11397740",
        "v1": "11397741",
        "v2": "19947917",
    },
    "mariani2023": {
        "latest": "11325393",
        "v1": "11325394",
        "v2": "11356900",
    },
    "mcnamara2015": {
        "latest": "11321666",
        "v1": "11321667",
        "v2": "11357019",
    },
    "meador2022": {
        "latest": "11300860",
        "v1": "11300861",
        "v2": "11357190",
        "v3": "11357228",
    },
    "niederhoffer2017": {
        "latest": "11293797",
        "v1": "11293798",
        "v2": "11357309",
    },
    "paquet2020": {
        "latest": "11324388",
        "v1": "11324389",
        "v2": "11357270",
        "v3": "11357642",
    },
}


################################################################################
# Cache location
################################################################################


def get_location() -> Path:
    """Return the local cache root directory.

    Resolves ``$EXTRACTS_DATA_DIR`` if set; otherwise falls back to the
    OS-appropriate cache from :func:`pooch.os_cache`.
    """
    root = os.environ.get("EXTRACTS_DATA_DIR")
    return Path(root) if root else pooch.os_cache("extracts")


def set_location(path: str | Path) -> None:
    """Override the cache root via ``$EXTRACTS_DATA_DIR`` for this process.

    Parameters
    ----------
    path : str or :class:`~pathlib.Path`
        New cache root. ``~`` is expanded and the path is resolved to absolute.
    """
    os.environ["EXTRACTS_DATA_DIR"] = str(Path(path).expanduser().resolve())


################################################################################
# Pooch processor
################################################################################


class CacheParquet:
    """Pooch processor that parses a source file via ``build_fn`` and caches it as parquet.

    On the first download (or after an update), ``build_fn`` is called on the
    raw downloaded file and the resulting DataFrame is written next to the
    source as ``cache_name``. On subsequent fetches the cached parquet path
    is returned directly with no re-parsing.

    Parquet round-trips MultiIndex headers and dtypes losslessly, so the
    caller can :func:`pandas.read_parquet` the cached file without knowing
    anything about the original layout.
    """

    def __init__(
        self,
        build_fn: Callable[[Path], pd.DataFrame],
        cache_name: str,
    ) -> None:
        self.build_fn = build_fn
        self.cache_name = cache_name

    def __call__(self, fname: str, action: str, pup: pooch.Pooch) -> str:
        cache_path = Path(fname).parent / self.cache_name
        if action == "fetch" and cache_path.exists():
            return str(cache_path)
        df = self.build_fn(Path(fname))
        df.to_parquet(cache_path)
        return str(cache_path)


################################################################################
# Pooch factory
################################################################################


def _create_pup(dataset: str, version: str = "latest") -> pooch.Pooch:
    doi_id = DATASETS[dataset][version]
    base_url = f"doi:{DOI_PREFIX}{doi_id}"
    storage = get_location() / dataset
    pup = pooch.create(path=storage, base_url=base_url, registry=None)
    pup.load_registry_from_doi()
    return pup


################################################################################
# Discovery utilities
################################################################################


def list_available_datasets() -> list[str]:
    """Return the sorted list of dataset names known to this package.

    Examples
    --------
    >>> "barrett2020" in list_available_datasets()
    True
    """
    return sorted(DATASETS)


def list_available_tables(dataset: str, version: str | None = None) -> list[str]:
    """Return the filenames registered for ``dataset`` at ``version``."""
    return list(_create_pup(dataset, version or "latest").registry_files)


def fetch_path(dataset: str, table: str, version: str | None = None) -> Path:
    """Return the local path of the downloaded raw file.

    Parameters
    ----------
    dataset : str
        Dataset name from :func:`list_available_datasets`.
    table : str
        Filename in the Zenodo registry. If no extension is given, ``.tsv``
        is assumed.
    version : str, optional
        Zenodo version key (defaults to ``"latest"``).
    """
    pup = _create_pup(dataset, version or "latest")
    fname = table if "." in table else f"{table}.tsv"
    return Path(pup.fetch(fname))
