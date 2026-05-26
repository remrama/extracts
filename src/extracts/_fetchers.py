"""Fetchers for tables, text, and bibliographic references hosted on Zenodo."""

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
# Utility functions
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


################################################################################
# Special fetchers (text and reference)
################################################################################


def fetch_text(
    dataset: str,
    version: str | None = None,
    process: bool = True,
) -> str | Path:
    """Fetch the ``text.json`` file for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name.
    version : str, optional
        Zenodo version key (defaults to ``"latest"``).
    process : bool, default True
        If True (default), return the file contents as a string. If False,
        return the local :class:`~pathlib.Path` to the downloaded file.
    """
    pup = _create_pup(dataset, version or "latest")
    raw_path = Path(pup.fetch("text.json"))
    if not process:
        return raw_path
    return raw_path.read_text(encoding="utf-8")


def fetch_reference(
    dataset: str,
    version: str | None = None,
    process: bool = True,
) -> dict[str, Any] | Path:
    """Fetch the BibTeX ``reference.bib`` file for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name.
    version : str, optional
        Zenodo version key (defaults to ``"latest"``).
    process : bool, default True
        If True (default), parse the BibTeX entry and return a dict with
        ``type``, ``key``, and ``fields`` keys. If False, return the local
        :class:`~pathlib.Path` to the downloaded file.
    """
    pup = _create_pup(dataset, version or "latest")
    raw_path = Path(pup.fetch("reference.bib"))
    if not process:
        return raw_path
    data = raw_path.read_text(encoding="utf-8")
    type_match = re.search(r"^@(\w+){", data)
    key_match = re.search(r"{(\w+),$", data, re.MULTILINE)
    assert type_match is not None, "reference.bib missing @type{ header"
    assert key_match is not None, "reference.bib missing citation key"
    entry_fields = dict(re.findall(r"^\s{4}(\w+)\s+=\s+{(.*)},$", data, re.MULTILINE))
    return {"type": type_match.group(1), "key": key_match.group(1), "fields": entry_fields}


################################################################################
# Table fetchers
################################################################################


def fetch_barrett2020(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Barrett, 2020, *Dreaming*,
    Dreams about COVID-19 versus normative dreams: Trends by gender,
    doi:`10.1037/drm0000149 <https://doi.org/10.1037/drm0000149>`_

    Table captions
    --------------
    * **Table 1:** Female Pandemic Survey Dreams Versus Hall and Van de Castle Female
      Normative Dreams.
    * **Table 2:** Male Pandemic Survey Dreams Versus Hall and Van de Castle Male Normative Dreams.

    Notes
    -----
    Table 2 has "male" in the column names, but Table 1 does not have "female"
    in the same respective location. Note that Table 1 is female-only values.

    Parameters
    ----------
    table : str
        Name of desired table.
    version : str, optional
        Zenodo version key (defaults to ``"latest"``).
    process : bool, default True
        If True, return the processed (cached) DataFrame. If False, return the
        raw DataFrame loaded with :func:`pandas.read_table` and ``**kwargs``.
    **kwargs
        Forwarded to :func:`pandas.read_table` when ``process=False``.
    """
    pup = _create_pup("barrett2020", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        return pd.read_table(source_path, index_col=0)

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_cariola2010(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Cariola, 2010, *unpublished paper*,
    Assessing the latent linguistic structure of oral dream narratives,
    url:`<https://www.research.ed.ac.uk/en/publications/assessing-the-latent-linguistic-structure-of-oral-dream-narrative>`_

    Table captions
    --------------
    * **Table 1:** Descriptive statistics of linguistic variables in orally elicited
      dream narratives.
    """
    pup = _create_pup("cariola2010", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        return pd.read_table(source_path, index_col=0)

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_cariola2014(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Cariola, 2014, *Imagin Cogn Pers*,
    Lexical tendencies of high and low barrier personalities in narratives of everyday and
    dream memories,
    doi:`10.2190/IC.34.2.d <https://doi.org/10.2190/IC.34.2.d>`_

    Table captions
    --------------
    * **Table 1:** Univariate Results of Body Boundary Imagery and LIWC Linguistic Variables of
      Low and High Barrier Personalities in Narratives of Everyday Memories.
    * **Table 2:** Univariate Results of Body Boundary Imagery and LIWC Linguistic Variables of
      Low and High Barrier Personalities in Narratives of Dream Memories.
    """
    pup = _create_pup("cariola2014", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        return pd.read_table(source_path, index_col=0, header=[0, 1, 2])

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_hawkins2017(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Hawkins II & Boyd, 2017, *Dreaming*,
    Such stuff as dreams are made on: Dream language, LIWC norms, and personality correlates,
    Dreams about COVID-19 versus normative dreams: Trends by gender,
    doi:`10.1037/drm0000049 <https://doi.org/10.1037/drm0000049>`_

    Table captions
    --------------
    * **Table 1:** Means and Standard Deviations (SDs) for the LIWC (2007) Linguistic Features
      of Dreams From Studies 1 to 3.

    Notes
    -----
    2007 Norms are a subset of the norms published in the LIWC2007 manual.

    Ave. recent dream is UNWEIGHTED.
    """
    pup = _create_pup("hawkins2017", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        return pd.read_table(source_path, index_col=0, header=[0, 1])

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_mariani2023(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Mariani et al., 2023, *Psychoanal Psychol*,
    Referential processes in dreams: A brief report from a COVID-19 dreams analysis,
    doi:`10.1037/pap0000420 <https://doi.org/10.1037/pap0000420>`_

    Table captions
    --------------
    * **Table 1:** ANOVA One Way Between Dreams' Clusters and LIWC Text Analysis.
    """
    pup = _create_pup("mariani2023", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        return pd.read_table(source_path, index_col=0)

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_mcnamara2015(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """McNamara et al., 2015, *Dreaming*,
    Aggression in nightmares and unpleasant dreams and in people reporting recurrent nightmares,
    doi:`10.1037/a0039273 <https://doi.org/10.1037/a0039273>`_

    Table captions
    --------------
    * **Table 1:** LIWC and Content Scale Means and SDs Across All Types of Dreams With LIWC Norms.
    * **Table 6:** Categorical Comparisons Between Nightmares That Woke A Dreamer Up to
      Nightmares Where the Dreamer Was Not Woken Up.
    """
    pup = _create_pup("mcnamara2015", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        return pd.read_table(source_path, index_col=0)

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_meador2022(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Meador et al., 2022, *Appl Cognit Psychol*,
    Lexical tendencies of high and low barrier personalities in narratives of everyday and
    dream memories,
    doi:`10.1002/acp.3976 <https://doi.org/10.1002/acp.3976>`_

    Table captions
    --------------
    * **Table 1:** Change in symptoms and language.
    """
    pup = _create_pup("meador2022", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        return pd.read_table(source_path, index_col=0, skiprows=[5])

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_niederhoffer2017(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Niederhoffer et al., 2017, *CLPsych*,
    In your wildest dreams: the language and psychological features of dreams
    doi:`10.18653/v1/W17-3102 <https://doi.org/10.18653/v1/W17-3102>`_

    PDF available at https://aclanthology.org/W17-3102.pdf

    Table captions
    --------------
    * **Table 1:** Linguistic Processes Categories in LIWC2015.
    * **Table 2:** Top and Bottom Five dream Topics on CDI continuum.
    * **Table 3:** Most positively and negatively-correlated topics for each emotion.
    * **Appendix A:** Full list of LDA topics.
    * **Appendix B:** Sample dreams by CDI.

    Notes
    -----
    I corrected a typo in Table 2 (``plave`` -> ``plane``).
    The correct spelling is "plane", as you can see it in the corresponding Topic in Appendix A.
    """
    pup = _create_pup("niederhoffer2017", version or "latest")
    read_kwargs: dict[str, Any]
    if table == "table1":
        read_kwargs = dict(index_col=0, header=[0, 1])
    elif table in ("table2", "table3", "appendixA"):
        read_kwargs = dict(index_col=0)
    elif table == "appendixB":
        read_kwargs = dict(header=None)
    else:
        read_kwargs = dict(index_col=0)

    def _processor(source_path: Path) -> pd.DataFrame:
        return pd.read_table(source_path, **read_kwargs)

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_paquet2020(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Paquet et al., 2020, *Dreaming*,
    A quantitative text analysis approach to describing posttrauma nightmares in a
    treatment-seeking population,
    doi:`10.1037/drm0000128 <https://doi.org/10.1037/drm0000128>`_

    Table captions
    --------------
    * **Table 1:** Participant Demographics by Group.
    * **Table 2:** Psychological Diagnosis and Nightmare Qualities Experienced by Sample.
    * **Table 3:** Results Table of LIWC Variables.
    """
    pup = _create_pup("paquet2020", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        return pd.read_table(source_path, index_col=0)

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)
