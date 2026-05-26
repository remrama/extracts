"""Fetchers for tables, text, and bibliographic references hosted on Zenodo.

Shared infrastructure (registry, cache helpers, Pooch processor and factory,
listing utilities) lives in :mod:`extracts._common`. This module only contains
the per-dataset fetch functions plus :func:`fetch_text` / :func:`fetch_reference`.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd

# DATASETS and DOI_PREFIX are re-exported here as a backwards-compatibility convenience
# for callers that previously imported them from ``extracts._fetchers``. New code should
# prefer ``from extracts._common import DATASETS``.
from ._common import DATASETS, DOI_PREFIX, CacheParquet, _create_pup  # noqa: F401

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
    """Fetch tables from Barrett (2020), *Dreaming*.

    Citation
    --------
    Barrett, 2020, *Dreaming*,
    Dreams about COVID-19 versus normative dreams: Trends by gender,
    doi:`10.1037/drm0000149 <https://doi.org/10.1037/drm0000149>`_

    Table captions
    --------------
    * **table1** — Female Pandemic Survey Dreams Versus Hall and Van de Castle Female
      Normative Dreams.
    * **table2** — Male Pandemic Survey Dreams Versus Hall and Van de Castle Male Normative Dreams.

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
    """Fetch tables from Cariola (2010), unpublished paper.

    Citation
    --------
    Cariola, 2010, *unpublished paper*,
    Assessing the latent linguistic structure of oral dream narratives,
    url:`<https://www.research.ed.ac.uk/en/publications/assessing-the-latent-linguistic-structure-of-oral-dream-narrative>`_

    Table captions
    --------------
    * **table1** — Descriptive statistics of linguistic variables in orally elicited
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
    """Fetch tables from Cariola (2014), *Imagin Cogn Pers*.

    Citation
    --------
    Cariola, 2014, *Imagin Cogn Pers*,
    Lexical tendencies of high and low barrier personalities in narratives of everyday and
    dream memories,
    doi:`10.2190/IC.34.2.d <https://doi.org/10.2190/IC.34.2.d>`_

    Table captions
    --------------
    * **table1** — Univariate Results of Body Boundary Imagery and LIWC Linguistic Variables of
      Low and High Barrier Personalities in Narratives of Everyday Memories.
    * **table2** — Univariate Results of Body Boundary Imagery and LIWC Linguistic Variables of
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
    """Fetch tables from Hawkins II & Boyd (2017), *Dreaming*.

    Citation
    --------
    Hawkins II & Boyd, 2017, *Dreaming*,
    Such stuff as dreams are made on: Dream language, LIWC norms, and personality correlates,
    Dreams about COVID-19 versus normative dreams: Trends by gender,
    doi:`10.1037/drm0000049 <https://doi.org/10.1037/drm0000049>`_

    Table captions
    --------------
    * **table1** — Means and Standard Deviations (SDs) for the LIWC (2007) Linguistic Features
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
    """Fetch tables from Mariani et al. (2023), *Psychoanal Psychol*.

    Citation
    --------
    Mariani et al., 2023, *Psychoanal Psychol*,
    Referential processes in dreams: A brief report from a COVID-19 dreams analysis,
    doi:`10.1037/pap0000420 <https://doi.org/10.1037/pap0000420>`_

    Table captions
    --------------
    * **table1** — ANOVA One Way Between Dreams' Clusters and LIWC Text Analysis.
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
    """Fetch tables from McNamara et al. (2015), *Dreaming*.

    Citation
    --------
    McNamara et al., 2015, *Dreaming*,
    Aggression in nightmares and unpleasant dreams and in people reporting recurrent nightmares,
    doi:`10.1037/a0039273 <https://doi.org/10.1037/a0039273>`_

    Table captions
    --------------
    * **table1** — LIWC and Content Scale Means and SDs Across All Types of Dreams With LIWC Norms.
    * **table6** — Categorical Comparisons Between Nightmares That Woke A Dreamer Up to
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
    """Fetch tables from Meador et al. (2022), *Appl Cognit Psychol*.

    Citation
    --------
    Meador et al., 2022, *Appl Cognit Psychol*,
    Lexical tendencies of high and low barrier personalities in narratives of everyday and
    dream memories,
    doi:`10.1002/acp.3976 <https://doi.org/10.1002/acp.3976>`_

    Table captions
    --------------
    * **table1** — Change in symptoms and language.
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
    """Fetch tables from Niederhoffer et al. (2017), *CLPsych*.

    Citation
    --------
    Niederhoffer et al., 2017, *CLPsych*,
    In your wildest dreams: the language and psychological features of dreams,
    doi:`10.18653/v1/W17-3102 <https://doi.org/10.18653/v1/W17-3102>`_

    PDF available at https://aclanthology.org/W17-3102.pdf.

    Table captions
    --------------
    * **table1** — Linguistic Processes Categories in LIWC2015.
    * **table2** — Top and Bottom Five dream Topics on CDI continuum.
    * **table3** — Most positively and negatively-correlated topics for each emotion.
    * **appendixA** — Full list of LDA topics.
    * **appendixB** — Sample dreams by CDI.

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
    """Fetch tables from Paquet et al. (2020), *Dreaming*.

    Citation
    --------
    Paquet et al., 2020, *Dreaming*,
    A quantitative text analysis approach to describing posttrauma nightmares in a
    treatment-seeking population,
    doi:`10.1037/drm0000128 <https://doi.org/10.1037/drm0000128>`_

    Table captions
    --------------
    * **table1** — Participant Demographics by Group.
    * **table2** — Psychological Diagnosis and Nightmare Qualities Experienced by Sample.
    * **table3** — Results Table of LIWC Variables.
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


################################################################################
# LIWC Psychometrics Manual fetchers
################################################################################


def fetch_liwc1999(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fetch tables from the LIWC1999 Psychometrics Manual.

    Citation
    --------
    LIWC1999 Psychometrics Manual, distributed on the
    `LIWC website psychometrics manuals page
    <https://www.liwc.app/help/psychometrics-manuals>`_.

    Tables were extracted from the manual PDF and uploaded to Zenodo.

    Table captions
    --------------
    * **table1** — LIWC1999 category descriptions (judges, examples, word counts).
    * **table2** — Corpus summary statistics.
    * **table3** — Per-category means and standard deviations.
    """
    pup = _create_pup("liwc1999", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        if table == "table1":
            return (
                pd.read_csv(source_path, sep="\t", dtype={"# Words": "Int8"})
                .rename(
                    columns={
                        "Dimension": "name",
                        "Abbrev": "category",
                        "Examples": "examples",
                        "# Words": "n_words",
                        "Judge 1": "judge1",
                        "Judge 2": "judge2",
                    }
                )
                .assign(parent=lambda x: x["name"].where(x["category"].isna()).ffill())
                .dropna(subset=["category"])
                .set_index(["parent", "name"])
            )
        if table == "table2":
            return (
                pd.read_csv(source_path, sep="\t", header=1, index_col=0, thousands=",")
                .drop(columns=["Totals"])
                .T.rename_axis("corpus")
                .rename(
                    columns={
                        "Number of files": "n_files",
                        "Number of writers/speakers": "n_authors",
                        "Number of words": "n_words",
                        "Number of studies": "n_studies",
                    }
                )
            )
        if table == "table3":
            return (
                pd.read_csv(source_path, sep="\t")
                .rename(columns={"Dimension": "name"})
                .assign(parent=lambda x: x["name"].where(x["name"].str.isupper()).ffill())
                .dropna()
                .pipe(
                    lambda x: x.assign(
                        **x["Mean (sd)"]
                        .str.extract(r"(?P<Mean>[\d.]+)\s*\((?P<SD>[\d.]+)\)")
                        .astype(float)
                    )
                )
                .drop(columns=["Mean (sd)"])
                .set_index(["parent", "name"])
            )
        raise ValueError(f"Unknown table {table!r} for liwc1999")

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_liwc2001(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fetch tables from the LIWC2001 Psychometrics Manual.

    Citation
    --------
    LIWC2001 Psychometrics Manual, distributed on the
    `LIWC website psychometrics manuals page
    <https://www.liwc.app/help/psychometrics-manuals>`_.

    Tables were extracted from the manual PDF and uploaded to Zenodo.
    Content matches the LIWC1999 manual tables; deposit is kept separate so
    each manual version has its own DOI.

    Table captions
    --------------
    * **table1** — LIWC2001 category descriptions (judges, examples, word counts).
    * **table2** — Corpus summary statistics.
    * **table3** — Per-category means and standard deviations.
    """
    pup = _create_pup("liwc2001", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        if table == "table1":
            return (
                pd.read_csv(source_path, sep="\t", dtype={"# Words": "Int8"})
                .rename(
                    columns={
                        "Dimension": "name",
                        "Abbrev": "category",
                        "Examples": "examples",
                        "# Words": "n_words",
                        "Judge 1": "judge1",
                        "Judge 2": "judge2",
                    }
                )
                .assign(parent=lambda x: x["name"].where(x["category"].isna()).ffill())
                .dropna(subset=["category"])
                .set_index(["parent", "name"])
            )
        if table == "table2":
            return (
                pd.read_csv(source_path, sep="\t", header=1, index_col=0, thousands=",")
                .drop(columns=["Totals"])
                .T.rename_axis("corpus")
                .rename(
                    columns={
                        "Number of files": "n_files",
                        "Number of writers/speakers": "n_authors",
                        "Number of words": "n_words",
                        "Number of studies": "n_studies",
                    }
                )
            )
        if table == "table3":
            return (
                pd.read_csv(source_path, sep="\t")
                .rename(columns={"Dimension": "name"})
                .assign(parent=lambda x: x["name"].where(x["name"].str.isupper()).ffill())
                .dropna()
                .pipe(
                    lambda x: x.assign(
                        **x["Mean (sd)"]
                        .str.extract(r"(?P<Mean>[\d.]+)\s*\((?P<SD>[\d.]+)\)")
                        .astype(float)
                    )
                )
                .drop(columns=["Mean (sd)"])
                .set_index(["parent", "name"])
            )
        raise ValueError(f"Unknown table {table!r} for liwc2001")

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_liwc2007(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fetch tables from the LIWC2007 Psychometrics Manual.

    Citation
    --------
    LIWC2007 Psychometrics Manual, distributed on the
    `LIWC website psychometrics manuals page
    <https://www.liwc.app/help/psychometrics-manuals>`_.

    Tables were extracted from the manual PDF and uploaded to Zenodo.

    Table captions
    --------------
    * **table1** — LIWC2007 category descriptions with alpha (binary/raw).
    * **table2** — Corpus summary statistics.
    * **table3** — Per-corpus means and standard deviations.
    * **table4** — LIWC2007 vs LIWC2001 cross-version correlations.
    """
    pup = _create_pup("liwc2007", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        if table == "table1":
            return (
                pd.read_csv(source_path, sep="\t", dtype={"Words in category": "Int8"})
                .assign(parent=lambda x: x["Category"].where(x["Abbrev"].isna()).ffill())
                .dropna(subset=["Abbrev"])
                .pipe(
                    lambda x: x.assign(
                        **x["Alpha: Binary/raw"]
                        .str.extract(r"(?P<alpha_binary>[\d.]+)/(?P<alpha_raw>[\d.]+)")
                        .astype(float)
                    )
                )
                .drop(columns=["Alpha: Binary/raw"])
                .rename(
                    columns={
                        "Category": "name",
                        "Abbrev": "category",
                        "Examples": "examples",
                        "Word in category": "n_words",
                        "Validity (judges)": "validity",
                    }
                )
                .set_index(["parent", "category"])
            )
        if table == "table2":
            return (
                pd.read_csv(source_path, sep="\t", index_col=0, thousands=",")
                .T.rename_axis("corpus")
                .rename(columns=lambda x: x.replace("Total ", "n_"))
            )
        if table == "table3":
            return (
                pd.read_csv(source_path, sep="\t")
                .assign(parent=lambda x: x["Category"].where(x["Novels"].isna()).ffill())
                .dropna(subset=["Novels"])
                .rename(columns={"Category": "category", "Grand Means": "Mean", "Mean SDs": "StD"})
                .set_index(["parent", "category"])
            )
        if table == "table4":
            return (
                pd.read_csv(source_path, sep="\t", header=[0, 1], index_col=0)
                .rename_axis("name")
                .set_axis(
                    ["liwc2007_mean", "liwc2007_sd", "liwc2001_mean", "liwc2001_sd", "r"],
                    axis=1,
                )
            )
        raise ValueError(f"Unknown table {table!r} for liwc2007")

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_liwc2015(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fetch tables from the LIWC2015 Psychometrics Manual.

    Citation
    --------
    LIWC2015 Psychometrics Manual, distributed on the
    `LIWC website psychometrics manuals page
    <https://www.liwc.app/help/psychometrics-manuals>`_.

    Tables were extracted from the manual PDF and uploaded to Zenodo.

    Table captions
    --------------
    * **table1** — LIWC2015 category descriptions with internal consistency.
    * **table2** — Corpus summary statistics.
    * **table3** — Per-corpus means and standard deviations.
    * **table4** — LIWC2015 vs LIWC2007 cross-version correlations.
    """
    pup = _create_pup("liwc2015", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        if table == "table1":
            return (
                pd.read_csv(source_path, sep="\t", na_values="-")
                .assign(
                    parent=lambda x: (
                        x["Category"].where(x["Abbrev"].isna()).ffill().fillna(x["Abbrev"])
                    )
                )
                .dropna(subset=["Abbrev"])
                .rename(
                    columns={
                        "Category": "name",
                        "Abbrev": "category",
                        "Example": "examples",
                        "Words in category": "n_words",
                        "Internal Consistency (Uncorrected alpha)": "alpha_uncorrected",
                        "Internal Consistency (Corrected alpha)": "alpha_corrected",
                    }
                )
                .set_index(["parent", "category"])
            )
        if table == "table2":
            return (
                pd.read_csv(source_path, sep="\t", index_col=0, thousands=",", na_values="Unknown")
                .astype("Int32")
                .T.rename_axis("corpus")
                .rename(columns=lambda x: x.replace("Total ", "n_"))
            )
        if table == "table3":
            return (
                pd.read_csv(source_path, sep="\t")
                .assign(parent=lambda x: x["Category"].where(x["Novels"].isna()).ffill())
                .dropna(subset=["Novels"])
                .rename(columns={"Category": "category", "Grand Means": "Mean", "Mean SDs": "StD"})
                .set_index(["parent", "category"])
            )
        if table == "table4":
            return (
                pd.read_csv(source_path, sep="\t", thousands=",", na_values=["-"])
                .assign(
                    parent=lambda x: (
                        x["LIWC Dimension"]
                        .where(x["Output Label"].isna())
                        .ffill()
                        .fillna(x["Output Label"])
                    )
                )
                .dropna(subset=["Output Label"])
                .rename(
                    columns={
                        "LIWC Dimension": "name",
                        "Output Label": "category",
                        "LIWC2015 mean": "liwc2015_mean",
                        "LIWC2007 mean": "liwc2007_mean",
                        "LIWC 2015/2007 Correlation": "r",
                    }
                )
                .set_index(["parent", "category"])
            )
        raise ValueError(f"Unknown table {table!r} for liwc2015")

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)


def fetch_liwc22(
    table: str,
    version: str | None = None,
    process: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fetch tables from the LIWC-22 Psychometrics Manual.

    Citation
    --------
    LIWC-22 Psychometrics Manual, distributed on the
    `LIWC website psychometrics manuals page
    <https://www.liwc.app/help/psychometrics-manuals>`_.

    Tables were extracted from the manual PDF and uploaded to Zenodo.

    Table captions
    --------------
    * **table1** — Corpus word-count summary.
    * **table2** — Internal consistency per category.
    * **table3** — Per-corpus means and SDs (MultiIndex columns).
    * **table4** — LIWC-22 vs LIWC2015 cross-version correlations.
    * **tableA1** — Test-kitchen corpus appendix.
    """
    pup = _create_pup("liwc22", version or "latest")

    def _processor(source_path: Path) -> pd.DataFrame:
        if table == "table1":
            return (
                pd.read_csv(source_path, sep="\t")
                .replace({"Corpus": {"Overall mean": "Total"}})
                .pipe(
                    lambda x: x.assign(
                        **x["Word Count M (SD)"]
                        .str.extract(r"(?P<n_words_mean>\d+) \((?P<n_words_sd>\d+)\)")
                        .astype(int)
                    )
                )
                .drop(columns=["Word Count M (SD)"])
                .rename(columns={"Corpus": "corpus", "Description": "description"})
                .set_index("corpus")
                .reindex(columns=["n_words_mean", "n_words_sd", "description"])
            )
        if table == "table2":
            return (
                pd.read_csv(source_path, sep="\t", na_values=["-"])
                .assign(
                    parent=lambda x: (
                        x["Category"].where(x["Abbrev."].isna()).ffill().fillna(x["Abbrev."])
                    )
                )
                .dropna(subset=["Abbrev."])
                .rename(
                    columns={
                        "Category": "name",
                        "Abbrev.": "category",
                        "Description/Most frequently used exemplars": "examples",
                        "Words/Entries in category": "n_words",
                        "Internal Consistency: Cronbach's alpha": "alpha",
                        "Internal Consistency: KR-20": "kr20",
                    }
                )
                .set_index(["parent", "category"])
            )
        if table == "table3":
            _df = (
                pd.read_csv(source_path, sep="\t", skiprows=[1, 2], na_values=["mean", "SD"])
                .assign(parent=lambda x: x["Category"].where(x["Twitter"].isna()).ffill())
                .dropna(subset=["Twitter"])
                .rename(columns={"Category": "name"})
                .set_index(["parent", "name"])
            )
            columns = pd.Series(_df.columns).replace(r"^Unnamed: \d+", pd.NA, regex=True).ffill()
            _df.columns = pd.MultiIndex.from_product(
                (columns.unique(), ["mean", "sd"]), names=("corpus", "statistic")
            )
            return _df
        if table == "table4":
            return pd.read_csv(
                source_path,
                sep="\t",
                skiprows=3,
                names=["liwc22_mean", "liwc22_sd", "liwc2015_mean", "liwc2015_sd", "r"],
            )
        if table == "tableA1":
            return (
                pd.read_csv(source_path, sep="\t", thousands=",")
                .rename(
                    columns={
                        "Corpus": "corpus",
                        "Description": "description",
                        "Test Kitchen N": "n_files",
                        "Years Written": "timeframe",
                        "Population N": "n_authors",
                    }
                )
                .set_index("corpus")
                .reindex(columns=["n_files", "n_authors", "timeframe", "description"])
            )
        raise ValueError(f"Unknown table {table!r} for liwc22")

    if process:
        cache_path = pup.fetch(
            f"{table}.tsv",
            processor=CacheParquet(_processor, f"{table}.parquet"),
        )
        return pd.read_parquet(cache_path)
    raw_path = pup.fetch(f"{table}.tsv")
    return pd.read_table(raw_path, **kwargs)
