"""
Fetchers

LOADERS??

Fetch/load raw tables.
Minimal processing is applied.
"""
from pathlib import Path
import re

import pandas as pd
import pooch


# "Latest" DOIs are "Concept" DOIs that always resolve to latest Zenodo version.
DATASETS = {
    "barrett2020": {
        "latest": "10.5281/zenodo.11300322",
        "v1": "10.5281/zenodo.11300323",
        "v2": "10.5281/zenodo.11355831",
        "v3": "10.5281/zenodo.11357746",
    },
    "cariola2010": {
        "latest": "10.5281/zenodo.11301890",
        "v1": "10.5281/zenodo.11301891",
        "v2": "10.5281/zenodo.11356829",
    },
    "cariola2014": {
        "latest": "10.5281/zenodo.11301782",
        "v1": "10.5281/zenodo.11301783",
        "v2": "10.5281/zenodo.11356781",
    },
    "hawkins2017": {
        "latest": "10.5281/zenodo.11321093",
        "v1": "10.5281/zenodo.11321094",
        "v2": "10.5281/zenodo.11356868",
    },
    "mariani2023": {
        "latest": "10.5281/zenodo.11325393",
        "v1": "10.5281/zenodo.11325394",
        "v2": "10.5281/zenodo.11356900",
    },
    "mcnamara2015": {
        "latest": "10.5281/zenodo.11321666",
        "v1": "10.5281/zenodo.11321667",
        "v2": "10.5281/zenodo.11357019",
    },
    "meador2022": {
        "latest": "10.5281/zenodo.11300860",
        "v1": "10.5281/zenodo.11300861",
        "v2": "10.5281/zenodo.11357190",
        "v3": "10.5281/zenodo.11357228",
    },
    "niederhoffer2017": {
        "latest": "10.5281/zenodo.11293797",
        "v1": "10.5281/zenodo.11293798",
        "v2": "10.5281/zenodo.11357309",
    },
    "paquet2020": {
        "latest": "10.5281/zenodo.11324388",
        "v1": "10.5281/zenodo.11324389",
        "v2": "10.5281/zenodo.11357270",
        "v3": "10.5281/zenodo.11357642",
    },
}


################################################################################
# Utility functions
################################################################################

def _create_pup(dataset, version="latest"):
    """
    """
    doi = DATASETS[dataset][version]
    url = f"doi:{doi}"
    # joinpath with dataset so that files don't overwrite each other
    # This creates a folder extracts/extracts/Cache/<dataset>
    storage_location = pooch.os_cache("extracts").joinpath(dataset)
    pup = pooch.create(path=storage_location, base_url=url, registry=None)
    pup.load_registry_from_doi()
    return pup


def list_available_datasets():
    return sorted(DATASETS)


def list_available_tables(dataset, version):
    return _create_pup(dataset, version).registry_files


################################################################################
# Fetching functions
################################################################################


def fetch_text(dataset, version=None, **kwargs):
    fp = _create_pup(dataset, version).fetch("text.json", **kwargs)
    with open(fp, "rt", encoding="utf-8") as f:
        data = f.read()
    return data


def fetch_reference(dataset, version=None, **kwargs):
    """
    Parameters
    ----------
    dataset : str
        Name of dataset.
    """
    fp = _create_pup(dataset, version).fetch("reference.bib", **kwargs)
    with open(fp, "rt", encoding="utf-8") as f:
        data = f.read()
    entry_type = re.search(r"^@(\w+){", data).group(1)
    entry_key = re.search(r"{(\w+),$", data, re.MULTILINE).group(1)
    entry_fields = dict(re.findall(r"^\s{4}(\w+)\s+=\s+{(.*)},$", data, re.MULTILINE))
    entry = dict(type=entry_type, key=entry_key, fields=entry_fields)
    return entry
    # search = lambda field: re.search(fr"^\s{{4}}{field}\s+=\s+{{(.*)}},$", data, flags=re.MULTILINE).group(1)
    # return search(field)


def fetch_barrett2020(table, version=None, **kwargs):
    """
    Barrett, 2020, *Dreaming*,
    Dreams about COVID-19 versus normative dreams: Trends by gender,
    doi:`10.1037/drm0000149 <https://doi.org/10.1037/drm0000149>`_

    Table captions:

    * **Table 1:** Female Pandemic Survey Dreams Versus Hall and Van de Castle Female Normative Dreams.
    * **Table 2:** Male Pandemic Survey Dreams Versus Hall and Van de Castle Male Normative Dreams.

    .. note::
        Table 2 has "male" int the column names, but Table 1 does not have "female"
        in the same respective location. Note that Table 1 is female-only values.

    Parameters
    ----------
    table : str
        Name of desired table.
    version : int or None
        Version of zenodo repository. If None, defaults to latest version.
    **kwargs
        Optional keyword argument(s) passed to :meth:`~pooch.Pooch.fetch`.

    Returns
    -------
    df : :class:`~pandas.DataFrame`
        The desired table.
    """
    fp = _create_pup("barrett2020", version).fetch(f"{table}.tsv", **kwargs)
    return pd.read_table(fp, index_col=0)


def fetch_hawkins2017(table, version=None, **kwargs):
    """
    Hawkins II & Boyd, 2017, *Dreaming*,
    Such stuff as dreams are made on: Dream language, LIWC norms, and personality correlates,
    Dreams about COVID-19 versus normative dreams: Trends by gender,
    doi:`10.1037/drm0000049 <https://doi.org/10.1037/drm0000049>`_

    Table captions:

    * **Table 1:** Means and Standard Deviations (SDs) for the LIWC (2007) Linguistic Features of Dreams From Studies 1 to 3.

    .. note::
        2007 Norms are a subset of the norms published in the LIWC2007 manual.

    .. note::
        Ave. recent dream is UNWEIGHTED.

    Parameters
    ----------
    table : str
        Name of desired table.
    version : int or None
        Version of zenodo repository. If None, defaults to latest version.
    **kwargs
        Optional keyword argument(s) passed to :meth:`~pooch.Pooch.fetch`.

    Returns
    -------
    df : :class:`~pandas.DataFrame`
        The desired table.
    """
    fp = _create_pup("hawkins2017", version).fetch(f"{table}.tsv", **kwargs)
    return pd.read_table(fp, index_col=0, header=[0, 1])


def fetch_cariola2010(table, version=None, **kwargs):
    """
    Cariola, 2010, *unpublished paper*,
    Assessing the latent linguistic structure of oral dream narratives,
    url:`<https://www.research.ed.ac.uk/en/publications/assessing-the-latent-linguistic-structure-of-oral-dream-narrative>`_

    Table captions
    --------------
    * **Table 1:** Descriptive statistics of linguistic variables in orally elicited dream narratives.

    Parameters
    ----------
    table : str
        Name of desired table. Available tables are ``table1``.
    version : int or None
        Version of zenodo repository. If None, defaults to latest version.
    **kwargs : Additional keyword arguments
        Optional keyword argument(s) passed to :meth:`~pooch.Pooch.fetch`.

    Returns
    -------
    df : :class:`~pandas.DataFrame`
        The desired table.
    """
    # dataset = inspect.stack()[0][3].split("_")[-1]
    fp = _create_pup("cariola2010", version).fetch(f"{table}.tsv", **kwargs)
    return pd.read_table(fp, index_col=0)


def fetch_cariola2014(table, version=None, **kwargs):
    """
    Cariola, 2014, *Imagin Cogn Pers*,
    Lexical tendencies of high and low barrier personalities in narratives of everyday and dream memories,
    doi:`10.2190/IC.34.2.d <https://doi.org/10.2190/IC.34.2.d>`_

    Table captions
    --------------
    * **Table 1:** Univariate Results of Body Boundary Imagery and LIWC Linguistic Variables of Low and High Barrier Personalities in Narratives of Everyday Memories.
    * **Table 2:** Univariate Results of Body Boundary Imagery and LIWC Linguistic Variables of Low and High Barrier Personalities in Narratives of Dream Memories.

    Parameters
    ----------
    table : str
        Name of desired table. Available tables are ``table1`` and ``table2``.
    version : int or None
        Version of zenodo repository. If None, defaults to latest version.
    **kwargs : Additional keyword arguments
        Optional keyword argument(s) passed to :meth:`~pooch.Pooch.fetch`.

    Returns
    -------
    df : :class:`~pandas.DataFrame`
        The desired table.
    """
    # dataset = inspect.stack()[0][3].split("_")[-1]
    fp = _create_pup("cariola2014", version).fetch(f"{table}.tsv", **kwargs)
    return pd.read_table(fp, header=[0, 1, 2], index_col=0)


def fetch_mcnamara2015(table, version=None, **kwargs):
    """
    McNamara et al., 2015, *Dreaming*,
    Aggression in nightmares and unpleasant dreams and in people reporting recurrent nightmares,
    doi:`10.1037/a0039273 <https://doi.org/10.1037/a0039273>`_

    Table captions
    --------------
    * **Table 1:** LIWC and Content Scale Means and SDs Across All Types of Dreams With LIWC Norms.
    * **Table 6:** Categorical Comparisons Between Nightmares That Woke A Dreamer Up to Nightmares Where the Dreamer Was Not Woken Up.

    Parameters
    ----------
    table : str
        Name of desired table. Available tables are ``table1`` and ``table6``.
    version : int or None
        Version of zenodo repository. If None, defaults to latest version.
    **kwargs
        Optional keyword argument(s) passed to :meth:`~pooch.Pooch.fetch`.

    Returns
    -------
    df : :class:`~pandas.DataFrame`
        The desired table.
    """
    fp = _create_pup("mcnamara2015", version).fetch(f"{table}.tsv", **kwargs)
    return pd.read_table(fp, index_col=0)


def fetch_meador2022(table, version=None, **kwargs):
    """
    Meador et al., 2022, *Appl Cognit Psychol*,
    Lexical tendencies of high and low barrier personalities in narratives of everyday and dream memories,
    doi:`10.1002/acp.3976 <https://doi.org/10.1002/acp.3976>`_

    Table captions
    --------------
    * **Table 1:** Change in symptoms and language.

    Parameters
    ----------
    table : str
        Name of desired table. Available tables are ``table1``.
    version : int or None
        Version of zenodo repository. If None, defaults to latest version.
    **kwargs
        Optional keyword argument(s) passed to :meth:`~pooch.Pooch.fetch`.

    Returns
    -------
    df : :class:`~pandas.DataFrame`
        The desired table.
    """
    fp = _create_pup("meador2022", version).fetch(f"{table}.tsv", **kwargs)
    return pd.read_table(fp, index_col=0, skiprows=[5])


def fetch_niederhoffer2017(table, version=None, **kwargs):
    """
    Niederhoffer et al., 2017, *CLPsych*,
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

    Parameters
    ----------
    table : str
        Name of desired table.
        Available tables are ``table1``, ``table2``, ``table3``, ``appendixA``, and ``appendixB``.
    version : int or None
        Version of zenodo repository. If None, defaults to latest version.
    **kwargs
        Optional keyword argument(s) passed to :meth:`~pooch.Pooch.fetch`.

    Returns
    -------
    df : :class:`~pandas.DataFrame`
        The desired table.

    Notes
    -----
    I corrected a typo in Table 2 (``plave`` -> ``plane``).
    The correct spelling is "plane", as you can see it in the corresponding Topic in Appendix A.
    """
    fp = _create_pup("niederhoffer2017", version).fetch(f"{table}.tsv", **kwargs)
    if table == "table1":
        kwargs = dict(index_col=0, header=[0, 1])
    elif table == "table2":
        kwargs = dict(index_col=0)
    elif table == "table3":
        kwargs = dict(index_col=0)
    elif table == "appendixA":
        kwargs = dict(index_col=0)
    elif table == "appendixB":
        kwargs = dict(header=None)
    return pd.read_table(fp, **kwargs)


def fetch_paquet2020(table, version=None, **kwargs):
    """
    Paquet et al., 2020, *Dreaming*,
    A quantitative text analysis approach to describing posttrauma nightmares in a treatment-seeking population,
    doi:`10.1037/drm0000128 <https://doi.org/10.1037/drm0000128>`_

    Table captions
    --------------
    * **Table 1:** Participant Demographics by Group.
    * **Table 2:** Psychological Diagnosis and Nightmare Qualities Experienced by Sample.
    * **Table 3:** Results Table of LIWC Variables.

    Parameters
    ----------
    table : str
        Name of desired table. Available tables are ``table1``, ``table2``, and ``table3``.
    version : int or None
        Version of zenodo repository. If None, defaults to latest version.
    **kwargs
        Optional keyword argument(s) passed to :meth:`~pooch.Pooch.fetch`.

    Returns
    -------
    df : :class:`~pandas.DataFrame`
        The desired table.
    """
    fp = _create_pup("paquet2020", version).fetch(f"{table}.tsv", **kwargs)
    return pd.read_table(fp, index_col=0)


def fetch_mariani2023(table, version=None, **kwargs):
    """
    Mariani et al., 2023, *Psychoanal Psychol*,
    Referential processes in dreams: A brief report from a COVID-19 dreams analysis,
    doi:`10.1037/pap0000420 <https://doi.org/10.1037/pap0000420>`_

    Table captions
    --------------
    * **Table 1:** ANOVA One Way Between Dreams' Clusters and LIWC Text Analysis.

    Parameters
    ----------
    table : str
        Name of desired table.
    version : int or None
        Version of zenodo repository. If None, defaults to latest version.
    **kwargs
        Optional keyword argument(s) passed to :meth:`~pooch.Pooch.fetch`.

    Returns
    -------
    df : :class:`~pandas.DataFrame`
        The desired table.
    """
    fp = _create_pup("mariani2023", version).fetch(f"{table}.tsv", **kwargs)
    return pd.read_table(fp, index_col=0)

