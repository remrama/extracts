# API

```{eval-rst}
.. currentmodule:: extracts
```

The public API is intentionally small. For per-dataset usage examples, see the [Datasets catalog](datasets/index.md).

## Dataset discovery

```{eval-rst}
.. autofunction:: list_available_datasets
.. autofunction:: list_available_tables
.. autofunction:: fetch_path
```

## Cache location

```{eval-rst}
.. autofunction:: get_location
.. autofunction:: set_location
```

## Text and references

```{eval-rst}
.. autofunction:: fetch_text
.. autofunction:: fetch_reference
```

## Table fetchers

```{eval-rst}
.. autofunction:: fetch_barrett2020
.. autofunction:: fetch_cariola2010
.. autofunction:: fetch_cariola2014
.. autofunction:: fetch_hawkins2017
.. autofunction:: fetch_mariani2023
.. autofunction:: fetch_mcnamara2015
.. autofunction:: fetch_meador2022
.. autofunction:: fetch_niederhoffer2017
.. autofunction:: fetch_paquet2020
```

## LIWC Psychometrics Manual fetchers

```{eval-rst}
.. autofunction:: fetch_liwc1999
.. autofunction:: fetch_liwc2001
.. autofunction:: fetch_liwc2007
.. autofunction:: fetch_liwc2015
.. autofunction:: fetch_liwc22
```
