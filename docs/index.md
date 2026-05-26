---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# extracts

**extracts** is a small Python helper for loading tables, figures, and text that have been manually extracted from published journal articles. Each dataset corresponds to a [Zenodo](https://zenodo.org/) deposit; files are downloaded and cached locally via [Pooch](https://www.fatiando.org/pooch/).

The motivating use case is meta-analysis: papers report results in tables that are not always machine-readable. `extracts` collects those tables, hosts them on Zenodo, and exposes a small set of Python fetchers so the data can be pulled into a DataFrame in one call.

## Install

```bash
pip install extracts
```

## Quick example

```{code-cell} python
import extracts

# Discover what's available
extracts.list_available_datasets()
```

```{code-cell} python
# Fetch a table from one of the datasets
df = extracts.fetch_barrett2020("table1")
df.head()
```

See the [Datasets catalog](datasets/index.md) for the full list, the [Guide](guide/index.md) for caching/versioning details, and the [API](api.md) for the utility functions.

```{toctree}
:hidden:

datasets/index
guide/index
api
```
