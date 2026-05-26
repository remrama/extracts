---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# liwc2015_manual

Tables extracted from the LIWC2015 Psychometrics Manual PDF, distributed on the [LIWC psychometrics manuals page](https://www.liwc.app/help/psychometrics-manuals).

Zenodo deposit: [10.5281/zenodo.11397709](https://doi.org/10.5281/zenodo.11397709)

## Available tables

* **table1** — LIWC2015 category descriptions with internal consistency.
* **table2** — Corpus summary statistics.
* **table3** — Per-corpus means and standard deviations.
* **table4** — LIWC2015 vs LIWC2007 cross-version correlations.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_liwc2015_manual("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._fetchers import DATASETS

DATASETS["liwc2015_manual"]
```
