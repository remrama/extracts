---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# liwc2007_manual

Tables extracted from the LIWC2007 Psychometrics Manual PDF, distributed on the [LIWC psychometrics manuals page](https://www.liwc.app/help/psychometrics-manuals).

Zenodo deposit: [10.5281/zenodo.11397699](https://doi.org/10.5281/zenodo.11397699)

## Available tables

* **table1** — LIWC2007 category descriptions with alpha (binary/raw).
* **table2** — Corpus summary statistics.
* **table3** — Per-corpus means and standard deviations.
* **table4** — LIWC2007 vs LIWC2001 cross-version correlations.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_liwc2007_manual("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._fetchers import DATASETS

DATASETS["liwc2007_manual"]
```
