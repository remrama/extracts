---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# liwc22

LIWC-22 Psychometrics Manual, distributed on the
[LIWC website psychometrics manuals page](https://www.liwc.app/help/psychometrics-manuals).

Tables were extracted from the manual PDF and uploaded to Zenodo.

Zenodo deposit: [10.5281/zenodo.11397740](https://doi.org/10.5281/zenodo.11397740)

## Available tables

* **table1** — Corpus word-count summary.
* **table2** — Internal consistency per category.
* **table3** — Per-corpus means and SDs (MultiIndex columns).
* **table4** — LIWC-22 vs LIWC2015 cross-version correlations.
* **tableA1** — Test-kitchen corpus appendix.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_liwc22("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._common import DATASETS

DATASETS["liwc22"]
```
