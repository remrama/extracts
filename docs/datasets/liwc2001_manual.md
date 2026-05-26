---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# liwc2001_manual

Tables extracted from the LIWC2001 Psychometrics Manual PDF, distributed on the [LIWC psychometrics manuals page](https://www.liwc.app/help/psychometrics-manuals).

Zenodo deposit: [10.5281/zenodo.11397687](https://doi.org/10.5281/zenodo.11397687)

## Available tables

* **table1** — LIWC2001 category descriptions (judges, examples, word counts).
* **table2** — Corpus summary statistics.
* **table3** — Per-category means and standard deviations.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_liwc2001_manual("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._fetchers import DATASETS

DATASETS["liwc2001_manual"]
```
