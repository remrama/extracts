---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# liwc1999_manual

Tables extracted from the LIWC1999 Psychometrics Manual PDF, distributed on the [LIWC psychometrics manuals page](https://www.liwc.app/help/psychometrics-manuals).

Zenodo deposit: [10.5281/zenodo.11397664](https://doi.org/10.5281/zenodo.11397664)

## Available tables

* **table1** — LIWC1999 category descriptions (judges, examples, word counts).
* **table2** — Corpus summary statistics.
* **table3** — Per-category means and standard deviations.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_liwc1999_manual("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._fetchers import DATASETS

DATASETS["liwc1999_manual"]
```
