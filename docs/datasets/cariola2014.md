---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# cariola2014

Cariola, 2014, *Imagin Cogn Pers*
Lexical tendencies of high and low barrier personalities in narratives of everyday and
dream memories
doi: [10.2190/IC.34.2.d](https://doi.org/10.2190/IC.34.2.d)

Zenodo deposit: [10.5281/zenodo.11301782](https://doi.org/10.5281/zenodo.11301782)

## Available tables

* **table1** — Univariate Results of Body Boundary Imagery and LIWC Linguistic Variables of
  Low and High Barrier Personalities in Narratives of Everyday Memories.
* **table2** — Univariate Results of Body Boundary Imagery and LIWC Linguistic Variables of
  Low and High Barrier Personalities in Narratives of Dream Memories.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_cariola2014("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._common import DATASETS

DATASETS["cariola2014"]
```
