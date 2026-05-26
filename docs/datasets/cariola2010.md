---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# cariola2010

Cariola, 2010, *unpublished paper*
Assessing the latent linguistic structure of oral dream narratives
url: <https://www.research.ed.ac.uk/en/publications/assessing-the-latent-linguistic-structure-of-oral-dream-narrative>

Zenodo deposit: [10.5281/zenodo.11301890](https://doi.org/10.5281/zenodo.11301890)

## Available tables

* **table1** — Descriptive statistics of linguistic variables in orally elicited
  dream narratives.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_cariola2010("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._common import DATASETS

DATASETS["cariola2010"]
```
