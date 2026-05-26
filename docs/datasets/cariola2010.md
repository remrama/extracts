---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# cariola2010

Cariola, 2010, *unpublished paper*, Assessing the latent linguistic structure of oral dream narratives,
[Edinburgh Research Explorer](https://www.research.ed.ac.uk/en/publications/assessing-the-latent-linguistic-structure-of-oral-dream-narrative)

## Available tables

* **table1** — Descriptive statistics of linguistic variables in orally elicited dream narratives.

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
from extracts._fetchers import DATASETS

DATASETS["cariola2010"]
```
