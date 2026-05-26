---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# hawkins2017

Hawkins II & Boyd, 2017, *Dreaming*, Such stuff as dreams are made on: Dream language, LIWC norms, and personality correlates,
[doi:10.1037/drm0000049](https://doi.org/10.1037/drm0000049)

## Available tables

* **table1** — Means and Standard Deviations (SDs) for the LIWC (2007) Linguistic Features of Dreams From Studies 1 to 3.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_hawkins2017("table1")
df.head()
```

```{code-cell} python
df.shape, df.columns.nlevels
```

## Versions

```{code-cell} python
from extracts._fetchers import DATASETS

DATASETS["hawkins2017"]
```

## Notes

2007 Norms are a subset of the norms published in the LIWC2007 manual.

Ave. recent dream is UNWEIGHTED.
