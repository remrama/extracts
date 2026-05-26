---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# mcnamara2015

McNamara et al., 2015, *Dreaming*, Aggression in nightmares and unpleasant dreams and in people reporting recurrent nightmares,
[doi:10.1037/a0039273](https://doi.org/10.1037/a0039273)

## Available tables

* **table1** — LIWC and Content Scale Means and SDs Across All Types of Dreams With LIWC Norms.
* **table6** — Categorical Comparisons Between Nightmares That Woke A Dreamer Up to Nightmares Where the Dreamer Was Not Woken Up.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_mcnamara2015("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._fetchers import DATASETS

DATASETS["mcnamara2015"]
```
