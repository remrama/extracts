---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# paquet2020

Paquet et al., 2020, *Dreaming*
A quantitative text analysis approach to describing posttrauma nightmares in a
treatment-seeking population
doi: [10.1037/drm0000128](https://doi.org/10.1037/drm0000128)

Zenodo deposit: [10.5281/zenodo.11324388](https://doi.org/10.5281/zenodo.11324388)

## Available tables

* **table1** — Participant Demographics by Group.
* **table2** — Psychological Diagnosis and Nightmare Qualities Experienced by Sample.
* **table3** — Results Table of LIWC Variables.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_paquet2020("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._common import DATASETS

DATASETS["paquet2020"]
```
