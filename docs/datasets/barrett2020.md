---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# barrett2020

Barrett, 2020, *Dreaming*
Dreams about COVID-19 versus normative dreams: Trends by gender
doi: [10.1037/drm0000149](https://doi.org/10.1037/drm0000149)

Zenodo deposit: [10.5281/zenodo.11300322](https://doi.org/10.5281/zenodo.11300322)

## Available tables

* **table1** — Female Pandemic Survey Dreams Versus Hall and Van de Castle Female
  Normative Dreams.
* **table2** — Male Pandemic Survey Dreams Versus Hall and Van de Castle Male Normative Dreams.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_barrett2020("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._common import DATASETS

DATASETS["barrett2020"]
```

## Notes

Table 2 has "male" in the column names, but Table 1 does not have "female"
in the same respective location. Note that Table 1 is female-only values.
