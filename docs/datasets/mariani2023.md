---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# mariani2023

Mariani et al., 2023, *Psychoanal Psychol*, Referential processes in dreams: A brief report from a COVID-19 dreams analysis,
[doi:10.1037/pap0000420](https://doi.org/10.1037/pap0000420)

## Available tables

* **table1** — ANOVA One Way Between Dreams' Clusters and LIWC Text Analysis.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_mariani2023("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._fetchers import DATASETS

DATASETS["mariani2023"]
```
