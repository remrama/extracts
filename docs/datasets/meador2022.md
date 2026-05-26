---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# meador2022

Meador et al., 2022, *Appl Cognit Psychol*, Lexical tendencies of high and low barrier personalities in narratives of everyday and dream memories,
[doi:10.1002/acp.3976](https://doi.org/10.1002/acp.3976)

## Available tables

* **table1** — Change in symptoms and language.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_meador2022("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._fetchers import DATASETS

DATASETS["meador2022"]
```
