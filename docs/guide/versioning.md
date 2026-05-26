---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Versioning

Each dataset is published as a [Zenodo](https://zenodo.org/) deposit and may go through several revisions over time. Zenodo assigns two kinds of DOIs:

- A **Concept DOI** that always resolves to the latest version (exposed as the `"latest"` key).
- One **version-specific DOI** per revision (`"v1"`, `"v2"`, `"v3"`, ...).

`extracts` stores both forms internally. By default, fetchers use `"latest"` so you always get the most recent revision. Pass `version="v2"` (or whichever key you need) to pin to a specific revision.

## Inspecting available versions

```{code-cell} python
from extracts._fetchers import DATASETS

DATASETS["barrett2020"]
```

## Pinning a version

```{code-cell} python
import extracts

# Always-current (default)
df_latest = extracts.fetch_barrett2020("table1")

# Pinned to the v1 deposit
df_v1 = extracts.fetch_barrett2020("table1", version="v1")
```

Pinning is appropriate when you want reproducible analyses across time. For one-off exploration, the default `"latest"` is fine.

## Why the DOI prefix is implicit

Zenodo DOIs share the `10.5281/zenodo.` prefix, so `DATASETS` stores only the numeric suffix. The fetchers prepend the prefix programmatically via the `DOI_PREFIX` constant in [`_fetchers.py`](https://github.com/remrama/extracts/blob/main/src/extracts/_fetchers.py).
