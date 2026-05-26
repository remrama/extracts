---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Datasets

Every dataset is hosted on [Zenodo](https://zenodo.org/) and accessed through a `fetch_<dataset>()` function. Per-paper datasets carry the author's surname + publication year (e.g. `barrett2020`); the `liwc<NNNN>_manual` entries are tables extracted from the LIWC Psychometrics Manual PDFs. The table below summarizes what's currently available.

```{code-cell} python
:tags: [remove-input]

import pandas as pd
from IPython.display import HTML

import extracts
from extracts._fetchers import DATASETS, DOI_PREFIX

# Build one row per dataset registered in extracts.DATASETS.
rows = []
for name in extracts.list_available_datasets():
    versions = DATASETS[name]
    latest_id = versions["latest"]
    pinned = sorted(v for v in versions if v != "latest")
    doi = f"{DOI_PREFIX}{latest_id}"
    rows.append(
        {
            "Dataset": f'<a href="{name}.html"><code>{name}</code></a>',
            "Pinned versions": ", ".join(pinned) or "—",
            "Latest DOI": f'<a href="https://doi.org/{doi}">{doi}</a>',
        }
    )

catalog = pd.DataFrame(rows).set_index("Dataset")
HTML(catalog.to_html(escape=False))
```

```{toctree}
:hidden:

barrett2020
cariola2010
cariola2014
hawkins2017
liwc1999
liwc2001
liwc2007
liwc2015
liwc22
mariani2023
mcnamara2015
meador2022
niederhoffer2017
paquet2020
```
