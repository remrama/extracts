---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Datasets

Every dataset is hosted on [Zenodo](https://zenodo.org/) and accessed through a `fetch_<dataset>()` function. The table below summarizes what's currently available.

```{code-cell} python
:tags: [remove-input]

import pandas as pd
from IPython.display import HTML

import extracts
from extracts._fetchers import DATASETS, DOI_PREFIX

rows = []
for name in extracts.list_available_datasets():
    versions = DATASETS[name]
    latest_id = versions["latest"]
    pinned = sorted(v for v in versions if v != "latest")
    doi = f"{DOI_PREFIX}{latest_id}"
    rows.append(
        {
            "Dataset": f'<a href="{name}.html"><code>{name}</code></a>',
            "Pinned versions": ", ".join(pinned),
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
mariani2023
mcnamara2015
meador2022
niederhoffer2017
paquet2020
```
