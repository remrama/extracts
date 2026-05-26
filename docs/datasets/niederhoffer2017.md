---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# niederhoffer2017

Niederhoffer et al., 2017, *CLPsych*
In your wildest dreams: the language and psychological features of dreams
doi: [10.18653/v1/W17-3102](https://doi.org/10.18653/v1/W17-3102)

PDF available at https://aclanthology.org/W17-3102.pdf.

Zenodo deposit: [10.5281/zenodo.11293797](https://doi.org/10.5281/zenodo.11293797)

## Available tables

* **table1** — Linguistic Processes Categories in LIWC2015.
* **table2** — Top and Bottom Five dream Topics on CDI continuum.
* **table3** — Most positively and negatively-correlated topics for each emotion.
* **appendixA** — Full list of LDA topics.
* **appendixB** — Sample dreams by CDI.

## Preview

```{code-cell} python
import extracts

df = extracts.fetch_niederhoffer2017("table1")
df.head()
```

```{code-cell} python
df.shape
```

## Versions

```{code-cell} python
from extracts._common import DATASETS

DATASETS["niederhoffer2017"]
```

## Notes

I corrected a typo in Table 2 (``plave`` -> ``plane``).
The correct spelling is "plane", as you can see it in the corresponding Topic in Appendix A.
