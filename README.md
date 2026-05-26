[![PyPI](https://img.shields.io/pypi/v/extracts.svg)](https://pypi.org/project/extracts)
[![Python Versions](https://img.shields.io/pypi/pyversions/extracts.svg)](https://pypi.org/project/extracts)
[![License](https://img.shields.io/pypi/l/extracts.svg)](https://github.com/remrama/extracts/blob/main/LICENSE.txt)
[![Tests](https://github.com/remrama/extracts/actions/workflows/tests.yaml/badge.svg)](https://github.com/remrama/extracts/actions/workflows/tests.yaml)
[![Coverage](https://codecov.io/gh/remrama/extracts/branch/main/graph/badge.svg)](https://codecov.io/gh/remrama/extracts)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Repo Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

# extracts

Load tables, figures, and text manually extracted from published journal articles.

Documentation: [https://remrama.github.io/extracts](https://remrama.github.io/extracts)

## Installation

```shell
pip install --upgrade extracts
```

## Usage

```pycon
>>> import extracts
>>> extracts.list_available_datasets()
['barrett2020', 'cariola2010', 'cariola2014', 'hawkins2017', 'liwc1999', 'liwc2001', 'liwc2007', 'liwc2015', 'liwc22', 'mariani2023', 'mcnamara2015', 'meador2022', 'niederhoffer2017', 'paquet2020']
>>> df = extracts.fetch_barrett2020("table1")
>>> df.head()
                                      Pandemic M  Pandemic SD  Normative M  Normative SD     t          p
LIWC category and content examples
Positive emotions: love, nice, sweet        1.11         1.82         1.48          1.52  4.64  <.0001***
Negative emotions: hurt, ugly, nasty        2.31         3.32         1.40          1.47  9.14  <.0001***
Anxiety: worried, fearful, nervous          0.76         2.20         0.46          0.74  5.05  <.0001***
Anger: hate, furious, annoyed               0.42         1.32         0.31          0.64  2.66    .0078**
Sadness: crying, grief, sad                 0.46         1.37         0.27          0.63  4.55  <.0001***
```
