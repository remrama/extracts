> [!CAUTION]
> This package is a work in progress and under active development. Features have not been tested and may change without notice.

[![PyPI](https://img.shields.io/pypi/v/extracts.svg)](https://pypi.org/project/extracts)
[![Python Versions](https://img.shields.io/pypi/pyversions/extracts.svg)](https://pypi.org/project/extracts)
[![Downloads](https://static.pepy.tech/badge/extracts)](https://pepy.tech/projects/extracts)
[![License](https://img.shields.io/pypi/l/extracts.svg)](https://github.com/remrama/extracts/blob/main/LICENSE.txt)
[![Tests](https://github.com/remrama/extracts/actions/workflows/tests.yaml/badge.svg)](https://github.com/remrama/extracts/actions/workflows/tests.yaml)
[![Coverage](https://codecov.io/gh/remrama/extracts/branch/main/graph/badge.svg)](https://codecov.io/gh/remrama/extracts)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Repo Status](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

# extracts

Load tables, figures, and text manually extracted from published journal articles.

## Installation

```shell
pip install --upgrade extracts
```

## Usage

```python
import extracts

# List datasets known to the package
extracts.list_available_datasets()

# Fetch a table from a Zenodo-hosted dataset
df = extracts.fetch_barrett2020("table1")
```

## License

BSD. See [LICENSE](LICENSE) for details.
