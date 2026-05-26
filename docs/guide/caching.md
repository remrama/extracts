---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Cache location

Every fetcher downloads its source files to a local cache and reuses them on subsequent calls. The cache root defaults to the OS-appropriate user cache directory (via [`pooch.os_cache`](https://www.fatiando.org/pooch/latest/api/generated/pooch.os_cache.html)):

| OS      | Default location                          |
| ------- | ----------------------------------------- |
| Linux   | `~/.cache/extracts/`                      |
| macOS   | `~/Library/Caches/extracts/`              |
| Windows | `%LOCALAPPDATA%\extracts\extracts\Cache\` |

Each dataset gets its own subdirectory under the root (e.g. `~/.cache/extracts/barrett2020/`), and within that subdirectory `extracts` keeps both the raw downloaded file (e.g. `table1.tsv`) and the parquet-cached parsed DataFrame (e.g. `table1.parquet`).

## Inspecting the current cache root

```{code-cell} python
import extracts

extracts.get_location()
```

## Overriding the cache root

There are two ways to point `extracts` at a different directory:

**1. Set the `EXTRACTS_DATA_DIR` environment variable** before launching Python:

```bash
export EXTRACTS_DATA_DIR=/path/to/my/cache  # Linux/macOS
$env:EXTRACTS_DATA_DIR = "C:\path\to\my\cache"  # PowerShell
```

**2. Call `extracts.set_location(path)` inside Python** to set the env var for the current process:

```{code-cell} python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    extracts.set_location(tmp)
    print(extracts.get_location())
```

`set_location` expands `~` and resolves the path to absolute form.

The environment variable is read on every fetch, so changes take effect immediately — no need to re-import `extracts`.
