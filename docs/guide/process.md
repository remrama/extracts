---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Processing and raw access

Every dataset fetcher takes a `process` keyword that controls how the downloaded file becomes a DataFrame.

## `process=True` (default) — parquet-cached, processed DataFrame

The fetcher runs a per-dataset `_processor(path)` closure that knows the right `index_col`, `header`, `skiprows`, etc. for that table. The resulting DataFrame is cached as a parquet file next to the raw download, so subsequent calls are near-instant and the parquet round-trips MultiIndex headers and dtypes losslessly.

```{code-cell} python
import extracts

df = extracts.fetch_hawkins2017("table1")
df.head()
```

```{code-cell} python
# MultiIndex columns are preserved through the parquet cache
df.columns.nlevels
```

## `process=False` — raw `pandas.read_table` with your kwargs

When you need different parsing options, set `process=False` and pass your own kwargs through:

```{code-cell} python
df = extracts.fetch_barrett2020("table1", process=False, index_col=0)
df.head()
```

The processor and parquet cache are bypassed; `**kwargs` are forwarded directly to {py:func}`pandas.read_table`.

## Just the file path

If you only need the location of the downloaded raw file (e.g. to hand off to another tool), call `fetch_path`:

```{code-cell} python
extracts.fetch_path("barrett2020", "table1")
```

`fetch_path` triggers the download if needed, but never parses or caches a processed version.

## Special fetchers

`fetch_text(dataset)` and `fetch_reference(dataset)` follow the same pattern:

- Default (`process=True`): return the parsed content (a string for text, a dict for the BibTeX reference).
- `process=False`: return the local `Path` to the raw file.

```{code-cell} python
entry = extracts.fetch_reference("barrett2020")
entry["fields"].get("title")
```
